from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, PreTrainedModel
)
from peft import LoraConfig, get_peft_model, PeftModel
from typing import Optional, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, json, math, random

HEAD_FNAME = "reward_head.pt"
META_FNAME = "wrapper_meta.json"

def _write_json(obj: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

class PRMRewardWrapper(nn.Module):
    def __init__(self, backbone: PreTrainedModel, rw_token_id: Optional[int] = None, rw_token: Optional[str] = "<RW>"):
        super().__init__()
        self.backbone = backbone
        hidden = backbone.config.hidden_size
        self.reward_head = nn.Linear(hidden, 1)
        self.rw_token_id = rw_token_id
        self.rw_token = rw_token

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        if hasattr(out, "hidden_states") and out.hidden_states is not None:
            hs = out.hidden_states[-1]        # (B,T,H) 마지막 레이어
        elif hasattr(out, "last_hidden_state"):
            hs = out.last_hidden_state        # 백업 경로
        else:
            raise RuntimeError("No hidden states on model outputs.")
        return hs

    @property
    def config(self):
        return self.backbone.config

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def resize_token_embeddings(self, n):
        return self.backbone.resize_token_embeddings(n)
    
    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable(**kwargs)
        if hasattr(self.backbone, "enable_input_require_grads"):
            self.backbone.enable_input_require_grads()
        if hasattr(self.backbone.config, "use_cache"):
            self.backbone.config.use_cache = False
        
    def _ensure_head_matches(self, hs: torch.Tensor):
        # reward_head를 hs의 device/dtype으로 동기화
        p = next(self.reward_head.parameters(), None)
        if p is None or p.device != hs.device or p.dtype != hs.dtype:
            self.reward_head.to(device=hs.device, dtype=hs.dtype)

    def save_pretrained(self, save_directory: str, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        meta = {
            "is_peft": isinstance(self.backbone, PeftModel),
            "rw_token": self.rw_token,
            "rw_token_id": int(self.rw_token_id) if self.rw_token_id is not None else None,
        }
        if isinstance(self.backbone, PeftModel):
            adapter_dir = os.path.join(save_directory, "adapter")
            self.backbone.save_pretrained(adapter_dir, **kwargs)  # 어댑터(config+weights) 저장
            meta["storage"] = "adapter"
        else:
            bb_dir = os.path.join(save_directory, "backbone")
            self.backbone.save_pretrained(bb_dir, **kwargs)
            meta["storage"] = "backbone"
        torch.save(self.reward_head.state_dict(), os.path.join(save_directory, HEAD_FNAME))
        _write_json(meta, os.path.join(save_directory, META_FNAME))

        # Check head state_dict consistency
        chk = torch.load(os.path.join(save_directory, HEAD_FNAME), map_location="cpu")
        for k, v in self.reward_head.state_dict().items():
            assert k in chk, f"Missing key in saved head: {k}"
            assert torch.allclose(v.cpu(), chk[k]), f"Mismatch in saved head param: {k}"
        print("[HEAD SAVE] round-trip state_dict check: OK")
    
    @classmethod
    def from_pretrained(cls, model_dir: str, *, tokenizer=None, base_model_name_or_path: Optional[str] = None, 
        device_map: Optional[str] = "auto", torch_dtype: Optional[torch.dtype] = None, rw_token: Optional[str] = "<RW>", **kwargs) -> "PRMRewardWrapper":
        meta_path = os.path.join(model_dir, META_FNAME)
        head_path = os.path.join(model_dir, HEAD_FNAME)
        assert os.path.exists(head_path), f"Missing reward head at {head_path}"
        meta = _read_json(meta_path) if os.path.exists(meta_path) else {}

        storage = meta.get("storage")  # "adapter" or "backbone"
        if storage == "adapter":
            adapter_dir = os.path.join(model_dir, "adapter")
            assert os.path.isdir(adapter_dir), f"Missing adapter dir: {adapter_dir}"
            # adapter_config.json에서 base_model_name_or_path를 추출
            acfg_path = os.path.join(adapter_dir, "adapter_config.json")
            acfg = _read_json(acfg_path)
            base_name = base_model_name_or_path or acfg.get("base_model_name_or_path")
            assert base_name is not None, "base_model_name_or_path not found (pass it explicitly)."
            # 베이스 로드 후 PEFT 로드
            base = AutoModelForCausalLM.from_pretrained(
                base_name,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
            if tokenizer is not None:
                tok_len = len(tokenizer)
                emb = base.get_input_embeddings()
                emb_len = getattr(emb, "num_embeddings", None)
                if emb_len is not None and emb_len != tok_len:
                    print(f"[INFO] resize_token_embeddings: base {emb_len} -> tok {tok_len}")
                    base.resize_token_embeddings(tok_len)
                    # (선택) config에도 반영
                    if hasattr(base.config, "vocab_size"):
                        base.config.vocab_size = tok_len
            backbone = PeftModel.from_pretrained(base, adapter_dir, device_map=device_map)
        elif storage == "backbone":
            bb_dir = os.path.join(model_dir, "backbone")
            assert os.path.isdir(bb_dir), f"Missing backbone dir: {bb_dir}"
            backbone = AutoModelForCausalLM.from_pretrained(
                bb_dir,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
            if tokenizer is not None:
                tok_len = len(tokenizer)
                emb = backbone.get_input_embeddings()
                emb_len = getattr(emb, "num_embeddings", None)
                if emb_len is not None and emb_len != tok_len:
                    print(f"[INFO] resize_token_embeddings: base {emb_len} -> tok {tok_len}")
                    backbone.resize_token_embeddings(tok_len)
                    if hasattr(backbone.config, "vocab_size"):
                        backbone.config.vocab_size = tok_len
        else:
            adapter_dir = os.path.join(model_dir, "adapter")
            bb_dir = os.path.join(model_dir, "backbone")
            if os.path.isdir(adapter_dir):
                base_name = base_model_name_or_path
                if base_name is None:
                    acfg = _read_json(os.path.join(adapter_dir, "adapter_config.json"))
                    base_name = acfg.get("base_model_name_or_path")
                base = AutoModelForCausalLM.from_pretrained(
                    base_name, device_map=device_map, torch_dtype=torch_dtype, trust_remote_code=True
                )
                if tokenizer is not None:
                    tok_len = len(tokenizer)
                    emb = base.get_input_embeddings()
                    emb_len = getattr(emb, "num_embeddings", None)
                    if emb_len is not None and emb_len != tok_len:
                        print(f"[INFO] resize_token_embeddings: base {emb_len} -> tok {tok_len}")
                        base.resize_token_embeddings(tok_len)
                        # (선택) config에도 반영
                        if hasattr(base.config, "vocab_size"):
                            base.config.vocab_size = tok_len
                backbone = PeftModel.from_pretrained(base, adapter_dir, device_map=device_map)
            elif os.path.isdir(bb_dir):
                backbone = AutoModelForCausalLM.from_pretrained(
                    bb_dir, device_map=device_map, torch_dtype=torch_dtype, trust_remote_code=True
                )
                if tokenizer is not None:
                    tok_len = len(tokenizer)
                    emb = backbone.get_input_embeddings()
                    emb_len = getattr(emb, "num_embeddings", None)
                    if emb_len is not None and emb_len != tok_len:
                        print(f"[INFO] resize_token_embeddings: base {emb_len} -> tok {tok_len}")
                        backbone.resize_token_embeddings(tok_len)
                        if hasattr(backbone.config, "vocab_size"):
                            backbone.config.vocab_size = tok_len
            else:
                raise FileNotFoundError(f"Neither adapter/ nor backbone/ exists in {model_dir}")

        # 래퍼 구성
        rw_tok = meta.get("rw_token", rw_token)
        rw_tok_id = meta.get("rw_token_id", None)
        if tokenizer is not None and rw_tok_id is None and rw_tok is not None:
            if rw_tok in tokenizer.get_vocab():
                rw_tok_id = int(tokenizer.convert_tokens_to_ids(rw_tok))

        wrapper = cls(backbone=backbone, rw_token_id=rw_tok_id, rw_token=rw_tok)

        dtype = next(backbone.parameters()).dtype
        wrapper.reward_head.to(dtype=dtype, device=backbone.device)
        sd = torch.load(head_path, map_location=backbone.device)
        wrapper.reward_head.load_state_dict(sd)

        # check reward head state_dict consistency
        with torch.no_grad():
            w = wrapper.reward_head.weight
            b = wrapper.reward_head.bias
            w_mean, w_std = float(w.mean()), float(w.std())
            b_mean = float(b.mean()) if b is not None else None
        print(f"[HEAD LOADED] w: mean={w_mean:.6f}, std={w_std:.6f} | b={b_mean}")
        
        if wrapper.rw_token is not None:
            assert wrapper.rw_token in tokenizer.get_vocab(), f"rw_token '{wrapper.rw_token}' not in vocab"
            if wrapper.rw_token_id is None:
                wrapper.rw_token_id = int(tokenizer.convert_tokens_to_ids(wrapper.rw_token))
            print(f"[RW] token='{wrapper.rw_token}', id={wrapper.rw_token_id}")

        if hasattr(wrapper.backbone.config, "use_cache"):
            wrapper.backbone.config.use_cache = False

        return wrapper

    def predict_rewards_at_rw(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, rw_positions: Optional[List[torch.Tensor]] = None) -> List[torch.Tensor]:
        hs = self(input_ids=input_ids, attention_mask=attention_mask)  # (B,T,H)
        self._ensure_head_matches(hs)
        # print(f"[PRED] hs shape: {tuple(hs.shape)} | head in={self.reward_head.in_features}", flush=True)
        
        B, T, H = hs.shape
        out = []
        if rw_positions is None:
            assert self.rw_token_id is not None, "rw_positions가 없으면 rw_token_id가 필요합니다."
            rw_positions = []
            for b in range(B):
                pos = (input_ids[b] == self.rw_token_id).nonzero(as_tuple=False).flatten()
                if pos.device != hs.device:
                    pos = pos.to(hs.device)
                rw_positions.append(pos)
        for b in range(B):
            pos = rw_positions[b]
            if pos.device != hs.device:
                pos = pos.to(hs.device)
            if pos.numel() == 0:
                out.append(torch.empty(0, device=hs.device))
                continue
            vecs = hs[b, pos, :]               # (K,H)
            scores = self.reward_head(vecs).squeeze(-1)  # (K,)
            out.append(scores)
        return out

