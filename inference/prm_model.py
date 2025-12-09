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

# Two Layer RewardHead #
class TwoLayerRewardHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        inner_mult: float | int = 1.0,   # e.g., 1.0 * hidden (또는 2.0 등)
        activation: str = "gelu",        # "gelu" | "tanh" | "silu"
        dropout: float = 0.1,
        layernorm: bool = False,
    ):
        super().__init__()
        inner = int(hidden_size * inner_mult)
        act = {"gelu": nn.GELU(), "tanh": nn.Tanh(), "silu": nn.SiLU()}[activation]
        mods = [nn.Linear(hidden_size, inner), act, nn.Dropout(dropout)]
        if layernorm:
            mods.insert(0, nn.LayerNorm(hidden_size))
        mods.append(nn.Linear(inner, 1))
        self.net = nn.Sequential(*mods)

    def forward(self, x):  # x: (..., H)
        return self.net(x)  # (..., 1)


class ProcessRewardModel(nn.Module):
    def __init__(self, backbone: PreTrainedModel, rw_token_id: Optional[int] = None, rw_token: Optional[str] = "<RW>",
            head_type: str = "mlp2",              # "linear" or "mlp2"
            head_inner_mult: float = 1.0,         # 1.0 * H (보편적), 2.0도 실전에서 자주 사용
            head_activation: str = "gelu",        # "gelu" | "tanh" | "silu"
            head_dropout: float = 0.1,
            head_layernorm: bool = False,):
        
        super().__init__()
        self.backbone = backbone
        self.rw_token_id = rw_token_id
        self.rw_token = rw_token
        hidden = backbone.config.hidden_size
        if head_type == "linear":
            self.reward_head = nn.Linear(hidden, 1)
        elif head_type == "mlp2":
            self.reward_head = TwoLayerRewardHead(
                hidden_size=hidden,
                inner_mult=head_inner_mult,
                activation=head_activation,
                dropout=head_dropout,
                layernorm=head_layernorm,
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        call_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            use_cache=False,
        )
        raw = self.backbone
        if hasattr(raw, "get_base_model"):
            try:
                raw = raw.get_base_model()
            except Exception: pass

        inner = getattr(raw, "model", None)
        if inner is not None and callable(getattr(inner, "forward", None)):
            out = inner(**call_kwargs)  # BaseModelOutputWithPast
            hs = getattr(out, "last_hidden_state", None)
            if hs is None:
                hs = out.hidden_states[-1]
            return hs  # (B,T,H)

        out = self.backbone(**call_kwargs)  # CausalLMOutputWithPast
        if hasattr(out, "hidden_states") and out.hidden_states is not None:
            return out.hidden_states[-1]  # (B,T,H)
        return out[0]

    @property
    def config(self):
        return self.backbone.config

    def scores_at_rw(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, rw_positions: Optional[List[torch.Tensor]] = None) -> tuple[torch.Tensor, list[int]]:
        if rw_positions is not None:
            last_rw_each = [int(pos.max().item()) + 1 if (pos is not None and pos.numel() > 0) else 1
                            for pos in rw_positions]
            T_trim = max(last_rw_each)
            input_ids = input_ids[:, :T_trim]
            attention_mask = attention_mask[:, :T_trim]
        
        hs = self(input_ids=input_ids, attention_mask=attention_mask)  # (B,T,H)
        self._ensure_head_matches(hs)
        B, T, H = hs.shape
        dev = hs.device

        if rw_positions is None:
            assert self.rw_token_id is not None, "rw_positions absent → need rw_token_id"
            rw_mask = (input_ids == self.rw_token_id)                   # (B,T) bool
        else:
            rw_mask = torch.zeros((B, T), dtype=torch.bool, device=dev)
            for b, pos in enumerate(rw_positions):
                if pos.numel() > 0:
                    rw_mask[b, pos.to(dev)] = True
        
        vecs = hs[rw_mask]                                             # (N_rw, H)
        scores_flat = self.reward_head(vecs).squeeze(-1)               # (N_rw,)
        lengths = rw_mask.sum(dim=1).tolist()
        return scores_flat, lengths
    
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
        p = next(self.reward_head.parameters(), None)
        if p is None or p.device != hs.device or p.dtype != hs.dtype:
            self.reward_head.to(device=hs.device, dtype=hs.dtype)

    def save_pretrained(self, save_directory: str, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        meta = {
            "is_peft": isinstance(self.backbone, PeftModel),
            "rw_token": self.rw_token,
            "rw_token_id": int(self.rw_token_id) if self.rw_token_id is not None else None,
            "head": {
                "type": "mlp2" if isinstance(self.reward_head, TwoLayerRewardHead) else "linear",
                "inner_mult": getattr(self.reward_head, "inner_mult", 1.0),
                "activation": getattr(self.reward_head, "activation_name", "gelu"),
                "dropout": getattr(self.reward_head, "dropout_p", 0.1),
                "layernorm": getattr(self.reward_head, "use_layernorm", False),
            },
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
        device_map: Optional[str] = "auto", torch_dtype: Optional[torch.dtype] = None, rw_token: Optional[str] = "<RW>", **kwargs) -> "ProcessRewardModel":
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

        dev = next(backbone.parameters()).device
        wrapper = cls(backbone=backbone, rw_token_id=rw_tok_id, rw_token=rw_tok)
        dtype = next(backbone.parameters()).dtype
        wrapper.reward_head.to(dtype=dtype, device=dev)
        sd = torch.load(head_path, map_location=dev)
        wrapper.reward_head.load_state_dict(sd)

        # check reward head state_dict consistency
        # with torch.no_grad():
        #     w = wrapper.reward_head.weight
        #     b = wrapper.reward_head.bias
        #     w_mean, w_std = float(w.mean()), float(w.std())
        #     b_mean = float(b.mean()) if b is not None else None
        # print(f"[HEAD LOADED] w: mean={w_mean:.6f}, std={w_std:.6f} | b={b_mean}")
        
        with torch.no_grad():
            last_linear = None
            for m in wrapper.reward_head.modules():
                if isinstance(m, nn.Linear):
                    last_linear = m
            if last_linear is not None:
                w_mean = float(last_linear.weight.mean())
                w_std  = float(last_linear.weight.std())
                b_mean = float(last_linear.bias.mean()) if last_linear.bias is not None else None
                print(f"[HEAD LOADED] last_linear: w mean={w_mean:.6f}, std={w_std:.6f} | b={b_mean}")
            else:
                # Linear가 아닌 커스텀 헤드일 때: 전체 파라미터 통계
                params = [p.view(-1) for p in wrapper.reward_head.parameters()]
                if params:
                    allp = torch.cat(params)
                    print(f"[HEAD LOADED] param mean={float(allp.mean()):.6f}, std={float(allp.std()):.6f}")
                else:
                    print("[HEAD LOADED] reward_head has no parameters?")
        
        if (tokenizer is not None) and (wrapper.rw_token is not None):
            assert wrapper.rw_token in tokenizer.get_vocab(), f"rw_token '{wrapper.rw_token}' not in vocab"
            if wrapper.rw_token_id is None:
                wrapper.rw_token_id = int(tokenizer.convert_tokens_to_ids(wrapper.rw_token))
            print(f"[RW] token='{wrapper.rw_token}', id={wrapper.rw_token_id}")

        if hasattr(wrapper.backbone.config, "use_cache"):
            wrapper.backbone.config.use_cache = False

        return wrapper
