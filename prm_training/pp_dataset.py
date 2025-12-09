import re, copy, random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

def join_steps_with_rw(steps: List[str], rw_token: str = "<RW>", step_sep: str = "\n\n") -> str:
    # "step1 + <RW> + step_sep + step2 + <RW> + ... + <RW>" 형태
    cleaned = [str(s).strip() for s in steps if s is not None and str(s).strip()]
    if not cleaned:
        return rw_token  # 최소 1개 <RW>는 남기도록
    return (f"{step_sep}{rw_token}{step_sep}").join(cleaned) + f"{step_sep}{rw_token}"

class PRMDataset(Dataset):
    """ (전체 스텝 + 각 스텝 뒤 <RW>) → targets=[r_1,..,r_n], rw_positions=[...]"""
    def __init__(
        self,
        entries: List[dict],
        tokenizer,
        *,
        reward_key: Optional[str] = None,
        add_rw_token: bool = True,
        rw_token: str = "<RW>",
        max_length: int = 1024,
        use_chat_template: bool = True,
        system_prompt: Optional[str] = None,
        step_sep: str = "\n\n",
    ):
        self.tok = tokenizer
        self.max_length = max_length
        self.add_rw_token = add_rw_token
        self.rw_token = rw_token
        self.use_chat_template = use_chat_template
        self.step_sep = step_sep
        self.system_prompt = system_prompt or ("You are Qwen-Math, a meticulous math tutor. Solve the given math problem step by step and conclude with one final answer.")

        # special token 등록
        if self.add_rw_token and (rw_token not in self.tok.get_vocab()):
            self.tok.add_special_tokens({"additional_special_tokens": [rw_token]})
        self.rw_id = self.tok.convert_tokens_to_ids(rw_token) if self.add_rw_token else None

        self.samples = []  
        for e in entries:
            q = e["question"]
            steps: List[str] = e["completion"]
            
            # 1) Reward Source
            reward_vec = None
            if reward_key is not None:
                if reward_key == "gold_label":
                    cm = e.get("correct_mask", None)
                    if isinstance(cm, list) and len(cm) == len(steps):
                        reward_vec = [float(x) for x in cm]   # 0/1 그대로
                    else:
                        continue
                else:
                    rv = e.get(reward_key, None)
                    if isinstance(rv, list) and len(rv) == len(steps):
                        reward_vec = [float(x) for x in rv]
                    else:
                        continue
            else:
                continue

            correct_mask: Optional[List[int]] = e.get("correct_mask")
            if correct_mask is not None and len(correct_mask) != len(steps):
                correct_mask = None

            # 2) assitant text generation
            kept_steps = self._pack_until_fit(q, steps)
            if not kept_steps:
                assistant_text = f"{self.step_sep}{self.rw_token}"
            else:
                assistant_text = join_steps_with_rw(kept_steps, self.rw_token, step_sep=self.step_sep)

            # 3) build prompt
            conv_str = self.build_prompt(q, assistant_text)

            # 4) encoding
            enc = self._encode(conv_str, self.max_length)

            # 5) RW token position & label 정렬
            rw_positions = self._find_all_rw_positions(enc)
            n_valid = min(len(rw_positions), len(reward_vec))
            if n_valid <= 0:
                continue
            rw_positions = rw_positions[:n_valid]
            targets = [float(x) for x in reward_vec[:n_valid]]
            is_correct = [int(x) for x in (correct_mask[:n_valid] if correct_mask else [0]*n_valid)]
            
            # save samples
            self.samples.append({
                "input_ids": enc["input_ids"],            # (T,)
                "attention_mask": enc["attention_mask"],  # (T,)
                "rw_positions": rw_positions,             # List[int]
                "targets": targets,                        # List[float]
                "is_correct": is_correct,              # List[int]
                "meta": {"q": q, "n_steps": n_valid},
            })

    # -------------------- helpers --------------------
    def build_prompt(self, question: str, assistant_text: str) -> str:
        if not self.use_chat_template:
            base = f"Problem: {question}\nSolution (step-by-step):\n"
            return base + assistant_text
        base = (
            "You are a careful math solver. Follow the steps methodically. Keep each step concise.\n"
            "At the end, output exactly one line in the format:\n"
            "The answer is: <final answer>\n\n"
            "Problem: {q}\n"
            "Solution: Let's think step by step.\n"
        )
        messages = [
            {"role": "user", "content": base.format(q=question)},
            {"role": "assistant", "content": assistant_text},
        ]
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    
    def _base_prompt(self, question: str) -> str:
        """assistant_text 없이 base prompt 문자열만 생성 (길이 점검용)"""
        if not self.use_chat_template:
            return f"Problem: {question}\nSolution (step-by-step):\n"
        base = (
            "You are a careful math solver. Follow the steps methodically. Keep each step concise.\n"
            "At the end, output exactly one line in the format:\n"
            "The answer is: <final answer>\n\n"
            "Problem: {q}\n"
            "Solution: Let's think step by step.\n"
        )
        messages = [
            {"role": "user", "content": base.format(q=question)},
            {"role": "assistant", "content": ""},  # 비운 assistant
        ]
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    
    def _pack_until_fit(self, question: str, steps: List[str]) -> List[str]:
        kept = steps[:]
        base_text = self._base_prompt(question)

        while kept:
            assistant_text = join_steps_with_rw(kept, self.rw_token, step_sep=self.step_sep)
            trial = base_text + assistant_text
            ids_full = self.tok(trial, truncation=False, return_tensors=None)["input_ids"]
            ids_trunc = self.tok(trial, max_length=self.max_length, truncation=True, padding=False, return_tensors=None)["input_ids"]

            if len(ids_trunc) >= len(ids_full):
                return kept
            kept.pop()

        return []
    
    def _encode(self, text: str, max_length: int):
        enc = self.tok(text, max_length=max_length, truncation=True, padding=False, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}

    def _find_rw_position(self, enc):
        ids = enc["input_ids"].tolist()
        if self.rw_id is not None and self.rw_id in ids:
            return ids.index(self.rw_id)
        # fallback: 마지막 토큰
        return len(ids) - 1

    def _find_all_rw_positions(self, enc: Dict[str, torch.Tensor]) -> List[int]:
        if self.rw_id is None:
            return []
        ids = enc["input_ids"]
        pos = (ids == self.rw_id).nonzero(as_tuple=False).flatten().tolist()
        return pos

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]

# -------------------- Collators --------------------
class PRMPackCollator:
    def __init__(self, pad_token_id: int, rw_token_id: int = None, strict: bool = False):
        self.pad = pad_token_id
        self.rw_id = rw_token_id
        self.strict = strict

    def _ensure_pack_schema(self, b: dict) -> dict:
        # 1) rw_positions 복구: (a) 이미 있으면 OK (b) 'rw_pos'면 리스트로 승격 (c) input_ids에서 <RW> 스캔
        if "rw_positions" not in b:
            if "rw_pos" in b:
                b["rw_positions"] = [int(b["rw_pos"])]
            elif self.rw_id is not None:
                ids = b["input_ids"]
                if torch.is_tensor(ids):
                    pos = (ids == self.rw_id).nonzero(as_tuple=False).flatten().tolist()
                else:
                    ids_t = torch.tensor(ids, dtype=torch.long)
                    pos = (ids_t == self.rw_id).nonzero(as_tuple=False).flatten().tolist()
                b["rw_positions"] = pos
            else:
                if self.strict:
                    raise KeyError("Sample has no 'rw_positions' and no 'rw_pos', and rw_token_id not provided.")
                b["rw_positions"] = []
        # 2) targets 복구
        if "targets" not in b:
            if "target" in b:
                b["targets"] = [float(b["target"])]
            else:
                if self.strict:
                    raise KeyError("Sample has no 'targets' or 'target'.")
                b["targets"] = [0.0] * len(b["rw_positions"])
        # 3) is_correct 복구
        if "is_correct" not in b:
            b["is_correct"] = [0] * min(len(b["rw_positions"]), len(b["targets"]))
        # 4) 길이 정합
        n = min(len(b["rw_positions"]), len(b["targets"]), len(b["is_correct"]))
        b["rw_positions"] = b["rw_positions"][:n]
        b["targets"]      = b["targets"][:n]
        b["is_correct"] = b["is_correct"][:n]
        return b

    def __call__(self, batch):
        batch = [self._ensure_pack_schema(b) for b in batch]

        def pad_stack(key, pad_val, dtype=torch.long):
            seqs = [b[key] for b in batch]
            seqs = [s if torch.is_tensor(s) else torch.tensor(s, dtype=dtype) for s in seqs]
            maxlen = max(x.size(0) for x in seqs)
            out = torch.full((len(seqs), maxlen), pad_val, dtype=seqs[0].dtype)
            for i, x in enumerate(seqs):
                out[i, :x.size(0)] = x
            return out
        
        input_ids      = pad_stack("input_ids", self.pad, dtype=torch.long)
        attention_mask = pad_stack("attention_mask", 0, dtype=torch.long)
       
        rw_positions   = [torch.tensor(b["rw_positions"], dtype=torch.long) for b in batch]
        targets_list   = [torch.tensor(b["targets"], dtype=torch.float32) for b in batch]
        cor_list     = [torch.tensor(b["is_correct"], dtype=torch.long) for b in batch]
        meta           = [b.get("meta", {}) for b in batch]

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            rw_positions=rw_positions,
            targets_list=targets_list,
            is_correct_list=cor_list,
            meta=meta,
        )

