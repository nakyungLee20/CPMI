import re, math, random
from fractions import Fraction
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Tuple, Type
from decimal import Decimal, InvalidOperation
import os, json, torch
import argparse
from collections import defaultdict, Counter
from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    AutoTokenizer,
)
from datasets import concatenate_datasets, load_dataset, get_dataset_config_names
from torch.utils.data import DataLoader
from vllm import LLM, SamplingParams

from prompts import build_prompt, build_prm_prompt
from gsm_scorer import GSM8KScorer
from math_scorer import MATHScorer
from mmlu_scorer import MMLUScorer
from omni_scorer import OMNIScorer
from ob_scorer import OBScorer
# from aime_scorer import AIMEScorer

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
# os.environ["CUDA_VISIBLE_DEVICES"]= "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##############################################################################
# 2. Dataset/Scorer loaders & field mapping                 #
##############################################################################
FIELD_MAP: Dict[str, Tuple[str, str]] = {
    # dataset name  : (question_field, answer_field, choices_field(optional via key lookup))
    "gsm8k": ("question", "answer"),
    "math": ("problem", "answer"), # "solution" for MATH, "answer" for MATH-500
    "omni": ("problem", "answer"),
    "ob": ("question", "final_answer"),
    "aime": ("problem", "answer"),
    "mmlu": ("question", "answer"),  # choices provided separately
}

def load_olympiadbench_english(split: str = "train"):
    all_cfgs = get_dataset_config_names("Hothan/OlympiadBench")
    en_cfgs = [cfg for cfg in all_cfgs if "_en_" in cfg or cfg.endswith("_en")]
    ds_list = []
    for cfg in en_cfgs:
        try:
            ds = load_dataset("Hothan/OlympiadBench", cfg, split=split)
            ds_list.append(ds)
        except Exception as e:
            print(f"⚠️  {cfg} load failed: {e}")
    if len(ds_list) == 0:
        raise ValueError("Fail to load English configs")
    full_ds = concatenate_datasets(ds_list)
    return full_ds

def get_loader(ds_name: str, split: str, batch_size: int):
    """Return dataset with standardized fields: question, answer, and optional choices (mmlu)."""
    if ds_name == "math":
        # ds = load_dataset("HuggingFaceTB/MATH", "all", split=split)
        ds = load_dataset("HuggingFaceH4/MATH-500", split=split)
    elif ds_name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split=split)
    elif ds_name == "omni":
        ds = load_dataset("KbsdJames/Omni-MATH", split=split)
        ds = ds.select(range(3100, len(ds)))
        print(f"Evaluate {len(ds)} omnimath dataset!", flush=True)
    elif ds_name == "ob":
        ds = load_olympiadbench_english(split)
    elif ds_name == "aime":
        ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
    elif ds_name == "mmlu":
        ds = load_dataset("TIGER-Lab/MMLU-STEM", split=split)
    else:
        raise ValueError(f"Unsupported dataset {ds_name}")

    q_key, a_key = FIELD_MAP[ds_name]

    def _std(ex):
        out = {"question": ex[q_key], "answer": ex[a_key]}
        if ds_name == "mmlu" and ("choices" in ex):
            out["choices"] = ex["choices"]
        return out

    ds = ds.map(_std, remove_columns=[])
    return ds, len(ds)

def resolve_scorer(ds_name: str):
    if not ds_name or not isinstance(ds_name, str):
        raise ValueError("ds_name must be a non-empty string")

    raw = ds_name.strip()
    ds_upper = re.sub(r'[^0-9A-Za-z]+', '_', raw).upper()   # e.g., 'gsm8k' -> 'GSM8K'
    instance_name = ds_upper + "Scorer"

    sym = globals().get(instance_name)
    required = ("extract_gold", "extract_pred", "grade")
    if all(callable(getattr(sym, m, None)) for m in required):
        return sym

    raise ImportError(f"Could not resolve scorer for dataset '{instance_name}'. ")

import torch.nn as nn
HEAD_FNAME = "reward_head.pt"
def ensure_head_matches_checkpoint(prm, model_dir: str, *, activation="gelu", dropout=0.1, layernorm=False):
    head_path = os.path.join(model_dir, HEAD_FNAME)
    sd = torch.load(head_path, map_location=next(prm.parameters()).device)
    own = prm.reward_head.state_dict()

    # 이미 키/shape가 같으면 그대로 사용
    same_keys = set(sd.keys()) == set(own.keys())
    same_shapes = same_keys and all(tuple(sd[k].shape) == tuple(own[k].shape) for k in own)
    if same_shapes:
        return prm  # 끝

    # checkpoint로부터 구조 추정 (linear vs 2-layer)
    hidden = prm.backbone.config.hidden_size
    # 마지막 Linear weight 찾기
    last_w_name = max([k for k in sd.keys() if k.endswith(".weight")])
    last_w_shape = sd[last_w_name].shape  # (out_features, in_features)

    # linear: (1, hidden), mlp2: 마지막 (1, inner), 첫 Linear (hidden, inner)
    if last_w_shape[0] == 1 and last_w_shape[1] == hidden:
        # 1층
        new_head = nn.Linear(hidden, 1)
    else:
        inner = last_w_shape[1]
        try:
            from prm_model import TwoLayerRewardHead  # 네임스페이스 동일 가정
        except Exception:
            # 이 파일 안이라면 위에 정의돼있음
            from __main__ import TwoLayerRewardHead
        inner_mult = float(inner) / float(hidden)
        new_head = TwoLayerRewardHead(
            hidden_size=hidden,
            inner_mult=inner_mult,
            activation=activation,
            dropout=dropout,
            layernorm=layernorm,
        )

    # dtype/device 정렬 + strict 로드
    dev = next(prm.parameters()).device
    dtype = next(prm.parameters()).dtype
    prm.reward_head = new_head.to(device=dev, dtype=dtype)
    prm.reward_head.load_state_dict(sd, strict=True)
    return prm

##############################################################################
# Step parsing utilities #
##############################################################################
# def split_steps(text: str) -> List[str]:
#     """Parse a model completion into a list of steps (strings).
#     Priority:
#       1) Existing <extra_0> separators
#       2) Blank-line separation ("\n\n")
#       3) Lines starting with "Step k:" markers
#       4) Fallback: entire reasoning (before the Answer line) as one step
#     """
#     if "<extra_0>" in text:
#         parts = [p.strip() for p in text.split("<extra_0>") if p.strip()]
#     else:
#         # Cut off after the explicit Answer line if present
#         lines = text.splitlines()
#         body_lines: List[str] = []
#         for ln in lines:
#             if ANSWER_LINE_RE.match(ln):
#                 break
#             body_lines.append(ln)
#         body = "\n".join(body_lines)
#         # 2) try blank line separation
#         if "\n\n" in body:
#             parts = [p.strip() for p in body.split("\n\n") if p.strip()]
#         else:
#             # 3) try explicit step markers
#             parts_tmp: List[str] = []
#             cur: List[str] = []
#             for ln in body.splitlines():
#                 if STEP_RE.match(ln):
#                     if cur:
#                         parts_tmp.append(" ".join(cur).strip())
#                         cur = []
#                     cur.append(STEP_RE.sub(r"\1", ln).strip())
#                 else:
#                     cur.append(ln.strip())
#             if cur:
#                 parts_tmp.append(" ".join(cur).strip())
#             parts = [p for p in parts_tmp if p]
#     if not parts:
#         parts = [text.strip()]
#     # print("Split steps:", parts, flush=True)
#     return parts

BOXED_RE = r"\\boxed\{([^}]+)\}"
STEP_RE = re.compile(r"^\s*step\s*\d+\s*:\s*(.*)$", re.IGNORECASE)
ANSWER_LINE_RE = re.compile(r"^\s*(?:answer\s*:|the\s+answer\s+is)\b", re.IGNORECASE)

def split_steps(text: str) -> List[str]:
    # 0) <extra_0> 최우선
    if "<extra_0>" in text:
        parts = [p.strip() for p in text.split("<extra_0>") if p.strip()]
        return parts or [text.strip()]
    # 1) Answer 라인 이전만 사용
    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    body_lines = []
    for ln in lines:
        if ANSWER_LINE_RE.match(ln):
            break
        body_lines.append(ln)
    body = "\n".join(body_lines).strip()
    if not body:
        return []
    # 2) Step k: 기준으로 우선 split 라인 스캔하면서 'Step k:'가 나오면 새 스텝 시작
    steps = []
    cur = []
    found_step = False
    for ln in body.splitlines():
        m = STEP_RE.match(ln)
        if m:
            found_step = True
            # 이전 스텝 flush
            if cur:
                s = "\n".join(cur).strip()
                if s:
                    steps.append(s)
                cur = []
            # 현재 라인의 'Step k:' 뒤 텍스트를 첫 줄로 시작
            first = m.group(1).strip()
            cur = [first] if first else []
        else:
            cur.append(ln.rstrip())
    if cur:
        s = "\n".join(cur).strip()
        if s:
            steps.append(s)
    if found_step and steps:
        return steps
    # 3) 빈 줄 기준 split
    if "\n\n" in body:
        parts = [p.strip() for p in body.split("\n\n") if p.strip()]
        if parts:
            return parts
    # 4) 단일 개행 기준 split
    if "\n" in body:
        parts = [p.strip() for p in body.split("\n") if p.strip()]
        if parts:
            return parts
    # 5) 폴백: 전체를 한 스텝
    return [body]

##############################################################################
# Naive Generation #
##############################################################################
def _chunk(lst, size):
    for i in range(0, len(lst), size):
        yield i, lst[i:i+size]

def batched_generate_vllm(items: List[Dict[str, Any]], llm: LLM, tokenizer, *, dataset_name: str, model_type: str = "", model_name: Optional[str] = None, n: int = 1, 
                          temperature: float = 0.2, top_p: float = 0.9, max_tokens: int = 512, seed: Optional[int] = 123) -> List[List[str]]:
    assert n >= 1
    do_sample = (n > 1) or (temperature and temperature > 1e-8)

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_id = tokenizer.eos_token_id
    stop_token_ids = [tid for tid in (im_end_id, eos_id) if tid is not None]

    # prompts = [to_chat_prompt(tokenizer, q, eval_style=eval_style) for q in questions]
    prompts = build_prompt(items, tokenizer, dataset_name=dataset_name)

    top_k = None
    min_p = None
    if model_name and "Qwen/Qwen3" in model_name:
        top_k = 20
        min_p = 0.0
        temperature = 0.6
        top_p = 0.95

    sp_kwargs = dict(
        temperature=(temperature if do_sample else 0.0),
        top_p=(top_p if do_sample else 1.0),
        max_tokens=max_tokens,
        n=n,
        seed=seed,
        repetition_penalty=1.05,
        skip_special_tokens=True,
        stop_token_ids=stop_token_ids,
    )

    if top_k is not None:
        sp_kwargs["top_k"] = top_k
    # if min_p is not None:
    #     sp_kwargs["min_p"] = min_p
    
    sp = SamplingParams(**sp_kwargs)
    outs = llm.generate(prompts, sp, use_tqdm=False)

    result: List[List[str]] = []
    for out in outs:
        gens_i = [o.text.strip() for o in out.outputs]
        # for idx, gen in enumerate(gens_i):
        #     print(f"[Debug] {idx}th-Generations:", gen, flush=True)
        result.append(gens_i)
    return result

def evaluate_vllm(dataset, llm: LLM, tokenizer, *, ds_name: str, model_type: str = "", model_name: Optional[str] = None, limit: Optional[int] = None, n: int = 1, temperature: float = 0.2, top_p: float = 0.9, seed: int = 123, max_tokens: int = 512, 
                  batch_size: int = 32, save_incorrect_path: Optional[str] = None, scorer: Any = None) -> Tuple[float, List[Dict[str, Any]], List[Dict[str, Any]]]:
    total = 0
    correct = 0
    logs: List[Dict[str, Any]] = []
    incorrect_samples: List[Dict[str, Any]] = []
    N = len(dataset)
    if limit is not None:
        N = min(N, limit)
    
    for start, batch in _chunk(list(range(N)), batch_size):
        items = [dataset[i] for i in batch]
        if ds_name == "mmlu":
            # Prepare golds/choices
            gold_indices: List[int] = [int(it["answer"]) for it in items]  # 1-based in TIGER MMLU
            choices_batch: List[List[str]] = [list(it.get("choices", [])) for it in items]
            gens_batch = batched_generate_vllm(items, llm, tokenizer, dataset_name=ds_name, model_type=model_type, model_name=model_name, n=n,
                                               temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed)
            for j, i_ex in enumerate(batch):
                q = items[j]["question"]
                choices = choices_batch[j]
                gold_idx = gold_indices[j]
                gens = gens_batch[j]
                pred_label = scorer.extract_pred(gens[0], choices)
                is_correct = scorer.grade(gens[0], gold_idx, choices)
                total += 1
                correct += int(is_correct)
                logs.append({
                    "idx": i_ex,
                    "question": q,
                    "choices": choices,
                    "gold_index": gold_idx,
                    "pred_label": pred_label,
                    "gen": gens[0],
                    "correct_first": bool(is_correct),
                })
                if not is_correct:
                    incorrect_samples.append({
                        "idx": i_ex,
                        "question": q,
                        "choices": choices,
                        "gold_index": gold_idx,
                        "pred": pred_label,
                        "gen": gens[0],
                    })
        else:
            # 1) collect batch inputs
            gens_batch: List[List[List[str]]] = batched_generate_vllm(
                items, llm, tokenizer,
                dataset_name=ds_name, model_type=model_type, model_name=model_name, n=n,
                temperature=temperature, top_p=top_p,
                max_tokens=max_tokens, seed=seed
            )
            # 3) score each item in batch
            for j, i_ex in enumerate(batch):
                q = items[j]["question"]
                gold = scorer.extract_gold(items[j]["answer"])
                gens = gens_batch[j]
                preds = [scorer.extract_pred(t) for t in gens]
                is_correct = scorer.grade(preds[0], gold)
                total += 1
                correct += int(is_correct)
                logs.append({
                    "idx": i_ex,
                    "question": q,
                    "gold": gold,
                    "gens": gens,
                    "preds": preds,
                    "correct_first": bool(is_correct),
                })
                if not is_correct:
                    incorrect_samples.append({
                        "idx": i_ex,
                        "question": q,
                        "gold": gold,
                        "pred_chosen": preds[0] if preds else "",
                        "preds_all": preds,
                        "gens_all": gens,
                    })

        if (total % 20) == 0:
            acc = 100.0 * correct / total
            print(f"[{total}/{N}] running acc = {acc:.2f}%", flush=True)

    acc = 100.0 * correct / max(total, 1)
    print(f"{ds_name} Accuracy = {acc:.2f}%  on {total} examples.")

    if save_incorrect_path:
        os.makedirs(os.path.dirname(save_incorrect_path) or ".", exist_ok=True)
        with open(save_incorrect_path, "w", encoding="utf-8") as f:
            json.dump(incorrect_samples, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(incorrect_samples)} incorrect samples to: {save_incorrect_path}")

    return acc, logs, incorrect_samples

##############################################################################
# PRM scoring wrappers #
##############################################################################
def make_step_rewards(logits: torch.Tensor, token_masks: torch.Tensor, positive_index: int = 1) -> List[List[float]]:
    # 0) list/tuple 방어
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    # 1) dtype/shape 정리
    token_masks = token_masks.to(torch.bool)
    if token_masks.ndim == 1:
        token_masks = token_masks.unsqueeze(0)  # (T,) -> (1,T)
    # 2) 케이스별 확률로 변환: probs를 (B,T,C') 또는 (B,T)로 맞추고, 아래에서 인덱싱
    if logits.ndim == 3:
        # (B,T,C)
        if logits.size(-1) == 1:
            probs = torch.sigmoid(logits)          # (B,T,1)
        else:
            probs = torch.softmax(logits, dim=-1)  # (B,T,C>=2)
    elif logits.ndim == 2:
        # 모호: (B,T) 또는 (T,C)
        if logits.shape == token_masks.shape:
            # (B,T): single-logit -> sigmoid
            probs = torch.sigmoid(logits).unsqueeze(-1)  # (B,T,1)
        else:
            # (T,C): 배치=1로 간주
            probs = torch.softmax(logits, dim=-1).unsqueeze(0)  # (1,T,C)
            if token_masks.size(0) != 1:
                token_masks = token_masks[:1]
    elif logits.ndim == 1:
        # (T,) -> 배치=1, 채널=1로 간주
        probs = torch.sigmoid(logits).unsqueeze(0).unsqueeze(-1)  # (1,T,1)
        if token_masks.size(0) != 1:
            token_masks = token_masks[:1]
    else:
        # 예외 케이스: 전부 펴서 (1,T,1)로 맞춤
        probs = logits.view(1, -1, 1)
        if token_masks.size(0) != 1:
            token_masks = token_masks[:1]

    # 길이 정합성
    T_model = probs.size(-2) if probs.ndim >= 2 else probs.size(0)
    if token_masks.size(1) != T_model:
        T_eff = min(token_masks.size(1), T_model)
        token_masks = token_masks[:, :T_eff]
        if probs.ndim == 3: probs = probs[:, :T_eff, :]
        elif probs.ndim == 2: probs = probs[:, :T_eff]
        else: probs = probs[:T_eff]
    
    B, T = token_masks.shape
    out_scores: List[List[float]] = []

    for i in range(B):
        p_i = probs[i] 
        pos = token_masks[i].nonzero(as_tuple=True)[0] # (S,) 
        if pos.numel() and p_i.size(0) != token_masks.size(1):
            pos = pos[pos < p_i.size(0)] 
        if pos.numel() == 0:
            out_scores.append([])
            continue
        if p_i.ndim == 2:  # (T,C')
            col = positive_index if p_i.size(-1) > positive_index else (p_i.size(-1) - 1)
            scores_i = p_i.index_select(0, pos)[:, col]  # (S,)
        elif p_i.ndim == 1:  # (T,)
            scores_i = p_i.index_select(0, pos)          # (S,)
        else:
            # 매우 예외적인 경우, 일단 평탄화
            p_i = p_i.reshape(T, -1)
            col = min(positive_index, p_i.size(-1) - 1)
            scores_i = p_i.index_select(0, pos)[:, col]
        out_scores.append(scores_i.detach().cpu().tolist())
    return out_scores

def aggregate(scores: List[float], how: str = "mean") -> float:
    if not scores:
        return 0.0
    if how == "mean":
        return float(sum(scores) / len(scores))
    if how == "last":
        return float(scores[-1])
    if how == "sum":
        return float(sum(scores))
    if how == "median":
        s = sorted(scores)
        m = len(s) // 2
        return float((s[m] if len(s) % 2 else 0.5*(s[m-1] + s[m])))
    return float(sum(scores) / len(scores))

def score_candidates_with_prm(prm_model, prm_tokenizer, question: str, candidates: List[str], rw_token: str = "<RW>", ds_name: str = "", model_type: str = "", choices: Optional[List[str]] = None, model: Optional[Any]= None):
    if prm_tokenizer.pad_token_id is None:
        prm_tokenizer.pad_token = prm_tokenizer.eos_token

    conv_strs = [build_prm_prompt(prm_tokenizer, question, split_steps(txt), rw_token=rw_token, ds_name=ds_name, model_type=model_type, choices=choices) for txt in candidates]
    enc = prm_tokenizer(conv_strs, return_tensors="pt", padding=True, truncation=True, max_length=prm_tokenizer.model_max_length)
    if hasattr(prm_model, "get_input_embeddings"):
        dev = prm_model.get_input_embeddings().weight.device
    else:
        dev = next(prm_model.parameters()).device
    input_ids = enc["input_ids"].to(dev)
    attention_mask = enc["attention_mask"].to(dev)

    rw_id = getattr(prm_model, "rw_token_id", None)
    if rw_id is None:
        if rw_token in prm_tokenizer.get_vocab():
            rw_id = prm_tokenizer.convert_tokens_to_ids(rw_token)
        else:
            raise ValueError(f"RW token '{rw_token}' not found and model.rw_token_id is None.")

    rw_positions: List[torch.Tensor] = []
    for i in range(input_ids.size(0)):
        pos = (input_ids[i] == rw_id).nonzero(as_tuple=False).squeeze(-1).cpu()
        rw_positions.append(pos)

    with torch.no_grad():
        scores_flat, lengths = prm_model.scores_at_rw(input_ids=input_ids, attention_mask=attention_mask, rw_positions=rw_positions)

    all_step_scores: List[List[float]] = []
    offset = 0
    for i, L in enumerate(lengths):
        Li = int(L)
        step_scores_i = scores_flat[offset:offset + Li].detach().cpu().tolist()
        offset += Li
        if len(split_steps(candidates[i])) != Li:
            print(f"[WARN] steps({len(split_steps(candidates[i]))}) != RW tokens({Li}) for sample {i}", flush=True)
        all_step_scores.append(step_scores_i)
        
    return all_step_scores

def pick_best_by_prm(prm_model,prm_tokenizer,question: str, candidates: List[str], agg: str = "mean", rw_token: str = "<RW>", ds_name: str = "", model_type: str ="", choices: Optional[List[str]] = None,) -> Tuple[int, List[List[float]], List[float]]:
    step_scores = score_candidates_with_prm(prm_model=prm_model, prm_tokenizer=prm_tokenizer,
            question=question, candidates=candidates, rw_token=rw_token, ds_name=ds_name, model_type=model_type, choices=choices,)
    agg_scores_custom = [aggregate(s, how=agg) for s in step_scores]
    best_idx = int(max(range(len(agg_scores_custom)), key=lambda i: agg_scores_custom[i]))
    return best_idx, step_scores, agg_scores_custom

def evaluate_dataset_bon_vllm(dataset, llm: LLM, policy_tokenizer, prm_model, prm_tokenizer, *, ds_name: str, model_type: str = "", model_name: Optional[str] = None, limit: Optional[int] = None, n: int = 8, temperature: float = 0.6, top_p: float = 0.95, seed: int = 123, 
    max_tokens: int = 512, batch_size: int = 32, agg: str = "mean", save_incorrect_path: Optional[str] = None, scorer: Any = None,) -> Tuple[float, List[Dict[str, Any]], List[Dict[str, Any]]]:
    total = 0
    correct = 0
    logs: List[Dict[str, Any]] = []
    incorrect_samples: List[Dict[str, Any]] = []
    N = len(dataset)
    if limit is not None:
        N = min(N, limit)
    all_indices = list(range(N))

    for start, batch_idx in _chunk(all_indices[:N], batch_size):
        items = [dataset[i] for i in batch_idx]
        gens_batch = batched_generate_vllm(items, llm, policy_tokenizer, dataset_name=ds_name, model_type=model_type, model_name=model_name, n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed,)

        if ds_name == "mmlu":
            choices_batch: List[List[str]] = [list(it.get("choices", [])) for it in items]
            gold_indices: List[int] = [int(it["answer"]) for it in items]
            for j, i_ex in enumerate(batch_idx):
                q = items[j]["question"]
                choices = choices_batch[j]
                gold_idx = gold_indices[j]
                gens: List[str] = gens_batch[j]
                # pick best by PRM (text only)
                best_idx, step_scores, agg_scores = pick_best_by_prm(prm_model, prm_tokenizer, q, gens, agg=agg, rw_token="<RW>", ds_name="mmlu", model_type=model_type, choices=choices,)
                chosen_gen = gens[best_idx]
                pred_label = scorer.extract_pred(chosen_gen, choices)
                is_correct = scorer.grade(chosen_gen, gold_idx, choices)
                total += 1
                correct += int(is_correct)
                logs.append({
                    "idx": i_ex,
                    "question": q,
                    "choices": choices,
                    "gold_index": gold_idx,
                    "gens": gens,
                    "prm_step_scores": step_scores,
                    "prm_agg_scores": agg_scores,
                    "chosen_idx": best_idx,
                    "pred_chosen": pred_label,
                    "correct": bool(is_correct),
                })
                if not is_correct:
                    incorrect_samples.append({
                        "idx": i_ex,
                        "question": q,
                        "choices": choices,
                        "gold_index": gold_idx,
                        "pred_chosen": pred_label,
                        "gens_all": gens,
                        "prm_step_scores": step_scores,
                        "prm_agg_scores": agg_scores,
                    })
        else:
            # Non-MMLU path
            golds: List[str] = [scorer.extract_gold(it["answer"]) for it in items]
            for j, i_ex in enumerate(batch_idx):
                q = items[j]["question"]
                gold = golds[j]
                gens: List[str] = gens_batch[j]
                preds: List[str] = [scorer.extract_pred(t) for t in gens]
                best_idx, step_scores, agg_scores = pick_best_by_prm(prm_model, prm_tokenizer, q, gens, agg=agg, rw_token="<RW>", model_type=model_type, ds_name=ds_name,)
                chosen_pred = preds[best_idx] if preds else ""
                is_correct = scorer.grade(chosen_pred, gold)
                total += 1
                correct += int(is_correct)
                logs.append({
                    "idx": i_ex,
                    "question": q,
                    "gold": gold,
                    "gens": gens,
                    "preds": preds,
                    "prm_step_scores": step_scores,
                    "prm_agg_scores": agg_scores,
                    "chosen_idx": best_idx,
                    "pred_chosen": chosen_pred,
                    "correct": bool(is_correct),
                })
                if not is_correct:
                    incorrect_samples.append({
                        "idx": i_ex,
                        "question": q,
                        "gold": gold,
                        "pred_chosen": chosen_pred,
                        "preds_all": preds,
                        "gens_all": gens,
                        "prm_step_scores": step_scores,
                        "prm_agg_scores": agg_scores,
                    })

        if (total % 20) == 0:
            acc = 100.0 * correct / max(total, 1)
            print(f"[{total}/{N}] running BoN acc = {acc:.2f}% (agg={agg}, N={n})", flush = True)

    acc = 100.0 * correct / max(total, 1)
    print(f"{ds_name} BoN Accuracy = {acc:.2f}% on {total} examples. (agg={agg}, N={n})")

    if save_incorrect_path:
        os.makedirs(os.path.dirname(save_incorrect_path) or ".", exist_ok=True)
        with open(save_incorrect_path, "w", encoding="utf-8") as f:
            json.dump(incorrect_samples, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(incorrect_samples)} incorrect samples to: {save_incorrect_path}")

    return acc, logs, incorrect_samples

##############################################################################
# Majority Voting (with optional PRM weights)
##############################################################################
def _softmax_weights(scores: List[float], temperature: float = 1.0) -> List[float]:
    if not scores:
        return []
    # Numerical stability via max-subtraction; protect very small temperature
    t = max(1e-6, float(temperature))
    max_s = max(scores)
    exps = [math.exp((s - max_s) / t) for s in scores]
    z = float(sum(exps))
    if z <= 0.0 or not math.isfinite(z):
        return [1.0 / len(scores)] * len(scores)
    return [e / z for e in exps]

def _build_candidates_with_scores(question: str, gens: List[str], prm_model, prm_tokenizer, ds_name: str, agg: str = "mean", rw_token: str = "<RW>", scorer: Any = None, compute_prm: bool = True, choices: Optional[List[str]] = None,) -> Tuple[List[Dict[str, Any]], List[float]]:
    cands: List[Dict[str, Any]] = []
    # if compute_prm:
    #     # PRM scores for all candidates at once
    #     step_scores = score_candidates_with_prm(prm_model, prm_tokenizer, question, gens, rw_token=rw_token, ds_name=ds_name, choices=choices,)
    #     for txt, step_s in zip(gens, step_scores):
    #         if choices is not None:
    #             ans = scorer.extract_pred(txt, choices)
    #             ans_norm = (ans or "").strip().upper()
    #         else:
    #             ans = scorer.extract_pred(txt)
    #             ans_norm = scorer.normalize_number(ans) if hasattr(scorer, "normalize_number") else (ans or "")
    #         cands.append({
    #             "body": txt,
    #             "answer": ans,
    #             "answer_norm": ans_norm,
    #             "steps": split_steps(txt),
    #             "prm_step_scores": step_s,
    #         })
    #     return cands
    # else:
    for txt in gens:
        if choices is not None:
            ans = scorer.extract_pred(txt, choices)
            ans_norm = (ans or "").strip().upper()
        else:
            ans = scorer.extract_pred(txt)
            ans_norm = scorer.normalize_number(ans) if hasattr(scorer, "normalize_number") else (ans or "")
        cands.append({
            "body": txt,
            "answer": ans,
            "answer_norm": ans_norm,
            "steps": split_steps(txt),
        })
    return cands

def choose_by_majority(cands: List[Dict[str, Any]], use_prm_weights: bool = False, temperature: float = 1.0, tie_break: str = "prm", ignore_empty: bool = True,) -> Tuple[int, Dict[str, Any]]:
    if not cands:
        return -1, {"reason": "no_candidates"}
        # (idx, cand)로 보관해서 원래 인덱스 추적을 명확히
    def _has_nonempty_answer(c):
        return bool((c.get("answer_norm") or c.get("answer")))

    vote_pool: List[Tuple[int, Dict[str, Any]]] = [
        (i, c) for i, c in enumerate(cands)
        if (not ignore_empty) or _has_nonempty_answer(c)
    ]

    # PRM을 실제로 사용할 수 있는지: 요청 + 모든 후보가 prm_score 보유
    has_prm = bool(use_prm_weights) and all(("prm_score" in c) for c in cands)
    # 모두 비어있을 때의 폴백
    if not vote_pool:
        if has_prm:
            best_idx = int(max(range(len(cands)),
                               key=lambda i: float(cands[i].get("prm_score", float("-inf")))))
            return best_idx, {"reason": "fallback_all_empty_prm"}
        else:
            return 0, {"reason": "fallback_all_empty"}
    # 가중치: PRM 가능 시 softmax(PRM), 아니면 균등
    if has_prm:
        weights = _softmax_weights(
            [float(c.get("prm_score", 0.0)) for _, c in vote_pool],
            temperature=temperature
        )
    else:
        weights = [1.0] * len(vote_pool)
    vote_by_answer: Dict[str, float] = defaultdict(float)
    members: Dict[str, List[int]] = defaultdict(list)

    # 투표 집계 (vote_pool 순서를 기준으로 index 매칭)
    for j, (global_i, cand) in enumerate(vote_pool):
        key = cand.get("answer_norm") or cand.get("answer") or "__EMPTY__"
        w = float(weights[j])
        vote_by_answer[key] += w
        members[key].append(global_i)
    max_votes = max(vote_by_answer.values(), default=0.0)
    winners = [ans for ans, v in vote_by_answer.items() if abs(v - max_votes) < 1e-12]
    if not winners:
        if has_prm:
            fallback = int(max(range(len(cands)),
                               key=lambda i: float(cands[i].get("prm_score", float("-inf")))))
            return fallback, {"reason": "fallback_no_winner_prm"}
        else:
            return 0, {"reason": "fallback_no_winner"}
    # 타이브레이크
    if len(winners) == 1:
        win_ans = winners[0]
    else:
        if tie_break == "prm" and has_prm:
            def best_prm_for_answer(ans_key: str) -> float:
                return max(float(cands[i].get("prm_score", float("-inf"))) for i in members[ans_key])
            win_ans = max(winners, key=best_prm_for_answer)
        else:
            # 다수결 동률 시, 처음 등장한 답(안정적)을 선택
            win_ans = min(winners, key=lambda a: min(members[a]))
    # 승자 버킷 내에서 최종 후보 선택
    idxs = members[win_ans]
    if has_prm:
        chosen_idx = max(idxs, key=lambda i: float(cands[i].get("prm_score", float("-inf"))))
    else:
        chosen_idx = min(idxs)  # 먼저 나온 후보로 안정적 선택

    diag = {
        "votes": {k: float(v) for k, v in vote_by_answer.items()},
        "use_prm_weights": bool(use_prm_weights),
        "temperature": float(temperature),
        "tie_break": tie_break,
        "winning_answer": win_ans,
        "chosen_idx": int(chosen_idx),
        "chosen_score": float(cands[chosen_idx].get('prm_score', float('nan'))),  # 안전
    }
    return int(chosen_idx), diag

def evaluate_dataset_majority_vllm(dataset, llm: LLM, policy_tokenizer, prm_model, prm_tokenizer, *, ds_name: str, model_name: Optional[str] = None, limit: Optional[int] = None, n: int = 8, temperature: float = 0.6, top_p: float = 0.95, seed: int = 123, 
    max_tokens: int = 512, batch_size: int = 32, agg: str = "mean", use_prm_weights: bool = False, maj_temp: float = 1.0,     # softmax temperature for PRM vote weights
    tie_break: str = "prm", save_incorrect_path: Optional[str] = None, scorer: Any = None,) -> Tuple[float, List[Dict[str, Any]], List[Dict[str, Any]]]:
    
    total = 0
    correct = 0
    logs: List[Dict[str, Any]] = []
    incorrect_samples: List[Dict[str, Any]] = []

    N = len(dataset)
    if limit is not None:
        N = min(N, limit)
    all_indices = list(range(N))

    for start, batch_idx in _chunk(all_indices[:N], batch_size):
        items = [dataset[i] for i in batch_idx]
        gens_batch = batched_generate_vllm(items, llm, policy_tokenizer, dataset_name=ds_name, model_name=model_name, n=n,
                                           temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed)

        if ds_name == "mmlu":
            choices_batch: List[List[str]] = [list(it.get("choices", [])) for it in items]
            gold_indices: List[int] = [int(it["answer"]) for it in items]
            for j, i_ex in enumerate(batch_idx):
                q = items[j]["question"]
                choices = choices_batch[j]
                gold_idx = gold_indices[j]
                gens: List[str] = gens_batch[j]

                compute_prm = bool(use_prm_weights and (prm_model is not None) and (prm_tokenizer is not None))
                cands = _build_candidates_with_scores(q, gens, prm_model, prm_tokenizer, ds_name=ds_name, agg=agg, rw_token="<RW>", scorer=scorer, compute_prm=compute_prm, choices=choices,)
                tie_break_eff = tie_break if compute_prm else 'first'
                chosen_idx, diag = choose_by_majority(cands, use_prm_weights=compute_prm, temperature=maj_temp, tie_break=tie_break_eff)
                if chosen_idx < 0:
                    chosen_idx = int(max(range(len(cands)), key=lambda i: cands[i].get('prm_score', float('-inf')))) if cands else -1

                chosen_pred = cands[chosen_idx]["answer"] if (0 <= chosen_idx < len(cands)) else ""
                # mmlu.grade expects full text, but we have candidates bodies, so use that
                chosen_body = cands[chosen_idx]["body"] if (0 <= chosen_idx < len(cands)) else ""
                is_correct = scorer.grade(chosen_body, gold_idx, choices)
                total += 1
                correct += int(is_correct)
                logs.append({
                    "idx": i_ex,
                    "question": q,
                    "choices": choices,
                    "gold_index": gold_idx,
                    "gens": gens,
                    "cands": cands,
                    "chosen_idx": chosen_idx,
                    "pred_chosen": chosen_pred,
                    "majority_diag": diag,
                    "correct": bool(is_correct),
                })
                if not is_correct:
                    incorrect_samples.append({
                        "idx": i_ex,
                        "question": q,
                        "choices": choices,
                        "gold_index": gold_idx,
                        "pred_chosen": chosen_pred,
                        "gens_all": gens,
                        "cands": cands,
                        "majority_diag": diag,
                    })
        else:
            # Non-MMLU datasets
            golds: List[str] = [scorer.extract_gold(it["answer"]) for it in items]
            for j, i_ex in enumerate(batch_idx):
                q = items[j]["question"]
                gold = golds[j]
                gens: List[str] = gens_batch[j]

                compute_prm = bool(use_prm_weights and (prm_model is not None) and (prm_tokenizer is not None))
                cands= _build_candidates_with_scores(q, gens, prm_model, prm_tokenizer, ds_name=ds_name, agg=agg, rw_token="<RW>", scorer=scorer, compute_prm=compute_prm)
                tie_break_eff = tie_break if compute_prm else 'first'
                chosen_idx, diag = choose_by_majority(cands, use_prm_weights=compute_prm, temperature=maj_temp, tie_break=tie_break_eff)
                if chosen_idx < 0:
                    chosen_idx = int(max(range(len(cands)), key=lambda i: cands[i].get('prm_score', float('-inf')))) if cands else -1

                chosen_pred = cands[chosen_idx]["answer"] if (0 <= chosen_idx < len(cands)) else ""
                is_correct = scorer.grade(chosen_pred, gold)
                total += 1
                correct += int(is_correct)
                logs.append({
                    "idx": i_ex,
                    "question": q,
                    "gold": gold,
                    "gens": gens,
                    "cands": cands,
                    "chosen_idx": chosen_idx,
                    "pred_chosen": chosen_pred,
                    "majority_diag": diag,
                    "correct": bool(is_correct),
                })
                if not is_correct:
                    incorrect_samples.append({
                        "idx": i_ex,
                        "question": q,
                        "gold": gold,
                        "pred_chosen": chosen_pred,
                        "gens_all": gens,
                        "cands": cands,
                        "majority_diag": diag,
                    })

        if (total % 20) == 0:
            acc = 100.0 * correct / max(total, 1)
            print(f"[{total}/{N}] running MAJ acc = {acc:.2f}% (N={n}, use_prm_weights={use_prm_weights}, maj_temp={maj_temp})", flush=True)

    acc = 100.0 * correct / max(total, 1)
    print(f"Final {ds_name} Majority Accuracy = {acc:.2f}% on {total} examples. (N={n}, use_prm_weights={use_prm_weights}, maj_temp={maj_temp})")

    if save_incorrect_path:
        os.makedirs(os.path.dirname(save_incorrect_path) or ".", exist_ok=True)
        with open(save_incorrect_path, "w", encoding="utf-8") as f:
            json.dump(incorrect_samples, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(incorrect_samples)} incorrect samples to: {save_incorrect_path}")

    return acc, logs, incorrect_samples

##############################################################################
# Main APIs #
##############################################################################
import re, os
def _slugify(s: str) -> str:
    s = re.sub(r"[^\w\-]+", "-", s.strip())
    s = re.sub(r"-{2,}", "-", s)
    return s.strip("-") or "untitled"

def make_prm_tag(path: str) -> str:
    p = os.path.normpath(path)
    last = os.path.basename(p)
    parent = os.path.basename(os.path.dirname(p))
    container_names = {"final_model", "checkpoint", "checkpoints"}
    tag = parent if last.lower() in container_names else last
    return _slugify(tag)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='mmlu', choices=['gsm8k', 'math', 'mmlu', 'ob', "omni", 'aime'])
    ap.add_argument('--prm_dir', type=str, default='', help="Path to PRM checkpoint dir.")
    ap.add_argument('--limit', type=int, default=10) # None for all
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--temperature', type=float, default=0.6)
    ap.add_argument('--top_p', type=float, default=0.95)
    ap.add_argument('--max_new_tokens', type=int, default=2500)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--model_type', type=str, default='contr_', help="Prompt style flags: '', 'cmi_*', 'prompt_*', 'contr_*'")
    ap.add_argument('--norm_type', type=str, default='norm', choices=['cal', 'norm'])
    ap.add_argument('--infer_type', type=str, default='bon', choices=['bon', 'single', 'maj'])
    ap.add_argument('--bon_agg', type=str, default='mean', choices=['mean', 'last', 'sum', 'median'])
    ap.add_argument('--maj_tie_break', type=str, default='first', choices=['prm','first'])
    ap.add_argument('--maj_use_prm_weights', action='store_true')
    args = ap.parse_args()

    # Load models
    model_name = "Qwen/Qwen3-4B-base"   # "mistralai/Mathstral-7B-v0.1", "Qwen/Qwen3-4B-base", "Qwen/Qwen2.5-Math-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    policy = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.55, 
        max_model_len=3500,
        enforce_eager=False,
        enable_prefix_caching=True,  
        distributed_executor_backend="mp",  
    )

    # Load Datasets
    datas, total = get_loader(args.dataset, "test" if args.dataset != "ob" else "train", args.batch_size)
    # Load Scorer
    SCORER = resolve_scorer(args.dataset)

    # Prepare PRM (needed for BoN and Majority)
    need_prm = (args.infer_type == "bon" or (args.infer_type == "maj" and (args.maj_use_prm_weights or args.maj_tie_break == "prm")))
    prm_model = None
    prm_tokenizer = None
    prm_tag = "noPRM"
    if need_prm:
        if not args.prm_dir:
            raise ValueError("PRM is required but --prm_dir was not provided.")
        path = args.prm_dir
        if not os.path.isdir(path):
            raise FileNotFoundError(f"PRM checkpoint dir not found: {path}")
        prm_tag = make_prm_tag(path)
        
        prm_tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
        if prm_tokenizer.pad_token_id is None and prm_tokenizer.eos_token_id is not None:
            prm_tokenizer.pad_token_id = prm_tokenizer.eos_token_id
        from prm_model import ProcessRewardModel  # type: ignore
        prm_model = ProcessRewardModel.from_pretrained(path, tokenizer=prm_tokenizer, torch_dtype=torch.bfloat16).eval()
        prm_model = ensure_head_matches_checkpoint(prm_model, path)
        print(f"[INFO] Loaded PRM from: {path}")
    
    incorr_dir = "/home/leena/rs_prm/analysis/hard_80k"
    os.makedirs(incorr_dir, exist_ok=True)
    incorr_path = os.path.join(incorr_dir, f"{args.dataset}_{args.infer_type}_{prm_tag}_test.json")
    print(f"[INFO] Incorrect samples will be saved to: {incorr_path}")
    
    # Evaluation
    if args.infer_type == 'single':
        print(f"Evaluating {args.dataset} with single generation...", flush=True)
        acc, logs, incorrect = evaluate_vllm(datas, policy, tokenizer, ds_name=args.dataset, model_type=args.model_type, model_name=model_name,
            n=1, temperature=args.temperature, top_p=args.top_p,
            seed=args.seed, max_tokens=args.max_new_tokens, limit=args.limit, 
            batch_size=args.batch_size, save_incorrect_path=incorr_path, scorer=SCORER
        )
    elif args.infer_type == 'bon':
        print(f"Evaluating {args.dataset} with PRM best-of-N...", flush=True)
        acc, logs, incorrect = evaluate_dataset_bon_vllm(datas, policy, tokenizer, prm_model, prm_tokenizer, ds_name=args.dataset, model_type=args.model_type, model_name=model_name,
            n=args.n, temperature=args.temperature, top_p=args.top_p, limit=args.limit, 
            seed=args.seed, max_tokens=args.max_new_tokens, batch_size=args.batch_size,
            agg=args.bon_agg, save_incorrect_path=incorr_path, scorer=SCORER,
        )
    elif args.infer_type == 'maj':
        print(f"Evaluating {args.dataset} with Majority Voting...", flush=True)
        acc, logs, incorrect = evaluate_dataset_majority_vllm(datas, policy, tokenizer, prm_model, prm_tokenizer, ds_name=args.dataset, model_name=model_name,
            n=args.n, temperature=args.temperature, top_p=args.top_p, limit=args.limit, 
            seed=args.seed, max_tokens=args.max_new_tokens, batch_size=args.batch_size,
            agg=args.bon_agg, use_prm_weights=False,
            maj_temp=1.0, tie_break=args.maj_tie_break,
            save_incorrect_path=incorr_path, scorer=SCORER,
        )
    else:
        raise ValueError(f"Unknown inference type: {args.infer_type}")
