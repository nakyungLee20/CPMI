import os, re, json, math, random
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple
import sympy as sp
from sympy.parsing.latex import parse_latex
import numpy as np

from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from math_scorer import MATHSCORER, MathScorer

##############################################################################
# Prompt helpers (reuse your GSM8K style; tweak Answer boxing for MATH)
##############################################################################
SYSTEM_PROMPT = (
    "You are Qwen-Math, a meticulous math tutor. "
    "Solve problems with clear, numbered steps and give only ONE final answer line."
)
FEWSHOT_EXAMPLES: List[Dict[str, Any]] = []

def format_fewshot_block(example: Dict[str, Any]) -> str:
    steps = example.get("steps") or []
    body = "Problem: " + example["question"].strip() + "\n"
    for i, s in enumerate(steps, 1):
        s = s if s.strip().lower().startswith("step") else f"Step {i}: {s}"
        body += s.strip() + "\n"
    body += f"Answer: {example['answer']}\n"
    return body.strip()

def build_user_prompt(question: str) -> str:
    header = (
        "Please reason step by step. Follow this EXACT format:\n"
        "Step 1: <short reasoning>\n"
        "Step 2: <short reasoning>\n"
        "...\n"
        "Answer: <final simplified answer>\n"
        "Constraints:\n"
        "- Keep each step concise, one idea per line.\n"
        "- Put the final answer in LaTeX box: `Answer: \\boxed{...}` if possible.\n"
        "- Do not include extra explanations after the final Answer line.\n"
    )
    parts = [header]
    if FEWSHOT_EXAMPLES:
        parts.append("Here are solved examples:\n")
        for ex in FEWSHOT_EXAMPLES:
            parts.append(format_fewshot_block(ex))
        parts.append("\nNow solve this new problem in the same format.\n")

    parts.append("Problem: " + question.strip())
    return "\n".join(parts).strip()

def to_chat_prompt(tokenizer, question: str, eval_style: str = "default") -> str:
    if eval_style == "qwen_eval":
        user = f"{question.strip()}\n\nPlease reason step by step, and put your final answer within `Answer: \\boxed{{}}`."
    else:
        user = build_user_prompt(question)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def batched_generate_vllm(questions: List[str], llm: LLM, tokenizer, n: int = 1, temperature: float = 0.2, top_p: float = 0.9, max_tokens: int = 768, seed: Optional[int] = 123, eval_style: Optional[str] = "default",) -> List[List[str]]:
    assert n >= 1
    do_sample = (n > 1) or (temperature and temperature > 1e-8)

    prompts = [to_chat_prompt(tokenizer, q, eval_style=eval_style) for q in questions]
    sp = SamplingParams(
        temperature=(temperature if do_sample else 0.0),
        top_p=(top_p if do_sample else 1.0),
        max_tokens=max_tokens,
        n=n,
        seed=seed,
        repetition_penalty=1.05,
        skip_special_tokens=True,
        stop=["<|im_end|>", "<|endoftext|>"]
    )
    outs = llm.generate(prompts, sp, use_tqdm=False)

    result: List[List[str]] = []
    for out in outs:
        gens_i = [o.text.strip() for o in out.outputs]
        result.append(gens_i)
    return result

##############################################################################
# Batch evaluation over MATH dataset
##############################################################################
def _chunk(lst, size):
    for i in range(0, len(lst), size):
        yield i, lst[i:i+size]

def evaluate_math_vllm(dataset, llm: LLM, tokenizer, limit: Optional[int] = None, n: int = 1, temperature: float = 0.2, top_p: float = 0.9, seed: int = 123, max_tokens: int = 2048, batch_size: int = 16, save_incorrect_path: Optional[str] = None,) -> Tuple[float, List[Dict[str, Any]], List[Dict[str, Any]]]:
    
    math_extract_gold = MATHSCORER.extract_gold
    math_extract_pred = MATHSCORER.extract_pred
    grade_math_answer = MATHSCORER.grade
    
    total = 0
    correct = 0
    logs: List[Dict[str, Any]] = []
    incorrect_samples: List[Dict[str, Any]] = []

    N = len(dataset)
    if limit is not None:
        N = min(N, limit)

    for start, idxs in _chunk(list(range(N)), batch_size):
        qs: List[str] = []
        recs: List[Dict[str, Any]] = []
        for k in idxs:
            rec = dataset[k]
            recs.append(rec)
            q = rec.get("problem") or rec.get("question") or ""
            qs.append(q)

        gens_batch: List[List[List[str]]] = batched_generate_vllm(qs, llm, tokenizer, n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed,)

        for j, k in enumerate(idxs):
            rec = recs[j]
            q = qs[j]
            gold = math_extract_gold(rec)
            gens = gens_batch[j]
            preds = [math_extract_pred(t) for t in gens]

            is_correct = grade_math_answer(preds[0], gold)
            total += 1
            correct += int(is_correct)

            logs.append({
                "idx": k,
                "question": q,
                "gold": gold,
                "gens": gens,
                "preds": preds,
                "correct_first": bool(is_correct),
            })

            if not is_correct:
                incorrect_samples.append({
                    "idx": k,
                    "question": q,
                    "gold_sol": rec.get("solution") or "",
                    "gold": gold,
                    "pred_chosen": preds[0] if preds else "",
                    "preds_all": preds,
                    "gens_all": gens,
                })

        if (total % 20) == 0:
            acc = 100.0 * correct / total
            print(f"[{total}/{N}] running acc = {acc:.2f}%")

    acc = 100.0 * correct / max(total, 1)
    print(f"MATH Accuracy = {acc:.2f}% on {total} examples.", flush=True)

    if save_incorrect_path:
        os.makedirs(os.path.dirname(save_incorrect_path) or ".", exist_ok=True)
        with open(save_incorrect_path, "w", encoding="utf-8") as f:
            json.dump(incorrect_samples, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(incorrect_samples)} incorrect samples to: {save_incorrect_path}")

    return acc, logs, incorrect_samples


if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-Math-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="bfloat16", 
        tensor_parallel_size=4,
        gpu_memory_utilization=0.90, 
        max_model_len=4096,
        enforce_eager=False,              # CUDA Graphs 활용
        enable_prefix_caching=True,  
        distributed_executor_backend="mp",      
    )

    # Load Datasets
    dataset = load_dataset("HuggingFaceTB/MATH", "all", split="test")
    # Harmonize records to dicts (for gold extraction)
    def _as_dict(x):
        if isinstance(x, dict):
            return x
        try:
            return dict(x)
        except Exception:
            return {k: x[k] for k in x.keys()}  # type: ignore

    dataset = dataset.map(lambda r: _as_dict(r))

    # Evaluation
    print("Starting evaluation on MATH dataset...", flush=True)
    incorr_path = "/home/leena/prm_shaping/analysis/incorr_math3_0820.json"
    acc, logs, incorrect = evaluate_math_vllm(dataset, llm=llm, tokenizer=tokenizer, n=1,
        # limit=20,  # None for full eval
        max_tokens=2048,
        batch_size=128,
        temperature=0.7, top_p=0.8,
        save_incorrect_path=incorr_path,
    )

    # Optionally dump logs next to incorrect file if provided
    path_logs = os.path.splitext(incorr_path)[0] + "_logs.json"
    with open(path_logs, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
    print(f"Saved full logs to: {path_logs}")
