import argparse
import os, json
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from tqdm import tqdm
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from prm_model import ProcessRewardModel
from prompts import build_prm_prompt

# ------------------------- Metric helpers ------------------------- #
def _f1_class(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int) -> float:
    assert y_true.shape == y_pred.shape
    pos = (y_true == pos_label)
    tp = np.sum(pos & (y_pred == pos_label))
    fp = np.sum(~pos & (y_pred == pos_label))
    fn = np.sum(pos & (y_pred != pos_label))

    if tp == 0 and (fp == 0 or fn == 0):
        return 0.0

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def _f1_for_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # error step(1)을 positive로 보는 F1
    return _f1_class(y_true, y_pred, pos_label=1)

def _f1_for_correct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # correct step(0)을 positive로 보는 F1
    return _f1_class(y_true, y_pred, pos_label=0)

def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    assert y_true.shape == y_pred.shape
    return float(np.mean(y_true == y_pred))

def _positive_negative_acc(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    positive_accuracy: correct steps 정확도 (label=0)
    negative_accuracy: error steps 정확도 (label=1)
    """
    mask_pos = (y_true == 0)
    mask_neg = (y_true == 1)
    pos_acc = float(np.mean(y_pred[mask_pos] == y_true[mask_pos])) if mask_pos.any() else 0.0
    neg_acc = float(np.mean(y_pred[mask_neg] == y_true[mask_neg])) if mask_neg.any() else 0.0
    return pos_acc, neg_acc

# ------------------------- Threshold tuning ------------------------- #
def tune_threshold( all_scores: List[List[float]], all_labels: List[List[int]], num_candidates: int = 200,) -> Dict[str, Any]:
    """
    Global threshold + direction (invert) tuning for PRMBench.
    all_scores: List over instances, each is step_scores [s_1, ..., s_T]
    all_labels: List over instances, each is step_labels [0/1] with same length.
    """
    assert len(all_scores) == len(all_labels), "scores/labels length mismatch"

    scores_flat = np.concatenate([np.asarray(s, dtype=np.float32) for s in all_scores], axis=0)
    labels_flat = np.concatenate([np.asarray(l, dtype=np.int32) for l in all_labels], axis=0)

    # Clean NaNs / infs if any
    mask = np.isfinite(scores_flat)
    scores_flat = scores_flat[mask]
    labels_flat = labels_flat[mask]

    if len(scores_flat) == 0:
        raise ValueError("No valid scores to tune threshold on.")

    # percentile range to avoid extreme outliers
    lo = float(np.percentile(scores_flat, 1))
    hi = float(np.percentile(scores_flat, 99))
    if lo == hi:
        # degenerate, just use mean
        thr_candidates = np.array([float(scores_flat.mean())])
    else:
        thr_candidates = np.linspace(lo, hi, num_candidates)

    best = {
        "threshold": None,
        "invert": False,
        "f1_neg": -1.0,
        "f1_pos": 0.0,
        "prm_score": 0.0,
        "acc": 0.0,
        "pos_acc": 0.0,
        "neg_acc": 0.0,
    }

    for invert in (False, True):
        for thr in thr_candidates:
            if invert:
                # error if score >= thr
                pred = (scores_flat >= thr).astype(np.int32)
            else:
                # error if score <= thr
                pred = (scores_flat <= thr).astype(np.int32)

            f1_neg = _f1_for_error(labels_flat, pred)
            if f1_neg > best["f1_neg"]:
                # f1_all = _f1_binary(labels_flat, pred)
                f1_pos = _f1_for_correct(labels_flat, pred)    # 정답 step을 잘 지켜주는지
                prm_score = 0.5 * (f1_neg + f1_pos)
                acc = _accuracy(labels_flat, pred)
                pos_acc, neg_acc = _positive_negative_acc(labels_flat, pred)
                best.update(
                    threshold=float(thr),
                    invert=invert,
                    f1_neg=float(f1_neg),
                    f1_pos=float(f1_pos),
                    prm_score=float(prm_score),
                    acc=float(acc),
                    pos_acc=float(pos_acc),
                    neg_acc=float(neg_acc),
                )

    return best

# ------------------------- PRM scoring ------------------------- #
@torch.no_grad()
def score_batch_steps( prm: ProcessRewardModel, tokenizer, questions: List[str], steps_list: List[List[str]], rw_token: str = "<RW>", max_length: Optional[int] = None,) -> List[List[float]]:
    """
    각 (question, steps) pair에 대해 PRM step scores를 계산해서
    [[s_1,...,s_T], ...] 형태로 반환.
    """
    assert len(questions) == len(steps_list)
    device = next(prm.parameters()).device

    prompts = [
        build_prm_prompt(tokenizer, q, steps, rw_token=rw_token, ds_name="prmbench")
        for q, steps in zip(questions, steps_list)
    ]

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True if max_length is not None else False,
        max_length=max_length,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # RW positions
    rw_id = tokenizer.convert_tokens_to_ids(rw_token)
    rw_positions = []
    for b in range(input_ids.size(0)):
        pos = (input_ids[b] == rw_id).nonzero(as_tuple=False).view(-1)
        rw_positions.append(pos)

    scores_flat, lengths = prm.scores_at_rw(input_ids, attention_mask, rw_positions=rw_positions)

    scores_flat = scores_flat.detach().cpu().numpy()
    scores_list: List[List[float]] = []
    offset = 0
    for L in lengths:
        if L == 0:
            scores_list.append([])
        else:
            seg = scores_flat[offset : offset + L]
            scores_list.append(seg.tolist())
            offset += L

    return scores_list

# ------------------------- Dataset utils ------------------------- #
def _auto_detect_columns(example: Dict[str, Any]) -> Tuple[str, str]:
    """
    PRMBench preview / 공식 데이터에서 step, label 컬럼 이름을 자동으로 추정.
    기본적으로:
      - label_col: 'label' / 'labels' / 'step_labels' 중 하나
      - step_col : 'steps' / 'modified_steps' / 'process' / 'reasoning_steps' 중 label 길이랑 맞는 것
    """
    col_names = list(example.keys())

    # label
    label_cands = ["label", "labels", "step_labels"]
    label_col = None
    for c in label_cands:
        if c in col_names:
            label_col = c
            break
    if label_col is None:
        raise ValueError(f"Cannot find label column in {col_names}")

    labels = example[label_col]
    if not isinstance(labels, (list, tuple)):
        raise ValueError(f"Label column {label_col} is not a list: {type(labels)}")

    # steps
    step_cands = ["steps", "modified_steps", "process", "reasoning_steps"]
    step_col = None
    for c in step_cands:
        if c in col_names:
            steps = example[c]
            if isinstance(steps, (list, tuple)) and len(steps) == len(labels):
                step_col = c
                break

    if step_col is None:
        raise ValueError(
            f"Cannot find step column matching label length in columns {col_names}. "
            f"Make sure one of {step_cands} exists and has same length as {label_col}."
        )

    return step_col, label_col

def collect_scores_and_labels( ds, prm: ProcessRewardModel, tokenizer, batch_size: int = 12, rw_token: str = "<RW>", max_length: Optional[int] = None, apply_sigmoid: bool = False,) -> Tuple[List[List[float]], List[List[int]]]:
    print("[INFO] Dataset columns:", ds.column_names)
    first_ex = ds[0]
    step_col, label_col = _auto_detect_columns(first_ex)
    print(f"[INFO] Using step_col='{step_col}', label_col='{label_col}'")

    all_scores: List[List[float]] = []
    all_labels: List[List[int]] = []

    N = len(ds)
    for i in tqdm(range(0, N, batch_size), desc="Scoring PRMBench", dynamic_ncols=True):
        sub = ds.select(range(i, min(i + batch_size, N)))
        questions = list(sub["question"]) if "question" in sub.column_names else list(sub["prompt"])
        steps_list = [list(x) for x in sub[step_col]]
        labels_list = [list(map(int, x)) for x in sub[label_col]]

        step_scores_batch = score_batch_steps( prm, tokenizer, questions, steps_list, rw_token=rw_token, max_length=max_length,)

        assert len(step_scores_batch) == len(labels_list)
        for scores, labels in zip(step_scores_batch, labels_list):
            if len(scores) != len(labels):
                # 길이 안 맞으면 잘라서 맞춰줌 (RW token misalignment 방지)
                L = min(len(scores), len(labels))
                scores = scores[:L]
                labels = labels[:L]
            if apply_sigmoid:
                scores = (1.0 / (1.0 + np.exp(-np.asarray(scores, dtype=np.float32)))).tolist()

            all_scores.append(scores)
            all_labels.append(labels)

    return all_scores, all_labels

# ------------------------- Category-wise eval ------------------------- #
def evaluate_split( ds, prm: ProcessRewardModel, tokenizer,batch_size: int = 8, rw_token: str = "<RW>",
    max_length: Optional[int] = None, apply_sigmoid: bool = False, category_field: Optional[str] = None,):
    """
    전체 + (옵션) category별 PRMBench 평가.
    category_field: 예) 'category' (NR, NCL, ES, ...), 없으면 전체만.
    """
    # 전체
    print("==== [OVERALL] ====")
    all_scores, all_labels = collect_scores_and_labels(ds, prm, tokenizer, batch_size=batch_size, rw_token=rw_token, max_length=max_length, apply_sigmoid=apply_sigmoid,)
    best_overall = tune_threshold(all_scores, all_labels)
    print(f"[TUNE][OVERALL] invert={best_overall['invert']}, thr={best_overall['threshold']:.6f}")
    print(
        f"[METRIC][OVERALL] "
        f"Neg-F1={best_overall['f1_neg']*100:.2f}, "
        f"Pos-F1={best_overall['f1_pos']*100:.2f}, "
        f"PRMScore={best_overall['prm_score']*100:.2f}, "
        f"Acc={best_overall['acc']*100:.2f}, "
        f"Pos-Acc={best_overall['pos_acc']*100:.2f}, "
        f"Neg-Acc={best_overall['neg_acc']*100:.2f}"
    )

    category_results: Dict[str, Dict[str, Any]] = {}
    if category_field is None:
        for cand in ["classification", "category"]:
            if cand in ds.column_names:
                category_field = cand
                break

    # category-wise
    if category_field is not None and category_field in ds.column_names:
        cats = sorted(set(ds[category_field]))
        print(f"\n[INFO] Category field '{category_field}' detected: {cats}")
        for cat in cats:
            sub_ds = ds.filter(lambda ex, c=cat: ex[category_field] == c)
            print(f"\n==== [CATEGORY={cat}] ====")
            scores_cat, labels_cat = collect_scores_and_labels(
                sub_ds,
                prm,
                tokenizer,
                batch_size=batch_size,
                rw_token=rw_token,
                max_length=max_length,
                apply_sigmoid=apply_sigmoid,
            )
            best_cat = tune_threshold(scores_cat, labels_cat)
            print(f"[TUNE][{cat}] invert={best_cat['invert']}, thr={best_cat['threshold']:.6f}")
            print(
                f"[METRIC][{cat}] "
                f"Neg-F1={best_cat['f1_neg']*100:.2f}, "
                f"Pos-F1={best_cat['f1_pos']*100:.2f}, "
                f"PRMScore={best_cat['prm_score']*100:.2f}, "
                f"Acc={best_cat['acc']*100:.2f}, "
                f"Pos-Acc={best_cat['pos_acc']*100:.2f}, "
                f"Neg-Acc={best_cat['neg_acc']*100:.2f}"
            )
            category_results[cat] = best_cat

    return best_overall, category_results

# ------------------------- Main ------------------------- #
def add_prmbench_columns(example: Dict[str, Any]) -> Dict[str, Any]:
    steps: List[str] = example["modified_process"]
    error_steps = set(example.get("error_steps", []))  # 1-based indices

    labels = []
    for i in range(len(steps)):  # i: 0-based
        step_idx_1based = i + 1
        is_error = 1 if step_idx_1based in error_steps else 0
        labels.append(is_error)

    return {
        "steps": steps,
        "labels": labels,
    }

def make_out_tag_from_prm_dir(prm_dir: str) -> str:
    p = os.path.normpath(prm_dir.rstrip("/"))
    last = os.path.basename(p)
    if last.lower() in {"final_model", "checkpoint", "checkpoints"}:
        parent = os.path.basename(os.path.dirname(p))
        return parent or last
    return last

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prm_dir", type=str, required=True, help="Directory of local PRM checkpoint (ProcessRewardModel.save_pretrained output)")
    p.add_argument("--batch_size", type=int, default=20)
    p.add_argument("--rw_token", type=str, default="<RW>")
    p.add_argument("--max_length", type=int, default=None, help="Optional max_length for tokenizer truncation")
    p.add_argument("--apply_sigmoid", action="store_true", help="Apply sigmoid to raw head outputs before threshold search.")
    p.add_argument("--category_field", type=str, default="classification", help="PRMBench error type column (NR, NCL, ...).")
    p.add_argument("--out_path", type=str, default="/home/leena/rs_prm/inference/prmbench")
    return p.parse_args()

def main():
    args = parse_args()
    prm_dir = args.prm_dir

    # Load tokenizer & PRM wrapper
    tokenizer = AutoTokenizer.from_pretrained(prm_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    prm = ProcessRewardModel.from_pretrained(prm_dir, tokenizer=tokenizer)
    prm.eval()

    # RW token sanity check
    if prm.rw_token is not None:
        rw_tok = prm.rw_token
    else:
        rw_tok = args.rw_token

    # Load PRMBench dataset
    ds = load_dataset("hitsmy/PRMBench_Preview", split="train")
    # ds = ds.select(range(10))  # for quick testing; remove this line for full eval
    ds = ds.map(add_prmbench_columns)
    print(f"[DATA] Loaded PRMBench preview dataset with {len(ds)} instances.")

    # Evaluate
    best_overall, category_results = evaluate_split(ds, prm, tokenizer, batch_size=args.batch_size, rw_token=rw_tok, max_length=args.max_length, apply_sigmoid=args.apply_sigmoid, category_field=args.category_field,)
    print("\n[FINAL] Best OVERALL (for this run):")
    print(best_overall)

    # output file save
    tag = make_out_tag_from_prm_dir(prm_dir)
    out_path = os.path.join(args.out_path, tag)

    result_payload = {
        "prm_dir": prm_dir,
        "dataset": "hitsmy/PRMBench_Preview",
        "split": "train",   # 필요하면 나중에 argparse로 뺄 수 있음
        "rw_token": rw_tok,
        "apply_sigmoid": args.apply_sigmoid,
        "metrics": {
            "overall": best_overall,
            "by_category": category_results,
        },
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] JSON results written to: {out_path}")


if __name__ == "__main__":
    main()
