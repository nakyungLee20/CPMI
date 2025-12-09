import os
import json
import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from prompts import build_prm_prompt
from prm_model import ProcessRewardModel

# -------------------- Debug helpers --------------------
def _debug_rw_markers(tokenizer, prm, problems, steps_list, rw_token):
    rw_tok = getattr(prm, "rw_token", rw_token) or "<RW>"
    dbg_prompts = [
        build_prm_prompt(tokenizer, q, steps, rw_token=rw_tok, ds_name="")
        for q, steps in zip(problems, steps_list)
    ]
    dev = next(prm.parameters()).device
    dbg_enc = tokenizer(dbg_prompts, return_tensors="pt", padding=True, truncation=False)
    dbg_ids = dbg_enc["input_ids"].to(dev)
    marker_ids = tokenizer.encode(rw_tok, add_special_tokens=False)

    rw_counts = []
    for b in range(dbg_ids.size(0)):
        pos = _find_marker_positions(dbg_ids[b], marker_ids)
        rw_counts.append(int(pos.numel()))
    rc = np.array(rw_counts, dtype=int)
    print(f"[DEBUG] rw_token='{rw_tok}', marker_ids={marker_ids}", flush=True)
    print(f"[DEBUG] RW markers per sample (first batch): "
          f"min={rc.min()}, mean={rc.mean():.2f}, max={rc.max()}, zero={int((rc==0).sum())}/{len(rc)}", flush=True)

def _debug_first_batch_scores(step_scores_list, threshold, use_prob):
    flat = [float(x) for xs in step_scores_list for x in xs]
    if len(flat) == 0:
        print("[DEBUG][WARN] No step scores produced in first batch.")
        return
    arr = np.asarray(flat, dtype=float)
    probs = 1.0 / (1.0 + np.exp(-arr))
    thr = (0.5 if threshold is None else float(threshold)) if use_prob \
          else (0.0 if threshold is None else float(threshold))
    pos_rate = float((probs > thr).mean()) if use_prob else float((arr > thr).mean())
    lens = np.array([len(x) for x in step_scores_list])
    print(f"[DEBUG] step_scores lens (first batch): "
          f"min={lens.min()}, mean={lens.mean():.2f}, max={lens.max()}")
    print(f"[DEBUG] logits: mean={arr.mean():.3f}, std={arr.std():.3f}, "
          f"min={arr.min():.3f}, max={arr.max():.3f}")
    print(f"[DEBUG] probs : mean={probs.mean():.3f}, std={probs.std():.3f}, "
          f"min={probs.min():.3f}, max={probs.max():.3f}")
    print(f"[DEBUG] positive_rate@thr={thr} (use_prob={use_prob}) = {pos_rate:.3f}")

    for j in range(min(10, len(step_scores_list))):
        ss = step_scores_list[j]
        show_k = len(ss)
        print(f"[DEBUG] sample#{j}: first {show_k} logits={np.round(np.array(ss[:show_k]), 3).tolist()}")
        if show_k > 0:
            print(f"[DEBUG] sample#{j}: first {show_k} probs = "
                  f"{np.round(1/(1+np.exp(-np.array(ss[:show_k]))), 3).tolist()}", flush=True)

# -------------------- Optimize threshold --------------------
def predict_first_error(step_scores: List[float], threshold: Optional[float] = None, use_prob: bool = False, invert_score: bool = False, patience: int = 2):
    ss = [ -float(s) if invert_score else float(s) for s in step_scores]
    if use_prob:
        thr = 0.5 if threshold is None else float(threshold)
        seq = [1/(1+math.exp(-s)) for s in ss]
        bad = [ (p <= thr) for p in seq ]
    else:
        thr = 0.0 if threshold is None else float(threshold)
        bad = [ (s <= thr) for s in ss ]

    if patience <= 1:
        for i, b in enumerate(bad):
            if b: return i
        return -1
    else:
        cnt = 0
        for i, b in enumerate(bad):
            cnt = cnt + 1 if b else 0
            if cnt >= patience:
                return i - patience + 1
        return -1

def _flatten_scores(step_scores_list, use_prob: bool, invert_score: bool) -> np.ndarray:
    flat = np.array([s for ss in step_scores_list for s in ss], dtype=float)
    if invert_score:
        flat = -flat
    if use_prob:
        flat = 1.0 / (1.0 + np.exp(-flat))
    return flat

def make_threshold_candidates(step_scores_list, use_prob: bool, invert_score: bool):
    xs = _flatten_scores(step_scores_list, use_prob, invert_score)
    if xs.size == 0:
        return np.array([0.0]) if not use_prob else np.array([0.5])
    # 중심부 분위수 위주(극단 outlier 회피)
    qs = np.linspace(0.05, 0.95, 37)   # 5%~95% 사이 37점
    cands = np.unique(np.quantile(xs, qs))
    # 대표값 보강
    if use_prob:
        cands = np.unique(np.concatenate([cands, np.array([0.5])]))
    else:
        cands = np.unique(np.concatenate([cands, np.array([0.0])]))
    return cands

def find_best_threshold_on_batch(step_scores_list, labels_first_err, use_prob, invert_score, patience=1, cand=None):
    if cand is None:
        cand = make_threshold_candidates(step_scores_list, use_prob, invert_score)
    best = (None, -1.0, (0.0,0.0,0.0))  # (thr, f1, (err_acc, cor_acc, f1))
    gts = [int(x) for x in labels_first_err]

    for th in cand:
        preds = [predict_first_error(ss, threshold=float(th), use_prob=use_prob, invert_score=invert_score, patience=patience)
                 for ss in step_scores_list]
        data_error   = [(p==g) for p,g in zip(preds,gts) if g!=-1]
        data_correct = [(p==g) for p,g in zip(preds,gts) if g==-1]
        err_acc = float(np.mean(data_error)) * 100.0 if data_error else 0.0
        cor_acc = float(np.mean(data_correct)) * 100.0 if data_correct else 0.0
        f1 = 0.0 if (err_acc+cor_acc)==0 else 2*err_acc*cor_acc/(err_acc+cor_acc)
        if f1 > best[1]:
            best = (float(th), float(f1), (err_acc, cor_acc, f1))
    return best

def _build_step_level_calibration_arrays(step_scores_list: List[List[float]], labels_first_err: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for scores, first_err in zip(step_scores_list, labels_first_err):
        fe = int(first_err)
        for idx, s in enumerate(scores):
            if fe == -1:
                is_correct = 1
            else:
                is_correct = 1 if idx < fe else 0
            xs.append([float(s)])
            ys.append(is_correct)
    if not xs:
        return np.empty((0, 1), dtype=float), np.empty((0,), dtype=int)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=int)

def fit_platt_scaler(step_scores_list: List[List[float]], labels_first_err: List[int]) -> Optional[Dict[str, float]]:
    """Fit a 1-D logistic regression calibrator on dev logits (Platt scaling)."""
    X, y = _build_step_level_calibration_arrays(step_scores_list, labels_first_err)
    if X.shape[0] == 0 or len(np.unique(y)) < 2:
        return None
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception:
        print("[WARN] sklearn not available; skip Platt calibration.")
        return None

    clf = LogisticRegression(solver="lbfgs")
    clf.fit(X, y)
    scale = float(clf.coef_[0][0])
    bias = float(clf.intercept_[0])
    return {"scale": scale, "bias": bias}

def apply_platt_scaler(step_scores_list: List[List[float]], scaler: Optional[Dict[str, float]]) -> List[List[float]]:
    if not scaler:
        return [list(map(float, ss)) for ss in step_scores_list]
    a = scaler.get("scale", 1.0)
    b = scaler.get("bias", 0.0)
    calibrated: List[List[float]] = []
    for scores in step_scores_list:
        calibrated.append([a * float(s) + b for s in scores])
    return calibrated

def choose_direction_and_threshold(dev_scores_list, dev_labels, *, try_patiences=(1,2,3), use_prob=False, force_invert: Optional[bool]=None):
    # 1) 부호 자동 판정(AUC 비교)
    from sklearn.metrics import roc_auc_score
    # step-level pseudo label 구성
    y_true, y_logit = [], []
    for ss, first_err in zip(dev_scores_list, dev_labels):
        for idx, s in enumerate(ss):
            is_correct = 1 if (first_err == -1 or idx < first_err) else 0
            y_true.append(is_correct)
            y_logit.append(float(s))
    y_true = np.array(y_true, dtype=int)
    y_logit = np.array(y_logit, dtype=float)
    auc = roc_auc_score(y_true, y_logit) if len(np.unique(y_true))>1 else 0.5
    auc_inv = roc_auc_score(y_true, -y_logit) if len(np.unique(y_true))>1 else 0.5
    print(f"[CHK][AUC] step-level AUC(logit)={auc:.4f} | AUC(-logit)={auc_inv:.4f}")
    if force_invert is None:
        invert = (auc_inv > auc)
    else:
        invert = bool(force_invert)

    # 2) patience & threshold grid
    best_all = (None, None, -1.0, (0.0,0.0,0.0))  # (pat, thr, f1, (err,cor,f1))
    for pat in try_patiences:
        thr, f1, triple = find_best_threshold_on_batch(dev_scores_list, dev_labels, use_prob=use_prob, invert_score=invert, patience=pat)
        if f1 > best_all[2]:
            best_all = (pat, thr, f1, triple)

    return {"invert": invert, "patience": best_all[0], "threshold": best_all[1], "f1": best_all[2], "err_acc": best_all[3][0], "cor_acc": best_all[3][1], "auc": auc, "auc_inv": auc_inv}

def _collect_dev_chunk_scores(prm, tokenizer, ds, budget_examples=512, batch_size=16, rw_token="<RW>"):
    # dev 튜닝은 gsm8k split에서 앞쪽 샘플 N개(예: 256~512) 로 수행
    step_scores_list, labels = [], []
    taken = 0
    for i in range(0, len(ds), batch_size):
        if taken >= budget_examples: break
        batch = ds.select(range(i, min(i+batch_size, len(ds))))
        problems = [ex["problem"] for ex in batch]
        steps_list = [list(ex["steps"]) for ex in batch]
        ss = batch_score_steps(prm, tokenizer, problems, steps_list, rw_token=rw_token)
        step_scores_list.extend(ss)
        labels.extend(list(batch["label"]))
        taken += len(batch)
    return step_scores_list, labels

# -------------------- Step-level analytics (ROC/PR/AUC/KS etc.) --------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def _transform_scores(x: np.ndarray, use_prob: bool, invert: bool) -> np.ndarray:
    if invert: x = -x
    return _sigmoid(x) if use_prob else x

def _roc_curve_np(y_true: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    order = np.argsort(-scores, kind="mergesort")
    y = y_true[order]; s = scores[order]
    P = y.sum(); N = len(y) - P
    if P == 0 or N == 0:
        return np.array([0.0,1.0]), np.array([0.0,1.0]), np.array([np.inf,-np.inf])
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    diff = np.r_[True, s[1:] != s[:-1]]
    idx = np.where(diff)[0]
    tpr = tp[idx] / P; fpr = fp[idx] / N; thr = s[idx]
    tpr = np.r_[0.0, tpr, 1.0]; fpr = np.r_[0.0, fpr, 1.0]
    thr = np.r_[thr[0] + 1e9, thr, thr[-1] - 1e9]
    return fpr, tpr, thr

def _auc_trapz(x: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(x)
    return float(np.trapezoid(y[order], x[order]))

def _pr_curve_np(y_true: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray,np.ndarray,float]:
    order = np.argsort(-scores, kind="mergesort")
    y = y_true[order]
    P = y.sum()
    if P == 0:
        return np.array([0.0,1.0]), np.array([1.0,0.0]), float("nan")
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / P
    ap = float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))
    recall = np.r_[0.0, recall]; precision = np.r_[1.0, precision]
    return recall, precision, ap

def _ks_statistic(y_true: np.ndarray, scores: np.ndarray) -> float:
    s_pos = scores[y_true == 1]; s_neg = scores[y_true == 0]
    if s_pos.size == 0 or s_neg.size == 0:
        return float("nan")
    allx = np.unique(np.concatenate([s_pos, s_neg]))
    def cdf_at(a, grid):
        a = np.sort(a)
        return np.searchsorted(a, grid, side='right') / a.size
    Fp = cdf_at(s_pos, allx); Fn = cdf_at(s_neg, allx)
    return float(np.max(np.abs(Fp - Fn)))

def build_step_level_arrays(step_scores_list: List[List[float]], labels_first_err: List[int], use_prob: bool, invert: bool) -> Tuple[np.ndarray, np.ndarray]:
    ys, ss = [], []
    for scores, first_err in zip(step_scores_list, labels_first_err):
        if not scores:  # empty
            continue
        L = len(scores)
        fe = int(first_err)
        if fe == -1:
            y = np.ones(L, dtype=int)
        else:
            y = np.zeros(L, dtype=int)
            y[:max(0, min(L, fe))] = 1  # steps strictly before first error are positives
        ys.append(y)
        ss.append(np.array(scores, dtype=float))
    if not ys:
        return np.array([], dtype=int), np.array([], dtype=float)
    y_all = np.concatenate(ys, axis=0)
    s_all = np.concatenate(ss, axis=0)
    s_all = _transform_scores(s_all, use_prob=use_prob, invert=invert)
    m = ~np.isnan(s_all)
    return y_all[m], s_all[m]

def step_level_metric_suite(split_name: str, step_scores_list: List[List[float]], labels_first_err: List[int], out_dir: str, use_prob: bool, invert: bool) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    y, s = build_step_level_arrays(step_scores_list, labels_first_err, use_prob, invert)
    res = dict(split=split_name, n_steps=int(y.size), n_pos=int(y.sum()), n_neg=int((y==0).sum()),
               roc_auc=float("nan"), pr_auc=float("nan"), ks=float("nan"),
               best_thresh=float("nan"), tpr_at_best=float("nan"), fpr_at_best=float("nan"),
               roc_path="", pr_path="", cal_path="")
    if y.size < 2 or res["n_pos"] == 0 or res["n_neg"] == 0:
        return res

    fpr, tpr, thr = _roc_curve_np(y, s)
    auc = _auc_trapz(fpr, tpr)
    ks = _ks_statistic(y, s)
    rec, prec, ap = _pr_curve_np(y, s)

    j = tpr - fpr
    j_idx = int(np.argmax(j))
    best_tau = thr[j_idx]; tpr_best = float(tpr[j_idx]); fpr_best = float(fpr[j_idx])

    roc_path = os.path.join(out_dir, f"{split_name}_stepROC.png")
    pr_path  = os.path.join(out_dir, f"{split_name}_stepPR.png")

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
    plt.plot([0,1],[0,1],"--",alpha=0.6,label="Chance")
    plt.scatter([fpr_best],[tpr_best], s=30, label=f"Best J (τ={best_tau:.3g})")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    title = f"{split_name} — Step-level ROC" + (" (prob)" if use_prob else " (logit)")
    plt.title(title); plt.legend(); plt.grid(alpha=0.2)
    plt.savefig(roc_path, bbox_inches="tight"); plt.close()

    plt.figure()
    plt.plot(rec, prec, label=f"PR (AP={ap:.3f})")
    base = res["n_pos"] / max(res["n_steps"], 1)
    plt.hlines(base, 0, 1, linestyles="--", alpha=0.6, label=f"Baseline={base:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    title = f"{split_name} — Step-level PR" + (" (prob)" if use_prob else " (logit)")
    plt.title(title); plt.legend(); plt.grid(alpha=0.2)
    plt.savefig(pr_path, bbox_inches="tight"); plt.close()

    cal_path = ""
    if use_prob:
        bins = np.linspace(0,1,12)
        inds = np.digitize(s, bins) - 1
        accs, confs, cnts = [], [], []
        for b in range(len(bins)-1):
            m = inds == b
            if m.sum() == 0: continue
            accs.append(y[m].mean())
            confs.append(s[m].mean())
            cnts.append(int(m.sum()))
        if accs:
            cal_path = os.path.join(out_dir, f"{split_name}_calibration.png")
            plt.figure()
            plt.plot([0,1],[0,1],"--",alpha=0.6,label="Perfect")
            sizes = (np.sqrt(np.array(cnts)) * 5).tolist()
            plt.scatter(confs, accs, s=sizes, label="Bins")
            plt.xlabel("Confidence"); plt.ylabel("Empirical Accuracy")
            plt.title(f"{split_name} — Calibration (size ∝ √count)")
            plt.legend(); plt.grid(alpha=0.2)
            plt.savefig(cal_path, bbox_inches="tight"); plt.close()

    res.update(dict(
        roc_auc=float(auc), pr_auc=float(ap), ks=float(ks) if ks==ks else float("nan"),
        best_thresh=float(best_tau), tpr_at_best=tpr_best, fpr_at_best=fpr_best,
        roc_path=roc_path, pr_path=pr_path, cal_path=cal_path
    ))
    return res

def per_sequence_average_precision(step_scores_list: List[List[float]], labels_first_err: List[int], use_prob: bool, invert: bool) -> float:
    aps = []
    for scores, fe in zip(step_scores_list, labels_first_err):
        if not scores: continue
        L = len(scores)
        fe = int(fe)
        if fe == -1:
            continue  # no negatives; skip AP
        y = np.zeros(L, dtype=int)
        y[:max(0, min(L, fe))] = 1
        s = _transform_scores(np.array(scores, dtype=float), use_prob, invert)
        order = np.argsort(-s, kind="mergesort")
        y_sorted = y[order]
        P = y.sum()
        if P == 0: continue
        tp = np.cumsum(y_sorted)
        prec = tp / (np.arange(L) + 1)
        aps.append(float((prec * y_sorted).sum() / P))
    return float(np.mean(aps)) if aps else float("nan")

def error_localization_stats(predictions: List[int], labels_first_err: List[int], out_dir: str, split_name: str) -> Dict[str,float]:
    diffs = []
    for p, g in zip(predictions, labels_first_err):
        g = int(g)
        if g == -1:
            continue
        diffs.append(int(p) - g)
    if not diffs:
        return {"mae": float("nan"), "median_ae": float("nan"), "within1": float("nan"), "within2": float("nan")}
    diffs = np.array(diffs, dtype=int)
    mae = float(np.abs(diffs).mean())
    med = float(np.median(np.abs(diffs)))
    within1 = float((np.abs(diffs) <= 1).mean() * 100.0)
    within2 = float((np.abs(diffs) <= 2).mean() * 100.0)

    os.makedirs(out_dir, exist_ok=True)
    hist_path = os.path.join(out_dir, f"{split_name}_errloc_hist.png")
    plt.figure()
    bins = np.arange(diffs.min()-0.5, diffs.max()+1.5)
    plt.hist(diffs, bins=bins)
    plt.xlabel("Predicted index - True first-error index")
    plt.ylabel("Count")
    plt.title(f"{split_name} — Error Localization Offset")
    plt.grid(alpha=0.2)
    plt.savefig(hist_path, bbox_inches="tight"); plt.close()
    return {"mae": mae, "median_ae": med, "within1": within1, "within2": within2}

# -------------------- Main evaluation --------------------
def _find_marker_positions(input_ids_1d: torch.Tensor, marker_ids: List[int]) -> torch.Tensor:
    """Return positions (indices) of the last token of each marker occurrence.
    Supports multi-token markers by matching the full sequence and recording the index of the last token of the matched marker. """
    ids = input_ids_1d.tolist()
    m = len(marker_ids)
    pos: List[int] = []
    if m == 0:
        return torch.tensor([], dtype=torch.long, device=input_ids_1d.device)
    if m == 1:
        tid = marker_ids[0]
        for i, t in enumerate(ids):
            if t == tid:
                pos.append(i)
        return torch.tensor(pos, dtype=torch.long, device=input_ids_1d.device)
    # multi-token
    for i in range(0, max(0, len(ids) - m + 1)):
        if ids[i] != marker_ids[0]:
            continue
        if ids[i : i + m] == marker_ids:
            pos.append(i + m - 1)
    return torch.tensor(pos, dtype=torch.long, device=input_ids_1d.device)

def batch_score_steps(prm: ProcessRewardModel, tokenizer: AutoTokenizer, problems: List[str], steps_list: List[List[str]],  rw_token: Optional[str] = "<RW>",) -> List[List[float]]:
    """Build PRM prompts over pre-split steps and return per-example step scores."""
    assert len(problems) == len(steps_list)
    device = next(prm.parameters()).device
    rw_token_eff = rw_token or getattr(prm, "rw_token", "<RW>")

    prompts: List[str] = [
        build_prm_prompt(tokenizer, q, steps, rw_token=rw_token_eff, ds_name="")
        for q, steps in zip(problems, steps_list)
    ]

    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False,)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # Find RW marker positions for each sample
    marker_ids = tokenizer.encode(rw_token_eff, add_special_tokens=False)
    rw_positions: List[torch.Tensor] = []
    for b in range(input_ids.size(0)):
        rw_positions.append(_find_marker_positions(input_ids[b], marker_ids))

    with torch.no_grad():
        scores_flat, lengths = prm.scores_at_rw(input_ids, attention_mask, rw_positions=rw_positions)

    # Split flat scores back into per-sample lists
    out: List[List[float]] = []
    offset = 0
    for L in lengths:
        if L <= 0:
            out.append([])
            continue
        seg = scores_flat[offset : offset + L]
        out.append([float(x) for x in seg.detach().cpu().tolist()])
        offset += L
    # ── DEBUG: 합계 무결성 체크 ─────────────────────────────────────────
    total_len = sum(len(x) for x in out)
    if total_len != int(scores_flat.numel()):
        print(f"[DEBUG][WARN] sum(lengths)={total_len} != scores_flat.size={int(scores_flat.numel())}")
    # ──────────────────────────────────────────────────────────────────
    return out

def eval_processbench(prm_dir: str, configs: List[str], out_dir: str, batch_size: int = 32, threshold: Optional[float] = None, use_prob: bool = False, rw_token: Optional[str] = "<RW>",
    invert_score: Optional[bool] = None, patience: int = 1, dev_budget_examples: int = 512, debug: bool = True, calibration_mode: Optional[str] = None,):
    os.makedirs(out_dir, exist_ok=True)

    # Load tokenizer and PRM wrapper (reward head)
    tokenizer = AutoTokenizer.from_pretrained(prm_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    prm = ProcessRewardModel.from_pretrained(prm_dir, tokenizer=tokenizer).eval()

    last_linear = _get_last_linear(prm.reward_head)
    if last_linear is not None:
        w = last_linear.weight.detach().cpu().numpy()
        b = last_linear.bias.detach().cpu().numpy() if last_linear.bias is not None else None
    else:
        params = [p.detach().flatten().cpu() for p in prm.reward_head.parameters()]
        if params:
            w = torch.cat(params).numpy()
            b = None
        else:
            w, b = None, None
    print("[CHK][HEAD] shape", w.shape, "|| w_mean/std", w.mean(), w.std(), "|| b", b)
    print(f"[DEBUG] use_prob={use_prob}, threshold={threshold}, invert_score={invert_score}, calibration_mode={calibration_mode}")
    
    # ---------- tune on dev (only when needed) ----------
    calibrator: Optional[Dict[str, float]] = None
    need_dev = (threshold is None) or (invert_score is None) or (calibration_mode not in (None, "", "none", "NONE"))
    if need_dev:
        try:
            ds_dev = load_dataset("Qwen/ProcessBench", split="gsm8k")
            dev_scores, dev_labels = _collect_dev_chunk_scores(
                prm, tokenizer, ds_dev, budget_examples=dev_budget_examples,batch_size=batch_size, rw_token=getattr(prm, "rw_token", rw_token) or "<RW>",)

            scores_for_tuning = dev_scores
            cal_mode = (calibration_mode or "").lower().strip()
            if cal_mode and cal_mode != "none":
                if cal_mode == "platt":
                    calibrator = fit_platt_scaler(dev_scores, dev_labels)
                    if calibrator:
                        scores_for_tuning = apply_platt_scaler(dev_scores, calibrator)
                        print(f"[TUNE] Platt calibration scale={calibrator['scale']:.6f}, bias={calibrator['bias']:.6f}")
                    else:
                        print("[TUNE] Platt calibration skipped (insufficient signal).")
                else:
                    raise ValueError(f"Unsupported calibration_mode='{calibration_mode}'")

            tune = choose_direction_and_threshold(
                scores_for_tuning, dev_labels, try_patiences=(1,2,3), use_prob=use_prob,
                force_invert=None if invert_score is None else bool(invert_score)
            )
            if invert_score is None:
                invert_score = tune["invert"]
            # invert_score = False
            if threshold is None:
                threshold = tune["threshold"]
            patience = tune["patience"]
            print(f"[TUNE] invert={invert_score}, patience={patience}, threshold={tune['threshold']:.4f} | "
                f"dev_F1={tune['f1']:.1f}, err_acc={tune['err_acc']:.1f}, cor_acc={tune['cor_acc']:.1f}, "
                f"AUC={tune['auc']:.3f}, AUC(-)={tune['auc_inv']:.3f}")
        except Exception as e:
            print("[WARN] DEV tuning failed, fallback to defaults:", repr(e))
            calibrator = None
            if invert_score is None: invert_score = False
            if threshold is None: threshold = 0.0 if not use_prob else 0.5

    for cfg in configs:
        ds = load_dataset("Qwen/ProcessBench", split=cfg)
        N = len(ds)
        predictions: List[int] = []

        all_step_scores_for_split: List[List[float]] = []
        all_labels_for_split: List[int] = []

        # ---------- evaluate on test ONLY ----------
        for i in tqdm(range(0, N, batch_size), desc=f"Processing {cfg}", dynamic_ncols=True):
            batch = ds.select(range(i, min(i + batch_size, N)))
            problems = [ex["problem"] for ex in batch]
            steps_list = [list(ex["steps"]) for ex in batch]

            if debug and i == 0:
                _debug_rw_markers(tokenizer, prm, problems, steps_list, rw_token)

            step_scores_list = batch_score_steps(prm, tokenizer, problems, steps_list, rw_token=getattr(prm, "rw_token", rw_token))

            if calibrator:
                step_scores_list = apply_platt_scaler(step_scores_list, calibrator)

            # collect for analytics
            all_step_scores_for_split.extend(step_scores_list)
            all_labels_for_split.extend(list(batch["label"]))

            if debug and i == 0:
                _debug_first_batch_scores(step_scores_list, threshold=threshold, use_prob=use_prob)

            for step_scores in step_scores_list:
                pred = predict_first_error(step_scores, threshold=threshold, use_prob=use_prob, invert_score=invert_score, patience=patience)
                predictions.append(pred)

        # ---------- metrics on test ----------
        assert len(predictions) == N, "[BUG] predictions size mismatch"
        res_data: List[Dict[str, Any]] = []
        for idx in range(N):
            d = ds[idx]
            pred = predictions[idx]
            new_d = dict(d)
            new_d["prediction"] = pred
            new_d["match"] = bool(pred == d.get("label", -1))
            res_data.append(new_d)

        data_error = [e for e in res_data if e.get("label", -1) != -1]
        data_correct = [e for e in res_data if e.get("label", -1) == -1]
        print(f"[DEBUG:{cfg}] test sizes: error={len(data_error)}, correct={len(data_correct)}")

        err_path = os.path.join(out_dir, f"{cfg}_error.jsonl")
        cor_path = os.path.join(out_dir, f"{cfg}_correct.jsonl")
        with open(err_path, "w", encoding="utf-8") as f:
            for e in data_error:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        with open(cor_path, "w", encoding="utf-8") as f:
            for e in data_correct:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

        acc1 = float(np.mean([e["match"] for e in data_error]) * 100.0) if data_error else 0.0
        acc2 = float(np.mean([e["match"] for e in data_correct]) * 100.0) if data_correct else 0.0
        f1 = 0.0 if (acc1 + acc2) == 0 else 2 * acc1 * acc2 / (acc1 + acc2)
        print(f"[Final] {cfg} error acc: {acc1:.1f}, correct acc: {acc2:.1f}, f1: {f1:.1f}")

        # ---------- step-level analytics (only when debug=True) ----------
        if debug:
            split_out = os.path.join(out_dir, f"{cfg}_analysis")
            os.makedirs(split_out, exist_ok=True)

            metrics = step_level_metric_suite(
                split_name=cfg,
                step_scores_list=all_step_scores_for_split,
                labels_first_err=all_labels_for_split,
                out_dir=split_out,
                use_prob=use_prob,
                invert=bool(invert_score),
            )
            print(f"[STEP] {cfg} n_steps={metrics['n_steps']} pos={metrics['n_pos']} neg={metrics['n_neg']} "
                  f"| ROC_AUC={metrics['roc_auc']:.4f} PR_AUC={metrics['pr_auc']:.4f} KS={metrics['ks']:.4f} "
                  f"| Bestτ={metrics['best_thresh']:.4g} TPR={metrics['tpr_at_best']:.3f} FPR={metrics['fpr_at_best']:.3f}")

            map_seq = per_sequence_average_precision(
                all_step_scores_for_split, all_labels_for_split, use_prob=use_prob, invert=bool(invert_score)
            )
            print(f"[STEP] {cfg} per-sequence mAP={map_seq:.4f}")

            loc = error_localization_stats(predictions, list(ds["label"]), out_dir=split_out, split_name=cfg)
            print(f"[LOC]  {cfg} MAE={loc['mae']:.3f}, MedAE={loc['median_ae']:.3f}, within±1={loc['within1']:.1f}%, within±2={loc['within2']:.1f}%")


import torch.nn as nn
def _get_last_linear(module: nn.Module) -> nn.Linear | None:
    last = None
    for m in module.modules():
        if isinstance(m, nn.Linear):
            last = m
    return last

def make_out_tag_from_prm_dir(prm_dir: str) -> str:
    p = os.path.normpath(prm_dir.rstrip("/"))
    last = os.path.basename(p)
    if last.lower() in {"final_model", "checkpoint", "checkpoints"}:
        parent = os.path.basename(os.path.dirname(p))
        return parent or last
    return last

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate custom PRM on ProcessBench (score-based)")
    ap.add_argument("--splits", type=str, default="gsm8k,math,olympiadbench,omnimath") # "gsm8k,math,olympiadbench,omnimath"
    ap.add_argument("--batch_size", type=int, default=18)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--use_prob", action="store_true", help="Apply sigmoid and compare to threshold (default 0.5)")
    ap.add_argument("--rw_token", type=str, default="<RW>")
    ap.add_argument("--invert_score", action="store_true", help="Flip PRM scores (s -> -s) before thresholding. Use when AUC(-logit) > AUC(logit).")
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--dev_budget_examples", type=int, default=400)
    ap.add_argument("--calibration", type=str, default=None, help="Optional calibration mode: platt/none")
    ap.add_argument("--debug", action="store_false", default=True, help="Enable step-level analytics & plots")
    ap.add_argument("--prm_dir", type=str, required=True, help="Checkpoint dir for the PRM to evaluate")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--analysis_root", type=str, default="/home/leena/rs_prm/analysis/processbench/ori")
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    out_dir = args.out_dir
    if out_dir is None:
        tag = make_out_tag_from_prm_dir(args.prm_dir)
        out_dir = os.path.join(args.analysis_root, tag)

    eval_processbench(
        prm_dir=args.prm_dir,
        configs=splits,
        out_dir=out_dir,
        batch_size=args.batch_size,
        threshold=args.threshold,
        use_prob=args.use_prob,
        rw_token=args.rw_token,
        invert_score=False,
        patience=(args.patience if args.patience is not None else 1),
        dev_budget_examples=args.dev_budget_examples,
        debug=args.debug,
        calibration_mode=args.calibration,
    )

if __name__ == "__main__":
    main()

