import json, argparse, sys, math
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from difflib import SequenceMatcher
from collections import defaultdict

# -----------------------------
# IO helpers
# -----------------------------
def _load_records(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    text = p.read_text(encoding="utf-8").strip()
    # Heuristics: if startswith '[' -> JSON array
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON file must contain an array.")
        return data
    # else assume JSONL
    recs = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        recs.append(json.loads(ln))
    return recs

def _dump_records(path: str, recs: List[Dict[str, Any]], json_array: bool=True) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if json_array:
        p.write_text(json.dumps(recs, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        with p.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

# -----------------------------
# Text normalization & keys
# -----------------------------
def _norm_txt(s: str) -> str:
    # 보수적 정규화: 공백 표준화, 양끝 공백 제거, \n->space, 연속 스페이스 1개로
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = " ".join(s.split())  # collapse whitespace incl. newlines/tabs
    return s

def _norm_steps(steps: List[str]) -> List[str]:
    if not isinstance(steps, list):
        return []
    return [_norm_txt(x) for x in steps]

def _make_key(entry: Dict[str, Any]) -> Tuple[str, Tuple[str, ...]]:
    q = _norm_txt(entry.get("question", ""))
    comp = entry.get("completion", [])
    comp_norm = _norm_steps(comp)
    return (q, tuple(comp_norm))

# -----------------------------
# Index builders
# -----------------------------
def _build_exact_index(recs: List[Dict[str, Any]]) -> Dict[Tuple[str, Tuple[str, ...]], Dict[str, Any]]:
    idx = {}
    for r in recs:
        idx[_make_key(r)] = r
    return idx

def _build_question_buckets(recs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets = defaultdict(list)
    for r in recs:
        q = _norm_txt(r.get("question", ""))
        buckets[q].append(r)
    return buckets

# -----------------------------
# MI normalization (robust z -> [0,1])
# -----------------------------
def _percentile(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs)-1) * (p/100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] + (xs[c]-xs[f]) * (k - f)

def _median(xs):
    return _percentile(xs, 50)

def _mad(xs, med):
    # Median absolute deviation
    return _median([abs(x - med) for x in xs])

def _robust_scale(stats_vals):
    """
    Compute robust scale using MAD; fall back to IQR or std if necessary.
    Returns (median, scale, used)
    """
    if not stats_vals:
        return 0.0, 1.0, "fallback:empty"

    med = _median(stats_vals)
    mad = _mad(stats_vals, med)
    # 1.4826: make MAD consistent with std for normal dist
    scale = 1.4826 * mad
    used = "mad"

    if scale <= 1e-12:
        # fallback to IQR
        q75 = _percentile(stats_vals, 75)
        q25 = _percentile(stats_vals, 25)
        iqr = max(q75 - q25, 0.0)
        # 0.7413 is approx factor s.t. 0.7413*IQR ≈ std for normal
        scale = 0.7413 * iqr
        used = "iqr"
        if scale <= 1e-12:
            # final fallback to std
            mu = sum(stats_vals)/len(stats_vals)
            var = sum((x-mu)**2 for x in stats_vals)/max(len(stats_vals)-1, 1)
            scale = math.sqrt(max(var, 1e-24))
            used = "std"
            if scale <= 1e-12:
                scale = 1.0
                used = "epsilon"
    return med, scale, used

def _map_to_unit(z, method="sigmoid"):
    if method == "sigmoid":
        # logistic; centered at 0 -> 0.5
        return 1.0 / (1.0 + math.exp(-z))
    elif method in ("cdf", "gaussian_cdf", "norm_cdf"):
        # standard normal CDF via erf
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    else:
        raise ValueError(f"unknown map method: {method}")

def collect_mi_tail_values(recs: List[Dict[str, Any]], switch_at: int) -> List[float]:
    vals = []
    for r in recs:
        rewards = r.get("step_reward", [])
        if not isinstance(rewards, list):
            continue
        for i in range(switch_at, len(rewards)):
            v = rewards[i]
            if isinstance(v, (int, float)) and math.isfinite(v):
                vals.append(float(v))
    return vals

def normalize_mi_tail_rewards_in_memory(
    recs: List[Dict[str, Any]],
    switch_at: int = 1,
    z_clip: float = 6.0,
    map_method: str = "sigmoid",
    write_meta: bool = True
):
    mi_vals = collect_mi_tail_values(recs, switch_at)
    med, scale, scale_src = _robust_scale(mi_vals)

    out = []
    for r in recs:
        rewards = r.get("step_reward", [])
        if not isinstance(rewards, list) or len(rewards) == 0:
            out.append(r)
            continue
        new_rewards = list(rewards)
        for i in range(min(len(rewards), switch_at), len(rewards)):
            x = rewards[i]
            if isinstance(x, (int, float)) and math.isfinite(x):
                z = (float(x) - med) / (scale if scale > 0 else 1.0)
                if z_clip is not None and z_clip > 0:
                    z = max(min(z, z_clip), -z_clip)
                new_rewards[i] = _map_to_unit(z, map_method)
        new_rec = dict(r)
        new_rec["step_reward"] = new_rewards
        if write_meta:
            meta = dict(new_rec.get("meta", {}))
            meta.update({
                "mi_tail_normalization": {
                    "type": "robust_z",
                    "median": med,
                    "scale": scale,
                    "scale_source": scale_src,
                    "z_clip": z_clip,
                    "map": map_method,
                    "switch_at": switch_at,
                    "n_tail_values": len(mi_vals)
                }
            })
            new_rec["meta"] = meta
        out.append(new_rec)

    info = {
        "median": med,
        "scale": scale,
        "scale_source": scale_src,
        "z_clip": z_clip,
        "map": map_method,
        "switch_at": switch_at,
        "n_tail_values": len(mi_vals)
    }
    return out, info

# -----------------------------
# Reward selection & merging
# -----------------------------
def _normalize_reward_type(rt: str) -> str:
    rt = (rt or "").strip().lower()
    if rt.startswith("mi_"):
        return rt
    return f"mi_{rt}"  # allow 'loo' -> 'mi_loo' 등

def _get_list(entry: Dict[str, Any], key: str) -> List[float]:
    val = entry.get(key, [])
    return val if isinstance(val, list) else []

def _pick_answer(entry: Dict[str, Any]) -> Optional[str]:
    # 선호도: gold_answer > answer > None
    for k in ("gold_answer", "answer"):
        if k in entry and isinstance(entry[k], str):
            return entry[k]
    return None

def _sequence_similarity(a_steps: List[str], b_steps: List[str]) -> float:
    # 간단 유사도: 두 completion 전체 문자열을 이어 붙여 비교
    a = "\n".join(a_steps)
    b = "\n".join(b_steps)
    return SequenceMatcher(a=a, b=b).ratio()

def _merge_one(
    base_rec: Dict[str, Any],
    mi_rec: Dict[str, Any],
    reward_type: str,
    on_length: str = "trim",  # 'trim' or 'strict'
    switch_at: int = 1 
) -> Dict[str, Any]:
    steps = base_rec.get("completion", [])
    steps = steps if isinstance(steps, list) else []
    n_base_steps = len(steps)

    base_rewards = _get_list(base_rec, "base_reward")
    mi_rewards   = _get_list(mi_rec, reward_type)

    # 길이 처리
    if on_length == "strict":
        if not (len(base_rewards) == len(mi_rewards) == n_base_steps):
            raise ValueError("Length mismatch in strict mode.")
        n = n_base_steps
    else:
        # trim to min length among [steps, base_rewards, mi_rewards]
        n = min(n_base_steps, len(base_rewards), len(mi_rewards) if len(mi_rewards)>0 else n_base_steps)

    if n == 0:
        merged = []
    else:
        take_base = min(switch_at, n)
        merged = base_rewards[:take_base] 
        if n > take_base:
            merged.extend(mi_rewards[take_base:n])

    # correct_mask는 가능한 쪽을 사용 (mi가 좀 더 최신 라벨일 수 있다고 가정)
    correct_mask = None
    if isinstance(mi_rec.get("correct_mask"), list):
        correct_mask = mi_rec["correct_mask"][:n]
    elif isinstance(base_rec.get("correct_mask"), list):
        correct_mask = base_rec["correct_mask"][:n]

    # answer 선택
    ans = _pick_answer(base_rec) or _pick_answer(mi_rec)

    out = {
        "question": base_rec.get("question"),
        "completion": steps[:n],
        "step_reward": merged,
        "reward_type": f"{switch_at}_qval_then_{reward_type}",
        "task": base_rec.get("task") or mi_rec.get("task"),
        "answer": ans,
        "meta": {
            "base_reward_key": "base_reward",
            "mi_reward_key": reward_type,
            "policy": f"{switch_at}_qval_then_{reward_type}",
            "trimmed_to": n,
            "base_steps_len": n_base_steps,
            "base_reward_len": len(base_rewards),
            "mi_reward_len": len(mi_rewards),
            "source_match": "exact"
        }
    }
    if correct_mask is not None:
        out["correct_mask"] = correct_mask
    return out

# -----------------------------
# Main merge routine
# -----------------------------
def merge_datasets(
    base_path: str,
    mi_path: str,
    reward_type: str,
    on_length: str="strict",
    fuzzy_threshold: float=0.95,
    switch_at: int = 1, 
) -> List[Dict[str, Any]]:
    reward_type = _normalize_reward_type(reward_type)
    base = _load_records(base_path)
    mi   = _load_records(mi_path)

    mi_exact_idx = _build_exact_index(mi)
    mi_q_buckets = _build_question_buckets(mi)

    merged: List[Dict[str, Any]] = []
    n_exact, n_fuzzy, n_missing, n_len_warn = 0, 0, 0, 0

    for b in base:
        key = _make_key(b)
        m = mi_exact_idx.get(key)
        match_mode = "exact"

        if m is None:
            q = _norm_txt(b.get("question", ""))
            candidates = mi_q_buckets.get(q, [])
            if candidates:
                b_steps = _norm_steps(b.get("completion", []))
                best, best_score = None, -1.0
                for cand in candidates:
                    score = _sequence_similarity(b_steps, _norm_steps(cand.get("completion", [])))
                    if score > best_score:
                        best, best_score = cand, score
                if best and best_score >= fuzzy_threshold:
                    m = best
                    match_mode = f"fuzzy({best_score:.3f})"

        if m is None:
            n_missing += 1
            # 매치 실패: base만으로라도 첫 스텝만 살리고 나머지는 비움(혹은 스킵)
            base_rewards = _get_list(b, "base_reward")
            steps = b.get("completion", [])
            steps = steps if isinstance(steps, list) else []
            if steps and base_rewards:
                take = min(len(steps), len(base_rewards), switch_at) 
                merged_rec = {
                    "question": b.get("question"),
                    "completion": steps[:take],
                    "step_reward": base_rewards[:take],
                    "reward_type": f"only_base_first_{take}_step_matched",
                    "task": b.get("task"),
                    "answer": _pick_answer(b),
                    "meta": {
                        "policy": "no_mi_found_keep_first_base_step",
                        "trimmed_to": take,
                        "base_steps_len": len(steps),
                        "base_reward_len": len(base_rewards),
                        "mi_reward_len": 0,
                        "source_match": "none"
                    }
                }
                if isinstance(b.get("correct_mask"), list):
                    merged_rec["correct_mask"] = b["correct_mask"][:1]
                merged.append(merged_rec)
            else:
                continue
        else:
            out = _merge_one(b, m, reward_type=reward_type, on_length=on_length, switch_at=switch_at)
            if out["meta"]["trimmed_to"] != len(b.get("completion", []) ):
                n_len_warn += 1
            out["meta"]["source_match"] = match_mode
            merged.append(out)
            if match_mode.startswith("exact"):
                n_exact += 1
            else:
                n_fuzzy += 1

    # 간단 리포트 출력
    sys.stderr.write(
        f"[merge] total_base={len(base)} matched_exact={n_exact} "
        f"matched_fuzzy={n_fuzzy} missing={n_missing} length_trimmed={n_len_warn}\n"
    )
    return merged

# -----------------------------
# CLI
# -----------------------------
def main():
    reward_type = "mi_cmi"  # mi_loo | mi_shapley | mi_cmi | mi_margin
    fuzzy_threshold = 0.95
    on_length = "strict"    # strict | trim
    switch_at = 4

    base_path = "/home/leena/rs_prm/datasets/hard_80k/qval/ms_qval_qw3_4b.json"
    mi_path = "/home/leena/rs_prm/datasets/hard_80k/mi_sum2/ms_mi_qw3_4b.json"
    
    # 1) merge
    out_path = f"/home/leena/rs_prm/datasets/hard_80k/qval_mi/ms_{switch_at}_qval_{reward_type}_qw3_4b.json"
    recs = merge_datasets(
        base_path=base_path,
        mi_path=mi_path,
        reward_type=reward_type,
        on_length=on_length,
        fuzzy_threshold=fuzzy_threshold,
        switch_at=switch_at,
    )
    _dump_records(out_path, recs)

    # 2) normalize MI-tail on the merged dataset (in-memory) and overwrite/alternate path
    norm_out_path = out_path.replace(".json", "_norm.json")
    norm_recs, info = normalize_mi_tail_rewards_in_memory(
        recs,
        switch_at=switch_at,
        z_clip=6.0,
        map_method="sigmoid",
        write_meta=True
    )
    _dump_records(norm_out_path, norm_recs)  # 기존 dump 그대로 사용
    print("[norm] stats:", json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()