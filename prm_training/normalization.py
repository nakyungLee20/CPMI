from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import copy, math
import numpy as np

EPS = 1e-8

# --------- helpers ---------
def _flatten_rewards(entries: List[dict], reward_key: str) -> np.ndarray:
    vals: List[float] = []
    for e in entries:
        v = e.get(reward_key, None)
        if isinstance(v, list):
            vals.extend([float(x) for x in v])
    return np.asarray(vals, dtype=float) if len(vals) else np.zeros((0,), dtype=float)

def _sigmoid(x: np.ndarray) -> np.ndarray:
    # stable sigmoid
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))

def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))

def _as_float(x):
    return float(x) if isinstance(x, (int, float)) else None

def _get_step_value(step: Dict[str, Any], reward_key: str) -> Optional[float]:
    # steps = [ { … } ] 형태 지원 (이전 버전 유지)
    v = _as_float(step.get(reward_key))
    if v is not None: return v
    for k in ("metrics", "rewards", "norm", "normalized"):
        d = step.get(k)
        if isinstance(d, dict):
            v = _as_float(d.get(reward_key))
            if v is not None: return v
    # DataFrame-like
    if step.get("metric") == reward_key:
        v = _as_float(step.get("norm_value")) or _as_float(step.get("value"))
        if v is not None: return v
    # suffix
    for k in (f"{reward_key}_norm", f"{reward_key}_normalized"):
        v = _as_float(step.get(k))
        if v is not None: return v
    return None

@dataclass
class NormParams:
    method: str
    # common
    q_low: float | None = None
    q_high: float | None = None
    eps: float | None = None
    # zsig
    mean: float | None = None
    std: float | None = None
    temp: float | None = None
    # logquant / qan_sig
    a: float | None = None
    b: float | None = None
    # exp_cdf
    xmin: float | None = None

# --------- fitters ---------
def fit_params(arr: np.ndarray, method: str, *, q_low: float = 0.01, q_high: float = 0.99,
               eps: float = 0.01, temp: float = 1.0) -> NormParams:
    method = method.lower()
    if arr.size == 0:
        # fallback params
        if method == "logquant":
            return NormParams(method=method, q_low=q_low, q_high=q_high, eps=eps, a=1.0, b=float(np.median(arr)) if arr.size else 0.0)
        if method == "zsig":
            return NormParams(method=method, mean=0.0, std=1.0, temp=temp)
        raise ValueError(f"Unknown method: {method}")

    if method == "zsig":
        m = float(np.mean(arr))
        s = float(np.std(arr)) if arr.size > 1 else 1.0
        s = max(s, 1e-6)
        return NormParams(method=method, mean=m, std=s, temp=float(temp))

    if method in ("logquant"):
        ql, qh = np.quantile(arr, [q_low, q_high])
        # a, b so that sigmoid(a*(ql-b))=eps and sigmoid(a*(qh-b))=1-eps
        L = _logit(eps)  # negative
        denom = max(qh - ql, 1e-12)
        a = (-2.0 * L) / denom
        b = ql - L / a
        return NormParams(method=method, q_low=q_low, q_high=q_high, eps=eps, a=float(a), b=float(b))

    raise ValueError(f"Unknown method: {method}")

# --------- mappers ---------
def map_values(x: np.ndarray, p: NormParams) -> np.ndarray:
    method = p.method.lower()
    if method == "zsig":
        z = (x - float(p.mean)) / max(float(p.std), 1e-6)
        z = z / max(float(p.temp or 1.0), 1e-6)
        y = _sigmoid(z)
    elif method in ("logquant"):
        y = _sigmoid(float(p.a) * (x - float(p.b)))
    else:
        raise ValueError(f"Unknown method for map: {method}")
    # clamp to open interval (0,1) for BCE stability
    return np.clip(y, 1e-6, 1.0 - 1e-6)

# --------- high-level API ---------
def fit_norm_params(entries_all: List[dict], reward_key: str, norm_type: str, **kw) -> NormParams:
    """Fit params from ALL entries (권장: 여러 데이터셋 합친 후 호출)."""
    arr = _flatten_rewards(entries_all, reward_key)
    return fit_params(arr, norm_type, **kw)

def apply_normalization(entries: List[dict], reward_key: str, params: NormParams, *, keep_raw: bool = True,  out_key: Optional[str] = None) -> List[dict]:
    out: List[dict] = []
    k_out = out_key or reward_key
    for e in entries:
        if reward_key not in e or not isinstance(e[reward_key], list):
            continue
        new_e = copy.deepcopy(e)
        if keep_raw and f"raw_{reward_key}" not in new_e:
            new_e[f"raw_{reward_key}"] = list(new_e[reward_key])
        vec = np.asarray(new_e[reward_key], dtype=float)
        new_e[k_out] = map_values(vec, params).tolist()
        out.append(new_e)
    return out

# --------- optional: summary for logging ---------
def global_info(entries: List[dict], reward_key: str) -> Tuple[float, float, Dict[str, float]]:
    arr = _flatten_rewards(entries, reward_key)
    if arr.size == 0:
        return 0.0, 1.0, {"min":0.0,"p25":0.0,"median":0.0,"p75":0.0,"max":0.0}
    mean = float(np.mean(arr))
    std  = float(np.std(arr)) if arr.size > 1 else 1.0
    q = {
        "min": float(np.min(arr)),
        "p25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "max": float(np.max(arr)),
    }
    return mean, max(std, 1e-6), q

# ──outlier_filtering ─────────────────────────────────────────────
def filter_entries_by_mi_steps(
    entries: List[Dict[str, Any]],
    reward_key: str,
    abs_max: Optional[float] = None,
    bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
    debug_sample: int = 3,
) -> List[Dict[str, Any]]:

    def _ok(v: Optional[float]) -> bool:
        if v is None: return False
        if abs_max is not None and abs(v) > float(abs_max): return False
        if bounds and reward_key in bounds:
            lo, hi = bounds[reward_key]
            if lo is not None and v < float(lo): return False
            if hi is not None and v > float(hi): return False
        return True

    kept_entries: List[Dict[str, Any]] = []
    total_steps = kept_steps = 0
    dropped_entries = 0
    none_count = 0
    none_samples: List[Any] = []

    for e in entries:
        # ── Case A: steps(dict list) ──
        if isinstance(e.get("steps"), list):
            steps = e["steps"]
            total_steps += len(steps)
            new_steps = []
            for s in steps:
                v = _get_step_value(s, reward_key)
                if v is None:
                    none_count += 1
                    if len(none_samples) < debug_sample:
                        none_samples.append({k: s.get(k) for k in list(s.keys())[:8]})
                if _ok(v):
                    new_steps.append(s)
            kept_steps += len(new_steps)
            if new_steps:
                e2 = dict(e)
                e2["steps"] = new_steps
                kept_entries.append(e2)
            else:
                dropped_entries += 1
            continue

        # ── Case B: completion(list of str) + per-step arrays ──
        comp = e.get("completion")
        if isinstance(comp, list):
            L = len(comp)
            total_steps += L
            # 우선순위: 정규화된 reward_key 배열 → raw_ 배열
            vals = None
            rk = e.get(reward_key)
            if isinstance(rk, list) and len(rk) == L:
                vals = [ _as_float(x) for x in rk ]
            if vals is None:
                raw_rk = e.get(f"raw_{reward_key}")
                if isinstance(raw_rk, list) and len(raw_rk) == L:
                    vals = [ _as_float(x) for x in raw_rk ]

            if vals is None:
                # 구조 미스매치 → 이 엔트리 전체 drop(스텝 카운트는 none으로 집계)
                none_count += L
                dropped_entries += 1
                continue

            mask = [ _ok(v) for v in vals ]
            kept_idx = [ i for i, m in enumerate(mask) if m ]
            kept_steps += len(kept_idx)

            if not kept_idx:
                dropped_entries += 1
                continue

            # 길이가 L인 per-step 배열들을 모두 필터링
            e2 = dict(e)
            for k, v in list(e.items()):
                if isinstance(v, list) and len(v) == L:
                    e2[k] = [ v[i] for i in kept_idx ]
            kept_entries.append(e2)
            continue

        # 알 수 없는 스키마 → 보수적으로 drop
        dropped_entries += 1

    dropped_steps = total_steps - kept_steps
    if total_steps > 0:
        print(f"[INFO] MI filter dropped {dropped_steps} steps ({dropped_steps/total_steps*100:.2f}%). Kept {kept_steps}.")
    if dropped_entries > 0:
        print(f"[INFO] MI filter dropped {dropped_entries} empty/invalid entries after step filtering.")
    else:
        print("[INFO] MI filter kept all entries.")

    if none_count > 0:
        print(f"[WARN] {none_count} steps had no '{reward_key}' value matched; check dataset keys (e.g., '{reward_key}', 'raw_{reward_key}').")
        if none_samples:
            print("[WARN] Example unmatched step snippets (truncated):")
            for i, samp in enumerate(none_samples, 1):
                print(f"  - unmatched[{i}]: {samp}")

    return kept_entries

# mse normalization # 
def fit_mse_mi_params(
    entries: List[dict],
    reward_key: str,
    q_low: float = 0.01,
    q_high: float = 0.99,
    eps: float = 1e-6,
    temp: float = 1.0,
) -> Dict[str, float]:
    """
    MSE 회귀용 MI 파라미터 추정: robust z-score (median/MAD) + quantile clipping.
    """
    y = _flatten_rewards(entries, reward_key)
    if y.size == 0:
        # 비어있으면 안전한 디폴트
        return {"lo": 0.0, "hi": 0.0, "med": 0.0, "scale": 1.0, "eps": eps, "temp": float(temp)}
    lo = float(np.quantile(y, q_low))
    hi = float(np.quantile(y, q_high))
    y_clip = np.clip(y, lo, hi)
    med = float(np.median(y_clip))
    mad = float(np.median(np.abs(y_clip - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < eps:
        # MAD가 0이면 표준편차 fallback
        std = float(np.std(y_clip))
        scale = max(std, eps)
    return {"lo": lo, "hi": hi, "med": med, "scale": scale, "eps": eps, "temp": float(temp)}

def _z_to_prob(z: np.ndarray, nonlinearity: str = "sigmoid") -> np.ndarray:
    nonlinearity = (nonlinearity or "sigmoid").lower()
    if nonlinearity == "sigmoid":
        return _sigmoid(z)
    elif nonlinearity == "tanh":
        return 0.5 * (np.tanh(z) + 1.0)
    else:
        raise ValueError(f"Unknown nonlinearity: {nonlinearity}")

def _mi_values_to_unit_interval(
    values: List[float],
    params: Dict[str, float],
    *,
    nonlinearity: str = "sigmoid",   # ← 기본을 sigmoid로
) -> List[float]:
    lo, hi = params["lo"], params["hi"]
    med, scale = params["med"], params["scale"]
    eps, temp = params.get("eps", 1e-6), params.get("temp", 1.0)

    out = []
    denom = (scale * temp) + eps
    for v in values:
        v_clip = min(max(float(v), lo), hi)
        z = (v_clip - med) / denom
        y = _z_to_prob(np.array([z], dtype=np.float64), nonlinearity=nonlinearity)[0]
        out.append(float(y))
    # BCE 안정성을 원한다면 clip, MSE면 굳이 안 해도 되지만 일관성 위해 유지
    return [float(np.clip(y, 1e-6, 1.0 - 1e-6)) for y in out]

def apply_mse_mi_normalization(
    entries: List[dict],
    reward_key: str,
    params: Dict[str, float],
    *,
    keep_raw: bool = True,
    out_key: Optional[str] = None,
    nonlinearity: str = "sigmoid",    # ← 새 파라미터
) -> List[dict]:
    out_key = out_key or reward_key
    for e in entries:
        if keep_raw and f"raw_{reward_key}" not in e and reward_key in e:
            e[f"raw_{reward_key}"] = e[reward_key]
        if isinstance(e.get(reward_key, None), list):
            e[out_key] = _mi_values_to_unit_interval(e[reward_key], params, nonlinearity=nonlinearity)
    return entries

# --- MSE-LOGIT 용: robust z-score만 적용 (tanh/sigmoid 없음) ---
def fit_mse_logit_params(
    entries: List[dict],
    reward_key: str,
    q_low: float = 0.01,
    q_high: float = 0.99,
    eps: float = 1e-6,
    temp: float = 1.0,
) -> Dict[str, float]:
    # winsorize + median/MAD
    y = _flatten_rewards(entries, reward_key)
    if y.size == 0:
        return {"lo": 0.0, "hi": 0.0, "med": 0.0, "scale": 1.0, "eps": eps, "temp": float(temp)}
    lo = float(np.quantile(y, q_low))
    hi = float(np.quantile(y, q_high))
    y_clip = np.clip(y, lo, hi)
    med = float(np.median(y_clip))
    mad = float(np.median(np.abs(y_clip - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < eps:
        std = float(np.std(y_clip))
        scale = max(std, eps)
    return {"lo": lo, "hi": hi, "med": med, "scale": scale, "eps": eps, "temp": float(temp)}

def apply_mse_logit_standardize(
    entries: List[dict],
    reward_key: str,
    params: Dict[str, float],
    *,
    keep_raw: bool = True,
    out_key: Optional[str] = None,
) -> List[dict]:
    out_key = out_key or reward_key
    lo, hi = params["lo"], params["hi"]
    med, scale = params["med"], params["scale"]
    eps, temp = params.get("eps", 1e-6), params.get("temp", 1.0)
    denom = (scale * temp) + eps

    outs: List[dict] = []
    for e in entries:
        vec = e.get(reward_key, None)
        if not isinstance(vec, list):
            continue
        e2 = copy.deepcopy(e)
        if keep_raw and f"raw_{reward_key}" not in e2:
            e2[f"raw_{reward_key}"] = list(vec)
        mapped = []
        for v in vec:
            v = float(v)
            v_clip = min(max(v, lo), hi)
            z = (v_clip - med) / denom  # tanh/sigmoid 없음!
            mapped.append(float(z))
        e2[out_key] = mapped
        outs.append(e2)
    return outs

# ========= PCMI calibration (MC prob ↔ PMI raw) ==============================
def _fit_sigmoid_quantile_match(pmi_arr: np.ndarray, mc_prob_arr: np.ndarray, q_low: float = 0.1, q_high: float = 0.9, clip: float = 50.0) -> Tuple[float, float]:
    """
    분위수 매칭으로 PMI→확률 맵핑의 (a,b) 추정:
    sigmoid(a*(x-b)) 가 되도록,
      x1=Q_low(PMI), x2=Q_high(PMI)
      y1=logit(Q_low(MC)), y2=logit(Q_high(MC))
    를 만족시키는 선형식을 a,b로 푼다.
    """
    if pmi_arr.size < 4 or mc_prob_arr.size < 4:
        # 데이터가 적으면 대충 안전한 기본값
        return 1.0, float(np.median(pmi_arr)) if pmi_arr.size else 0.0

    x1, x2 = np.quantile(pmi_arr, [q_low, q_high])
    p1, p2 = np.quantile(mc_prob_arr, [q_low, q_high])
    p1 = np.clip(p1, 1e-6, 1.0-1e-6); p2 = np.clip(p2, 1e-6, 1.0-1e-6)
    y1 = np.log(p1/(1.0-p1)); y2 = np.log(p2/(1.0-p2))

    denom = max(x2 - x1, 1e-12)
    a = (y2 - y1) / denom
    b = x1 - (y1 / a)
    # 수치 안정성
    a = float(np.clip(a, -clip, clip))
    b = float(np.clip(b, -clip, clip))
    return a, b

@dataclass
class PCMIParams:
    a: float
    b: float

def fit_pcmi_params(entries: List[dict], reward_key: str,
                    q_low: float = 0.1, q_high: float = 0.9) -> PCMIParams:
    """
    여러 엔트리에서 MC/PMI를 모아 전역 (a,b) 추정.
    """
    mc_all, pmi_all = [], []
    for e in entries:
        vec = e.get(reward_key)
        if not isinstance(vec, list): 
            continue
        mc, pmi = _split_pcmi(vec)
        if mc.size:  mc_all.append(mc)
        if pmi.size: pmi_all.append(pmi)
    mc_all  = np.concatenate(mc_all)  if mc_all  else np.zeros((0,), dtype=float)
    pmi_all = np.concatenate(pmi_all) if pmi_all else np.zeros((0,), dtype=float)
    a, b = _fit_sigmoid_quantile_match(pmi_all, mc_all, q_low=q_low, q_high=q_high)
    return PCMIParams(a=a, b=b)

def apply_pcmi_calibration(entries: List[dict], reward_key: str, pcmi: PCMIParams,
                           *, keep_raw: bool = True, out_key: Optional[str] = None) -> List[dict]:
    """
    한 엔트리 내에서 MC는 그대로, PMI는 sigmoid(a*(x-b))로 변환.
    최종 타깃은 (0,1) 확률 → BCEWithLogitsLoss 대상.
    """
    out_key = out_key or reward_key
    outs: List[dict] = []
    for e in entries:
        vec = e.get(reward_key)
        if not isinstance(vec, list):
            continue
        e2 = copy.deepcopy(e)
        if keep_raw and f"raw_{reward_key}" not in e2:
            e2[f"raw_{reward_key}"] = list(vec)
        mapped = []
        for v in vec:
            v = float(v)
            if 0.0 <= v <= 1.0:    # MC 그대로 사용
                y = np.clip(v, 1e-6, 1.0-1e-6)
            else:                  # PMI → 확률로
                z = pcmi.a * (v - pcmi.b)
                z = np.clip(z, -50.0, 50.0)
                y = 1.0 / (1.0 + np.exp(-z))
                y = np.clip(y, 1e-6, 1.0-1e-6)
            mapped.append(float(y))
        e2[out_key] = mapped
        outs.append(e2)
    return outs
