import re, math
from fractions import Fraction
from typing import Optional

def _sanitize(text: str) -> str:
    if not text:
        return ""
    s = text.strip()
    s = s.replace(",", "")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*-\s*", "-", s)
    return s

def _to_float(s: str) -> Optional[float]:
    s = _sanitize(s)
    # 혼합수 "a b/c"
    m = re.fullmatch(r"(-?\d+)\s+(\d+)/(\d+)", s)
    if m:
        w, n, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if d != 0:
            frac = Fraction(n, d)
            return float(Fraction(w,1) + (frac if w >= 0 else -frac))
    # a/b
    m = re.fullmatch(r"(-?\d+)/(-?\d+)", s)
    if m and m.group(2) != "0":
        return float(Fraction(int(m.group(1)), int(m.group(2))))
    # 퍼센트
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*%", s)
    if m:
        return float(m.group(1)) / 100.0
    try:
        return float(s)
    except Exception:
        return None

def numeric_equiv_fallback(a: str, b: str) -> bool:
    a, b = _sanitize(a), _sanitize(b)
    if a == b:
        return True
    av, bv = _to_float(a), _to_float(b)
    if av is not None and bv is not None:
        return math.isclose(av, bv, rel_tol=1e-6, abs_tol=1e-9)
    return False
