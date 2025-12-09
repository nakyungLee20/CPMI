import re
import math
from typing import Any, List, Optional, Tuple, Union, Dict
from fractions import Fraction
from decimal import Decimal, InvalidOperation
from sympy.parsing.latex import parse_latex
import sympy as sp 
import re as regex
import platform, signal, time
from contextlib import contextmanager

class OmniScorer:
    _UNITS_RX = re.compile(r"\\text\{[^}]*\}\s*$")
    _DEG_RX   = re.compile(r"\^\{?\\?circ\}?|°")
    _MOD_TAIL = re.compile(r"\s*(?:\\?mod|\\bmod)\s*[0-9]+\s*$", re.IGNORECASE)
    _UNIT_WORD_TAIL = re.compile(r"\s*[A-Za-z][A-Za-z.%\-]*(?:\s+[A-Za-z][A-Za-z.%\-]*)*\s*$")

    _RATIO_RX = re.compile(r"^\s*([+-]?\d+)\s*:\s*([+-]?\d+)\s*$")
    _ABS_LATEX = re.compile(r"\\left\|([^|]+)\\right\|")

    _LATEX_STRIP = [
        (re.compile(r"\\left\s*"), ""),
        (re.compile(r"\\right\s*"), ""),
        (re.compile(r"\\!"), ""),
        (re.compile(r"\\mathrm\{([^}]*)\}"), r"\1"),
        (re.compile(r"\\operatorname\{([^}]*)\}"), r"\1"),
        (re.compile(r"\\text\{([^}]*)\}"), r"\1"),
    ]

    ANS_LINE = re.compile(r"^\s*answer\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
    HINT_ANS = re.compile(r"(?i)\b(?:the\s+)?(?:final\s+)?answer\s*(?:is\s+equal\s+to|is|=|equals|:)\s*([^\n]+)")

    _LEAD_PUNCT  = re.compile(r"^[([\{\s]+")
    _TRAIL_PUNCT = re.compile(r"[)\].,;:\s]+$")

    # \boxed, \boxed*, \fbox, \bbox
    BOXED_OPEN = re.compile(r"\\boxed\*?\s*\{|\s\\fbox\s*\{|\s\\bbox\s*\{")

    # intervals/sets
    IN_SYMB = re.compile(r"\\?in\b")
    INTERVAL_RX = re.compile(r"([\[(])\s*([^,]+?)\s*,\s*([^)]*[^)\]])\s*([)\]])")
    SET_BRACES_RX = re.compile(r"^\s*\{(.+)\}\s*$")
    CSV_SIMPLE_RX = re.compile(r"^\s*[-+]?\d+(\s*,\s*[-+]?\d+)+\s*$")

    # numbers
    DEC_SCI = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?|[-+]?\.\d+(?:[eE][+-]?\d+)?")
    MIXED_FRAC = re.compile(r"([+-]?\d+)\s+(\d+)\s*/\s*(\d+)")
    PURE_FRAC  = re.compile(r"([+-]?\d+)\s*/\s*(\d+)")

    NO_SOLUTION_TOKENS = {"no solution", "nosolution", "none", "empty set", "emptyset", "∅", "varnothing", "\\varnothing", "{}"}
    DNE_TOKENS = {"dne", "does not exist", "undefined", "not defined"}
    INF_TOKENS = {"infinity", "\\infty", "∞"}

    # tolerant container helpers
    _SPACE_AFTER_COMMA = re.compile(r",\s*")
    _MULTI_SPACE = re.compile(r"\s+")
    _BARE_INF = re.compile(r"(?<!\\)inf(?:inity)?|∞|oo|\\infty", re.I)

    _ASSIGN_RE = re.compile(r"^\s*[a-zA-Z]\s*=\s*(.+)$")
    _MAT_RX = re.compile(r"\\begin\{p(?:b)?matrix\}(.+?)\\end\{p(?:b)?matrix\}", re.DOTALL)

    # MCQ choices
    _MCQ_RX = re.compile(r"\b([A-E])\b")

    def __init__(self, atol: float = 1e-6, sympy_timeout_s: float = 0.25,
                 max_expr_chars: int = 360, enable_parse_latex: bool = False):
        self.atol = float(atol)
        self.sympy_timeout_s = float(sympy_timeout_s)
        self.max_expr_chars = int(max_expr_chars)
        self.enable_parse_latex = bool(enable_parse_latex)

    # ----------------------------- helpers -----------------------------
    def _strip_wrappers(self, s: str) -> str:
        s = self._LEAD_PUNCT.sub("", s)
        s = self._TRAIL_PUNCT.sub("", s)
        return s.strip()

    def _basic_clean(self, s: str) -> str:
        if not s:
            return ""
        t = s.replace("\n", " ")
        t = re.sub(r"\$+", "", t)
        t = re.sub(r"\\\(|\\\)|\\\[|\\\]", "", t)
        t = t.replace("\\$", "$")
        t = re.sub(r"\\(?![A-Za-z])", "", t)
        for rx, rep in self._LATEX_STRIP:
            t = rx.sub(rep, t)
        t = self._UNITS_RX.sub("", t)
        t = self._DEG_RX.sub("", t)
        t = t.replace("%", "")
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def _fix_fracs_sqrt(self, s: str) -> str:
        def _fix_fracs(u: str) -> str:
            u = re.sub(r"\\[dt]frac\b", r"\\frac", u)
            u = re.sub(r"\\frac\s*\{\s*([^{}]+?)\s*\}\s*([^\s{])", r"\\frac{\1}{\2}", u)
            u = re.sub(r"\\frac\s*([^\s{])\s*\{\s*([^{}]+?)\s*\}", r"\\frac{\1}{\2}", u)
            u = re.sub(r"\\frac\s*([^\s{])\s*([^\s{])", r"\\frac{\1}{\2}", u)
            return u

        def _fix_sqrt(u: str) -> str:
            u = re.sub(r'(?<!\\)sqrt\s*\{', r'\\sqrt{', u)
            if "\\sqrt" not in u:
                return u
            parts = u.split("\\sqrt")
            out = parts[0]
            for tail in parts[1:]:
                if tail and tail[0] != "{":
                    out += "\\sqrt{" + tail[0] + "}" + tail[1:]
                else:
                    out += "\\sqrt" + tail
            return out

        return _fix_sqrt(_fix_fracs(s))

    def _unbox_last(self, text: str) -> Optional[str]:
        if not text:
            return None
        i = 0
        payload = None
        while True:
            m = self.BOXED_OPEN.search(text, i)
            if not m:
                break
            j = m.end()
            depth = 1
            buf: List[str] = []
            while j < len(text) and depth > 0:
                ch = text[j]
                if ch == '\\':
                    if j + 1 < len(text):
                        buf.append(text[j:j+2])
                        j += 2
                        continue
                if ch == '{':
                    depth += 1
                    buf.append(ch)
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        j += 1
                        break
                    buf.append(ch)
                else:
                    buf.append(ch)
                j += 1
            if depth == 0:
                payload = ''.join(buf)
                i = j
            else:
                break
        if payload is None:
            return None
        s = self._fix_fracs_sqrt(payload)
        s = self._basic_clean(s)
        return s.strip()

    def _expand_pm(self, expr: str) -> List[str]:
        if "±" not in expr:
            return [expr]
        a = expr.replace("±", "+", 1)
        b = expr.replace("±", "-", 1)
        if "±" in a or "±" in b:
            acc: List[str] = []
            acc.extend(self._expand_pm(a))
            acc.extend(self._expand_pm(b))
            return acc
        return [a, b]

    def _last_latex_token(self, s: str) -> Optional[str]:
        if not s:
            return None
        u = self._fix_fracs_sqrt(s)
        patt = re.compile(r"(\\frac\s*\{\s*[^{}]+\s*\}\s*\{\s*[^{}]+\s*\}|\\sqrt\s*\{\s*[^{}]+\s*\})")
        hits = list(patt.finditer(u))
        if hits:
            tok = hits[-1].group(0)
            return self._strip_wrappers(self._basic_clean(tok))
        return None

    def _maybe_take_whole_expr(self, t: str) -> Optional[str]:
        u = t.strip()
        if len(u) <= 80 and re.search(r"[+\-*/^]", u) and ("/" in u or "\\" in u) and re.search(r"[A-Za-z]", u):
            return u
        return None

    @contextmanager
    def _time_limit(self, seconds: float):
        if seconds is None or seconds <= 0 or platform.system() == "Windows":
            yield
            return

        def _handler(signum, frame):
            raise TimeoutError("SymPy step timed out")

        old = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)
    
    # ------------------------- public: extract --------------------------
    def extract_pred(self, text: str) -> str:
        if not text:
            return ""
        # (1) boxed if present
        b = self._unbox_last(text)
        if b:
            if self.IN_SYMB.search(b):
                m = self.INTERVAL_RX.search(b)
                if m:
                    return m.group(0)
            csv = self._csv_to_tuple_str(b)
            return csv or b
        # (2) explicit Answer-lines
        m = self.ANS_LINE.search(text)
        if m:
            return self._strip_wrappers(self._fix_fracs_sqrt(m.group(1)))
        m2 = self.HINT_ANS.search(text)
        if m2:
            return self._strip_wrappers(self._fix_fracs_sqrt(m2.group(1)))
        # (3) explicit MCQ letter in the end (A–E)
        tail = text.strip().upper()
        mcq = self._choice_answer_clean(tail)
        if mcq:
            return mcq
        # (4) last latex-y token
        tok = self._last_latex_token(text)
        if tok:
            return tok
        # (5) try interval unions "A \cup B"
        t = self._basic_clean(text)
        whole = self._maybe_take_whole_expr(t)
        if whole:
            return whole
        parts = re.split(r"\s*\\cup\s*", t)
        if len(parts) > 1:
            ivs: List[str] = []
            ok = True
            for p in parts:
                p = p.strip()
                m = self.INTERVAL_RX.search(p)
                if not m:
                    ok = False
                    break
                ivs.append(self._strip_wrappers(m.group(0)))
            if ok and ivs:
                return r" \cup ".join(ivs)
        # (6) single interval
        inter = self._extract_interval_str(t)
        if inter:
            return inter
        # (7) last number-like token (mixed -> pure -> decimal)
        for rx in (self.MIXED_FRAC, self.PURE_FRAC, self.DEC_SCI):
            hits = list(rx.finditer(t))
            if hits:
                return self._strip_wrappers(hits[-1].group(0))
        # (8) otherwise return cleaned tail
        return t.strip()

    def extract_gold(self, rec: Any) -> str:
        """Per request: simply read the `answer` field from an Omni record. Missing/odd cases are sanitized lightly (stringify + strip)."""
        ans = None
        if isinstance(rec, dict):
            ans = rec.get("answer")
        else:
            # allow passing the answer directly
            ans = rec
        if ans is None:
            return ""
        s = str(ans)
        # Some Omni answers are lists/tuples (e.g., coordinates) pre-formatted; keep as-is
        return s.strip()

    # ---------------------- normalization utilities ---------------------
    NO_SOLUTION = "<NO_SOLUTION>"; DNE = "<DNE>"; INF = "<INF>"

    def _normalize_special_tokens(self, s: str) -> str:
        t = self._basic_clean(s).lower().strip()
        if t in self.NO_SOLUTION_TOKENS: return self.NO_SOLUTION
        if t in self.DNE_TOKENS: return self.DNE
        if t in self.INF_TOKENS: return self.INF
        return self._basic_clean(s)

    def _choice_answer_clean(self, pred: str) -> Optional[str]:
        p = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
        hits = self._MCQ_RX.findall(p.upper())
        if hits:
            return hits[-1]
        return None

    def _ratio_to_frac(self, s: str) -> Optional[str]:
        m = self._RATIO_RX.fullmatch(s)
        if not m:
            return None
        a, b = int(m.group(1)), int(m.group(2))
        if b == 0:
            return None
        f = Fraction(a, b)
        return f"{f.numerator}/{f.denominator}"

    def _strip_assignment(self, s: str) -> str:
        m = self._ASSIGN_RE.match(s)
        return m.group(1).strip() if m else s

    def _normalize_atomic(self, x: str) -> str:
        if x is None:
            return ""
        s = self._strip_assignment(str(x))
        b = self._unbox_last(s)
        if b is not None:
            s = b
        s = self._strip_wrappers(s)
        s = self._MOD_TAIL.sub("", s)
        s = self._UNITS_RX.sub("", s)
        s = self._DEG_RX.sub("", s)
        # remove trailing English unit-words if the core is purely numeric/frac
        s_num_tail = self._UNIT_WORD_TAIL.sub("", s)
        if s_num_tail != s:
            if (re.fullmatch(r"\s*[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*\Z", s_num_tail)
                or re.fullmatch(r"\s*[-+]?\.\d+\s*\Z", s_num_tail)
                or re.fullmatch(r"\s*([+-]?\d+)\s*/\s*(\d+)\s*\Z", s_num_tail)
                or re.fullmatch(r"\s*\\frac\s*\{\s*[^{}]+\s*\}\s*\{\s*[^{}]+\s*\}\s*\Z", s_num_tail)):
                s = s_num_tail
        r = self._ratio_to_frac(s)
        if r is not None:
            return r
        m = re.fullmatch(r"\s*([+-]?\d+)\s+(\d+)\s*/\s*(\d+)\s*\Z", s)
        if m:
            whole, num, den = int(m.group(1)), int(m.group(2)), int(m.group(3))
            sign = -1 if whole < 0 else 1
            whole = abs(whole)
            try:
                frac = Fraction(whole * den + num, den) * sign
                return f"{frac.numerator}/{frac.denominator}"
            except ZeroDivisionError:
                return ""
        m = re.fullmatch(r"\s*([+-]?\d+)\s*/\s*(\d+)\s*\Z", s)
        if m:
            try:
                frac = Fraction(int(m.group(1)), int(m.group(2)))
                return f"{frac.numerator}/{frac.denominator}"
            except ZeroDivisionError:
                return ""
        s_num_like = re.fullmatch(r"\s*[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*\Z", s)
        if s_num_like:
            try:
                v = float(Decimal(s.replace(",", "")))
                if math.isfinite(v) and abs(v - round(v)) < 1e-12:
                    return str(int(round(v)))
                return f"{v:.12g}"
            except (InvalidOperation, ValueError):
                pass
        return s

    def normalize_number(self, x: str) -> str:
        return self._normalize_atomic(x)

    # -------------------- sympy-friendly normalization ------------------
    def _balance_brackets(self, s: str) -> str:
        if not s:
            return s
        openers = {"(": ")", "[": "]", "{": "}"}
        closers = {")": "(", "]": "[", "}": "{"}
        stack: List[str] = []
        out: List[str] = []
        for ch in s:
            if ch in openers:
                stack.append(ch)
                out.append(ch)
            elif ch in closers:
                if stack and stack[-1] == closers[ch]:
                    stack.pop()
                    out.append(ch)
                else:
                    continue
            else:
                out.append(ch)
        while out and stack:
            ch = out.pop()
            if ch in openers:
                stack.pop()
                continue
            out.append(ch)
            break
        return "".join(out)

    def _rebalance_parens(self, s: str) -> str:
        if not s:
            return s
        opens, closes = s.count("("), s.count(")")
        if closes > opens:
            s = "(" * (closes - opens) + s
        opens, closes = s.count("("), s.count(")")
        if opens > closes:
            s = s + ")" * (opens - closes)
        return s

    def _latex_cleanup(self, expr: str) -> str:
        if not expr:
            return ""
        t = self._balance_brackets(self._basic_clean(expr))
        t = self._rebalance_parens(t)
        t = re.sub(r"\\[dt]frac", r"\\frac", t)
        t = self._fix_fracs_sqrt(t)
        t = t.replace("\\pi", "pi")
        t = re.sub(r"\\frac\s*\{\s*([^{}]+?)\s*\}\s*\{\s*([^{}]+?)\s*\}", r"(\1)/(\2)", t)
        t = re.sub(r"(?:\\sqrt|(?<!\\)sqrt)\s*\{([^{}]+)\}", r"sqrt(\1)", t)
        t = t.replace("\\cdot", "*").replace("\\times", "*")
        t = t.replace("^", "**").replace("\\pm", "±")
        t = self._ABS_LATEX.sub(lambda m: f"Abs({m.group(1)})", t)
        t = re.sub(r"\\binom\{([^}]*)\}\{([^}]*)\}", r"binomial(\1,\2)", t)
        t = re.sub(r"\{([^}]*)\}\\choose\{([^}]*)\}", r"binomial(\1,\2)", t)
        t = re.sub(r'(\d)\s*(?=\\[A-Za-z])', r'\1*', t)
        t = re.sub(r'(\d)\s*(?=[A-Za-z(])', r'\1*', t)
        t = re.sub(r'(\))\s*(?=[A-Za-z(])', r'\1*', t)
        return t.strip()

    def _interval_to_sympy(self, s: str):
        m = self.INTERVAL_RX.search(s)
        if not m:
            return None
        L, a_raw, b_raw, R = m.groups()
        left_open  = (L == '(')
        right_open = (R == ')')

        def _parse_end(u: str):
            u = self._basic_clean(u)
            if u.lower() in {"-inf", "-infty", "-∞", "-oo", "-\\infty"}: return -sp.oo if sp else None
            if u.lower() in {"inf", "+inf", "infty", "∞", "oo", "\\infty"}: return sp.oo if sp else None
            try:
                if parse_latex is not None and re.search(r"\\[a-zA-Z]+", u):
                    return parse_latex(self._basic_clean(u))
            except Exception:
                pass
            try:
                return sp.sympify(self._latex_cleanup(u), rational=True) if sp else None
            except Exception:
                return None

        a = _parse_end(a_raw); b = _parse_end(b_raw)
        if a is None or b is None or sp is None:
            return None
        try:
            return sp.Interval(a, b, left_open=left_open, right_open=right_open)
        except Exception:
            return None

    def _safe_sympy_parse(self, expr: str):
        if sp is None or expr is None:
            return None
        raw = expr.strip()
        if not raw:
            return None
        # 길이/문자 가드: 너무 긴 식, 수상한 LaTeX는 심볼릭 생략
        if len(raw) > self.max_expr_chars:
            return None
        # 제한된 라텍스만 허용 (frac/sqrt/pi/infty/기본 연산만)
        looks_latex = bool(re.search(r"\\[a-zA-Z]+", raw))
        allow_cmd = re.compile(r"\\(frac|sqrt|infty|cdot|times|pm|pi|left|right|bmod|mod)\b")

        with self._time_limit(self.sympy_timeout_s):
            # 1) 간단한 인터벌/유니온 먼저
            iv = self._interval_to_sympy(raw)
            if iv is not None:
                return iv
            # 2) LaTeX 파싱은 기본 꺼두고, 켜진 경우 + 제한된 명령만
            if looks_latex and self.enable_parse_latex and allow_cmd.search(raw):
                try:
                    return parse_latex(self._latex_cleanup(raw))
                except Exception:
                    pass
            # 3) 일반 sympify
            try:
                return sp.sympify(self._latex_cleanup(raw), rational=True)
            except Exception:
                return None
    
    def _sympy_parse(self, expr: str):
        if expr is None:
            return None
        raw = expr.strip()
        if not raw:
            return None
        parts = re.split(r"\s*\\cup\s*", raw)
        if len(parts) > 1 and sp is not None:
            ivs = []
            for p in parts:
                p = p.strip()
                iv = self._interval_to_sympy(p)
                if iv is None:
                    try:
                        iv = sp.sympify(self._latex_cleanup(p), rational=True)
                    except Exception:
                        iv = None
                if iv is None:
                    return None
                ivs.append(iv)
            try:
                return sp.Union(*ivs)  # type: ignore[attr-defined]
            except Exception:
                return None
        iv = self._interval_to_sympy(raw)
        if iv is not None:
            return iv
        looks_latex = bool(re.search(r"\\[a-zA-Z]+", raw))
        try:
            if looks_latex and parse_latex is not None:
                return parse_latex(self._latex_cleanup(raw))
        except Exception:
            pass
        try:
            return sp.sympify(self._latex_cleanup(raw), rational=True) if sp else None
        except Exception:
            return None

    # def _sympy_equal(self, a_str: str, b_str: str) -> bool:
    #     if sp is None:
    #         return False
    #     a = self._safe_sympy_parse(a_str)
    #     b = self._safe_sympy_parse(b_str)
    #     # a = self._sympy_parse(a_str)
    #     # b = self._sympy_parse(b_str)
    #     if a is None or b is None:
    #         return False
    #     try:
    #         if isinstance(a, (sp.Set, sp.Interval)) or isinstance(b, (sp.Set, sp.Interval)):
    #             return sp.simplify(a) == sp.simplify(b)
    #     except Exception:
    #         pass
    #     try:
    #         if sp.simplify(a - b) == 0:
    #             return True
    #     except Exception:
    #         pass
    #     try:
    #         vars = sorted(list((a.free_symbols | b.free_symbols)), key=lambda s: s.name)  # type: ignore[attr-defined]
    #         if not vars:
    #             return False
    #         for val in [-2, -1, 0, 1, 2, 3]:
    #             subs = {v: val for v in vars}
    #             av = complex(a.evalf(subs=subs))  # type: ignore
    #             bv = complex(b.evalf(subs=subs))  # type: ignore
    #             if abs(av - bv) > 1e-8:
    #                 return False
    #         return True
    #     except Exception:
    #         return False

    def _sympy_equal(self, a_str: str, b_str: str) -> bool:
        """Symbolic equality with timeouts+guards while preserving original coverage."""
        if sp is None:
            return False

        parse_fn = getattr(self, "_safe_sympy_parse", None)
        if not callable(parse_fn):
            parse_fn = getattr(self, "_sympy_parse", None)
        if not callable(parse_fn):
            return False

        a = parse_fn(a_str)
        b = parse_fn(b_str)
        
        # fallback parsing
        if (a is None or b is None) and hasattr(self, "_sympy_parse"):
            try:
                with _ctx():
                    a2 = self._sympy_parse(a_str)
                    b2 = self._sympy_parse(b_str)
                a = a if a is not None else a2
                b = b if b is not None else b2
            except Exception:
                pass

        if a is None or b is None:
            return False

        try:
            from contextlib import nullcontext
        except Exception:
            nullcontext = None  # 아주 옛 파이썬이 아니라면 거의 안옴
        time_limit_ctx = getattr(self, "_time_limit", None)
        timeout_s = float(getattr(self, "sympy_timeout_s", 0.25))

        def _ctx():
            return time_limit_ctx(timeout_s) if callable(time_limit_ctx) else nullcontext()
        
        # 1) Set/Interval 등 집합류는 equals 우선 (simplify보다 안전)
        try:
            if isinstance(a, (sp.Set, sp.Interval)) or isinstance(b, (sp.Set, sp.Interval)):
                with _ctx():
                    try:
                        # equals가 있는 타입이면 사용
                        if hasattr(a, "equals"):
                            return bool(a.equals(b))
                    except Exception:
                        pass
                    try:
                        if hasattr(b, "equals"):
                            return bool(b.equals(a))
                    except Exception:
                        pass
                    # 최후의 수단으로만 simplify 비교 (타임아웃 보호)
                    return bool(sp.simplify(a) == sp.simplify(b))
        except Exception:
            # 집합 비교 단계 실패 → 다음 단계로 진행
            pass

        # 2) 정확 동치: a - b == 0  (nsimplify 우선, 실패시 simplify)
        try:
            with _ctx():
                diff = a - b
                try:
                    # 간단식에서 보통 simplify보다 빠름
                    diff_simpler = sp.nsimplify(diff, rational=True)
                except Exception:
                    diff_simpler = None
                if diff_simpler == 0:
                    return True
                try:
                    if sp.simplify(diff) == 0:
                        return True
                except Exception:
                    pass
        except Exception:
            pass

        # 3) 수치 대입 검사 (원래 로직 유지 + 각 포인트별 타임아웃/안전성 강화)
        try:
            vars_ = sorted(list((getattr(a, "free_symbols", set()) | getattr(b, "free_symbols", set()))),
                        key=lambda s: s.name)
            # 자유변수 없으면 상수식: evalf로 직접 비교
            if not vars_:
                with _ctx():
                    try:
                        av = complex(a.evalf())
                        bv = complex(b.evalf())
                        return abs(av - bv) <= 1e-8  # 원래 임계값 유지
                    except Exception:
                        return False

            # 샘플 포인트들: 실패하는 포인트는 건너뛰고, 최소 1회 성공해야 True 가능
            sample_points = (-2, -1, 0, 1, 2, 3)
            ok_evals = 0
            for val in sample_points:
                try:
                    with _ctx():
                        subs = {v: val for v in vars_}
                        av = complex(a.evalf(subs=subs))  # type: ignore
                        bv = complex(b.evalf(subs=subs))  # type: ignore
                        ok_evals += 1
                        if abs(av - bv) > 1e-8:  # 원래 임계값 유지
                            return False
                except Exception:
                    # 이 포인트는 평가 불가 → 다음 포인트 시도
                    continue
            # 하나도 성공적으로 평가되지 않았다면 보수적으로 불일치 처리
            return ok_evals > 0
        except Exception:
            return False
    
    # ------------------------ containers handling -----------------------
    def _extract_interval_str(self, s: str) -> Optional[str]:
        if not s:
            return None
        cand = None
        for m in self.INTERVAL_RX.finditer(s):
            cand = m.group(0)
        return cand

    def _csv_to_tuple_str(self, s: str) -> Optional[str]:
        t = self._basic_clean(s)
        if self.CSV_SIMPLE_RX.fullmatch(t):
            return f"({t})"
        return None

    def _split_top_level_commas(self, s: str) -> List[str]:
        items, buf, depth = [], [], 0
        for ch in s:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth = max(0, depth - 1)
            if ch == "," and depth == 0:
                items.append("".join(buf)); buf = []
            else:
                buf.append(ch)
        items.append("".join(buf))
        return [x.strip() for x in items if x.strip()]

    def _as_container(self, s: str) -> Tuple[Optional[str], List[str]]:
        t = self._basic_clean(s)
        if not t:
            return None, []
        if t.startswith("{") and t.endswith("}"):
            return "set", self._split_top_level_commas(t[1:-1])
        def _is_interval_piece(piece: str) -> bool:
            return bool(self.INTERVAL_RX.fullmatch(piece.strip()))
        if "\\cup" in t:
            parts = re.split(r"\s*\\cup\s*", t)
            if parts and all(_is_interval_piece(p) for p in parts):
                return "interval", [t]
        else:
            if _is_interval_piece(t):
                return "interval", [t]
        if t.startswith("(") and t.endswith(")"):
            items = self._split_top_level_commas(t[1:-1])
            if len(items) >= 2:
                return "tuple", items
        csv = self._csv_to_tuple_str(t)
        if csv:
            return "tuple", self._split_top_level_commas(csv[1:-1])
        return None, []

    def _normalize_tuple_like(self, s: str) -> Optional[str]:
        if not s:
            return None
        t = s.strip()
        if t.startswith("(") and t.endswith(")"):
            items = self._split_top_level_commas(t[1:-1])
            if len(items) >= 2:
                inner = ",".join(x.strip() for x in items)
                inner = self._MULTI_SPACE.sub(" ", inner)
                return f"({inner})"
        return None

    def _normalize_interval_like(self, s: str) -> Optional[str]:
        if not s:
            return None
        t = self._balance_brackets(s.strip())
        t = re.sub(r"-?\\frac\s*\{\s*\}\s*\{\s*1\s*\}\s*2", "-1/2", t)
        if re.fullmatch(r"\s*0\s*,\s*[^)\]]+\s*\)?\]?\s*", t):
            right = t.split(",", 1)[1].strip()
            right = self._BARE_INF.sub(lambda m: r"\\infty", right)
            right = right.rstrip(")]")
            return f"[0,{right})"
        parts = re.split(r"\s*\\cup\s*", t)
        ivs: List[str] = []
        for p in parts:
            p = p.strip()
            m = re.search(r"([\[(])\s*([^,]+)\s*,\s*([^\])\)]+)\s*([\])])", p)
            if not m:
                m2 = re.search(r"^\s*(?P<L>[\[(])?\s*(?P<a>[^,]+)\s*,\s*(?P<b>[^\])\)]+)\s*(?P<R>[\])])?\s*$", p)
                if not m2:
                    return None
                L = m2.group("L") or "("
                R = m2.group("R") or ")"
                a = m2.group("a"); b = m2.group("b")
            else:
                L, a, b, R = m.groups()
            a = re.sub(r"\s+", "", a.strip())
            b = re.sub(r"\s+", "", b.strip())
            a = self._BARE_INF.sub(lambda m: "-\\infty" if a.lstrip().startswith("-") else "\\infty", a)
            b = self._BARE_INF.sub(lambda m: "\\infty", b)
            ivs.append(f"{L}{a},{b}{R}")
        return r" \\cup ".join(ivs) if ivs else None

    def _containers_equivalent(self, p: str, g: str) -> Optional[bool]:
        pt = self._normalize_tuple_like(p); gt = self._normalize_tuple_like(g)
        if pt and gt:
            if pt == gt:
                return True
            return None
        pi = self._normalize_interval_like(p); gi = self._normalize_interval_like(g)
        if pi and gi:
            if pi == gi:
                return True
            return None
        return None

    # ----------------------------- matrix compare -----------------------
    def _parse_matrix(self, s: str) -> Optional[List[List[str]]]:
        m = self._MAT_RX.search(s)
        if not m:
            return None
        body = m.group(1)
        rows = [r.strip() for r in re.split(r"\\\\", body) if r.strip()]
        mat: List[List[str]] = []
        for r in rows:
            cols = [c.strip() for c in r.split("&")]
            mat.append(cols)
        return mat

    def _cell_equal(self, a: str, b: str) -> bool:
        pa = self._normalize_atomic(a); pb = self._normalize_atomic(b)
        if pa == pb and pa != "":
            return True
        na = self._parse_numeric(pa); nb = self._parse_numeric(pb)
        if na is not None and nb is not None:
            if isinstance(na, Fraction) and isinstance(nb, Fraction):
                return na == nb
            try:
                va = float(na) if not isinstance(na, float) else na
                vb = float(nb) if not isinstance(nb, float) else nb
                return math.isfinite(va) and math.isfinite(vb) and abs(va - vb) <= self.atol
            except Exception:
                pass
        return self._sympy_equal(pa, pb)

    def _maybe_compare_matrix(self, p: str, g: str) -> Optional[bool]:
        mp = self._parse_matrix(p)
        mg = self._parse_matrix(g)
        if mp is None or mg is None:
            return None
        if len(mp) != len(mg) or any(len(r1) != len(r2) for r1, r2 in zip(mp, mg)):
            return False
        for r1, r2 in zip(mp, mg):
            for a, b in zip(r1, r2):
                if not self._cell_equal(a, b):
                    return False
        return True

    # ---------------------------- numeric parsing -----------------------
    def _parse_numeric(self, s: str):
        if s is None:
            return None
        t = str(s).strip()
        if not t:
            return None
        if t in {self.NO_SOLUTION, self.DNE, self.INF}:
            return t
        m = re.fullmatch(r"([+-]?\d+)\s+(\d+)\s*/\s*(\d+)", t)
        if m:
            whole, num, den = int(m.group(1)), int(m.group(2)), int(m.group(3))
            sign = -1 if whole < 0 else 1
            whole = abs(whole)
            try:
                return Fraction(whole * den + num, den) * sign
            except ZeroDivisionError:
                return None
        m = re.fullmatch(r"([+-]?\d+)\s*/\s*(\d+)", t)
        if m:
            try:
                return Fraction(int(m.group(1)), int(m.group(2)))
            except ZeroDivisionError:
                return None
        try:
            v = float(Decimal(t))
            return v
        except (InvalidOperation, ValueError):
            return None

    # ----------------------------- public: grade ------------------------
    def grade(self, pred: Union[bool, float, str], gold: Union[float, str]) -> bool:
        if pred is None or gold is None:
            return False
        # direct string equality (case-insensitive, trimmed) for simple cases
        if str(pred).strip().lower() == str(gold).strip().lower():
            return True
        # MCQ letter match (A–E)
        g = str(gold).strip().upper()
        p_letter = self._choice_answer_clean(str(pred))
        if g in {"A","B","C","D","E"} and p_letter == g:
            return True

        p0 = self._normalize_special_tokens(str(pred))
        g0 = self._normalize_special_tokens(str(gold))

        # special tokens
        if p0 in {self.NO_SOLUTION, self.DNE, self.INF} or g0 in {self.NO_SOLUTION, self.DNE, self.INF}:
            return p0 == g0

        # matrices
        mat_eq = self._maybe_compare_matrix(p0, g0)
        if mat_eq is not None:
            return bool(mat_eq)

        # container tolerant equality
        tol = self._containers_equivalent(p0, g0)
        if tol is True:
            return True

        # container structural equality
        pk, pitems = self._as_container(p0)
        gk, gitems = self._as_container(g0)
        if pk and gk:
            if pk != gk:
                return False
            if pk == "set":
                used = [False] * len(gitems)
                for p in pitems:
                    hit = False
                    for j, gx in enumerate(gitems):
                        if not used[j] and self.grade(p, gx):
                            used[j] = True; hit = True; break
                    if not hit:
                        return False
                return all(used) and len(pitems) == len(gitems)
            if pk == "tuple":
                if len(pitems) != len(gitems):
                    return False
                return all(self.grade(p, g) for p, g in zip(pitems, gitems))
            if pk == "interval":
                return self._sympy_equal(p0, g0)

        # scalar branch with ± expansions
        for pc in self._expand_pm(p0):
            pn = self._normalize_atomic(pc)
            for gc in self._expand_pm(g0):
                gn = self._normalize_atomic(gc)
                if pn == gn and pn != "":
                    return True
                p_num = self._parse_numeric(pn)
                g_num = self._parse_numeric(gn)
                if p_num is not None and g_num is not None:
                    if isinstance(p_num, Fraction) and isinstance(g_num, Fraction):
                        if p_num == g_num:
                            return True
                    else:
                        try:
                            pv = float(p_num) if not isinstance(p_num, float) else p_num
                            gv = float(g_num) if not isinstance(g_num, float) else g_num
                            if math.isfinite(pv) and math.isfinite(gv) and abs(pv - gv) <= self.atol:
                                return True
                        except Exception:
                            pass
                if self._sympy_equal(pn, gn) or self._sympy_equal(p0, g0):
                    return True
        return False


OMNIScorer = OmniScorer(atol=1e-6)

# # Test
# if __name__ == "__main__":
#     pred = OMNIScorer.extract_pred
#     gold = OMNIScorer.extract_gold
#     grader = OMNIScorer.grade

#     from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
#     import torch
#     from datasets import load_dataset
#     import random

#     TRAIN_SYS_PROMPT = (
#         "You are Qwen-Math, a meticulous math tutor. "
#         "Solve the given math problem step by step. "
#         "Use the EXACT format:\n"
#         "Step 1: <reasoning>\n\n"
#         "Step 2: <reasoning>\n\n"
#         "...\n\n"
#         "Answer: \\boxed{<final answer>}"
#     )

#     # model load
#     model_name = "Qwen/Qwen2.5-Math-7B-Instruct"  # "mistralai/Mathstral-7B-v0.1"
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_compute_dtype=torch.bfloat16
#     )
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         quantization_config=bnb_config,
#         device_map="auto",
#         trust_remote_code=True,
#         torch_dtype=torch.bfloat16
#     ).eval()
#     tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
#     tokenizer.padding_side = "left"
#     if tokenizer.pad_token_id is None:
#         tokenizer.pad_token_id = tokenizer.eos_token_id
#     if getattr(model.config, "pad_token_id", None) is None:
#         model.config.pad_token_id = tokenizer.pad_token_id
 
#     def build_prompt(question_or_items: Union[str, List[Dict[str, Any]]], tokenizer, dataset_name: str) -> Union[str, List[str]]:
#         # Non-MMLU: use your training-time math prompt (single or batch)
#         def _one(q: str) -> str:
#             messages = [
#                 {"role": "system", "content": TRAIN_SYS_PROMPT},
#                 {"role": "user", "content": f"Problem: {q.strip()}\n"},
#             ]
#             return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

#         if isinstance(question_or_items, str):
#             return _one(question_or_items)
#         else:
#             return [_one(ex["problem"]) for ex in question_or_items]
        
#     # ---------------- Dataset ----------------
#     ds = load_dataset("KbsdJames/Omni-MATH", split="test")
#     ds = ds.select(range(20, 35))  # 20개만 예시

#     total = 0
#     correct = 0
#     results: List[Dict[str, Any]] = []
#     records = [ds[i] for i in range(len(ds))]

#     for start in range(0, len(records), 16):
#         chunk = records[start:start + 16]
#         prompts = build_prompt(chunk, tokenizer, dataset_name="omni")
#         if isinstance(prompts, str):
#             prompts = [prompts]
#         inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
#         inputs = {k: v.to(model.device) for k, v in inputs.items()}
#         with torch.inference_mode():
#             outputs = model.generate(
#                 **inputs,
#                 max_new_tokens=1024,
#                 temperature=0.7,
#                 top_p=0.8,
#                 pad_token_id=tokenizer.pad_token_id,
#                 eos_token_id=tokenizer.eos_token_id,
#                 )
#         # Return only the generated continuation per example
#         gens: List[str] = []
#         for i in range(len(prompts)):
#             prompt_len = inputs["input_ids"][i].shape[0]
#             gen_ids = outputs[i][prompt_len:]
#             gens.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
#         assert len(gens) == len(chunk)

#         for ex, prompt, out_text in zip(chunk, prompts, gens):
#                 p_raw = out_text
#                 p = OMNIScorer.extract_pred(p_raw)
#                 g = OMNIScorer.extract_gold(ex)
#                 ok = OMNIScorer.grade(p, g)
#                 total += 1
#                 correct += int(ok)
#                 results.append({
#                     "idx": ex.get("id", total-1),
#                     "question": ex.get("problem", ex.get("question", "")),   # ← 여기
#                     "answer": ex.get("answer", ex.get("solution", "")),
#                     "pred_raw": p_raw,
#                     "pred": p,
#                     "gold": g,
#                     "is_correct": bool(ok),
#                 })
#     acc = correct / max(1, total)
#     summary = {"total": total, "correct": correct, "accuracy": acc}

#     # Print a tiny report
#     n_show = min(15, len(results))
#     print("\n=== Omni Eval Summary ===")
#     print(f"Total: {total} | Correct: {correct} | Acc: {acc:.3f}")
#     print("\n--- Samples ---")
#     for r in results[:n_show]:
#         print("Q:", r["question"].replace("\n"," "))
#         print("PRED:", r["pred"])  # normalized
#         print("GOLD:", r["gold"])  # normalized-ish (via extract_gold)
#         print("OK:", r["is_correct"])
#         print("PRED_RAW:", r["pred_raw"])
#         print("-")
