import re
import math
import ast
from typing import Any, List, Optional, Tuple, Union, Dict
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import sympy as sp
from sympy import Eq, simplify, sympify
from sympy.parsing.latex import parse_latex

class ObScorer:
    # ----------------------------- regexes -----------------------------
    _BOXED_OPEN = re.compile(r"\\boxed\\*?\s*\{")
    _DOLLAR_INLINE = re.compile(r"\$(.*?)\$")
    _ANS_LINE = re.compile(r"^\s*answer\s*:\s*(.+?)\s*$", re.I | re.M)
    _TOP_COMMA_SPLIT = re.compile(r",(?=(?:[^(){}\[\]]|[(){}\[\]])*$)")

    # LaTeX/symbol cleanups
    _SPECIAL_STRIP = [
        (re.compile(r"\\left\s*"), ""),
        (re.compile(r"\\right\s*"), ""),
        (re.compile(r"\\,|\\;|\\:"), " "),
        (re.compile(r"\\!"), ""),
        (re.compile(r"\\mathrm\{([^}]*)\}"), r"\1"),
        (re.compile(r"\\operatorname\{([^}]*)\}"), r"\1"),
        (re.compile(r"\\text\{([^}]*)\}"), r"\1"),
        # display/inline math wrappers
        (re.compile(r"\\\["), ""),
        (re.compile(r"\\\]"), ""),
        (re.compile(r"\\\("), ""),
        (re.compile(r"\\\)"), ""),
        # boxed unwrap (내용만 남김)
        (re.compile(r"\\boxed\*?\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"), r"\1"),
    ]

    def __init__(self, atol: float = 1e-8):
        self.atol = float(atol)

    # --------------------------- helpers: text --------------------------
    def _extract_all_boxed(self, text: str) -> List[str]:
        out: List[str] = []
        if not text:
            return out
        for m in self._BOXED_OPEN.finditer(text):
            i = m.end()
            depth, j = 1, i
            while j < len(text) and depth > 0:
                ch = text[j]
                if ch == "\\":
                    j += 2
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                j += 1
            if depth == 0:
                out.append(text[i:j - 1])
        return out

    def _strip_and_replace(self, s: str) -> str:
        if not s:
            return ""
        t = s.replace("\n", " ")
        t = t.replace("\\$", "$")
        for rx, rep in self._SPECIAL_STRIP:
            t = rx.sub(rep, t)
        # latex 기호/심볼 정규화
        t = t.replace("∶", ":").replace("，", ",").replace("$", "")
        t = t.replace("\\approx", "=").replace("\\simeq", "=").replace("\\sim", "=")
        t = t.replace("^{\\prime}", "'").replace("^\\prime", "'")
        t = t.replace("^{\\circ}", "").replace("^\\circ", "")
        t = re.sub(r"\^\{?\\?circ\}?", "", t)
        t = t.replace("%", "")
        t = t.replace("\\\\", "\\")
        # \infty, \pi → sympy 기호
        t = t.replace("\\infty", "oo").replace("\\pi", "pi")
        # 'x \in S' 꼴: 집합 원소 표기는 우변만 취함
        t = re.sub(r"^[^=]*?\\in\s*", "", t).strip()
        # 숫자/수식 뒤에 오는 단위 꼬리 제거(최소침습)
        # 예: '90 square units' -> '90', '\frac{68}{3} pounds' -> '\frac{68}{3}'
        t = re.sub(r"^(\s*[-+]?[\d\.\(\)/\s\*a-zA-Z\\]+(?:oo|pi)?(?:\([^)]*\))?)\s+[A-Za-z][A-Za-z\s\.-]*$", r"\1", t)
        # 안전한 꼬리 표식 제거
        t = re.sub(r"[ \t]*[.,;:]+$", "", t)
        t= re.sub(r"\s+", " ", t).strip()
        m = re.match(r"^(.*?)(\s+[A-Za-z][A-Za-z\s\.-]*)$", t)
        if m:
            t = m.group(1).strip()
        return t

    def _balance_braces(self, s: str) -> str:
        if not s:
            return ""
        open_n = s.count("{")
        close_n = s.count("}")
        while close_n > open_n and s.endswith("}"):
            s = s[:-1]
            close_n -= 1
        return s

    def _latex_cleanup(self, expr: str) -> str:
        t = self._strip_and_replace(self._balance_braces(expr))
        # frac 계열 정규화 (plain `frac{}` 포함)
        t = re.sub(r"\\t?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", t)
        t = re.sub(r"(?<!\\)frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", t)
        t = re.sub(r"\\t?frac\s*\{([^{}]*)\}\s*([A-Za-z0-9_\.]+|\([^()]*\)|\\sqrt\{[^}]*\})", r"(\1)/(\2)", t)
        # sqrt{...}, sqrt... 모두 처리 (plain `sqrt{}` 포함)
        t = re.sub(r"\\sqrt\{([^}]*)\}", r"sqrt(\1)", t)
        t = re.sub(r"\\sqrt\s*([A-Za-z0-9]+)", r"sqrt(\1)", t)  # \\sqrt6 -> sqrt(6)
        t = re.sub(r"(?<!\\)sqrt\{([^}]*)\}", r"sqrt(\1)", t)
        # 곱셈/거듭제곱 및 암시적 곱 정규화
        t = t.replace("\\cdot", "*").replace("\\times", "*")
        t = t.replace("^", "**").replace("\\pm", "±")
        t = re.sub(r"(\d)\s*([A-Za-z])", r"\1*\2", t)
        t = re.sub(r"(\d|\))\s*([A-Za-z]+)\s*\(", r"\1*\2(", t)
        t = re.sub(r"(?<![A-Za-z0-9_])pi\s*\(", "pi*(", t)
        t = re.sub(r"(?<![A-Za-z0-9_])([A-Za-z])\s*\(", r"\1*(", t)
        # 괄호 인접 곱: ')(‘ → ')*('
        t = re.sub(r"\)\s*\(", r")*(", t)
        # interval 보정: '(-oo, a)' 류에서 빠진 여는 괄호 보강(최소)
        if "," in t and t.strip().endswith((")", "]")):
            t = re.sub(r"^\s*(-?\s*oo|-?\s*[\d\.]+)\s*,", r"(\1,", t)
        return t.strip()

    def _parse_matrix_manual(self, s: str, *, _depth: int) -> Optional[sp.Matrix]:
        match = re.search(r"\\begin\{(pmatrix|bmatrix|matrix|array)\}(?:\{[^}]*\})?(.*?)\\end\{\1\}", s, flags=re.DOTALL)
        if not match:
            return None
        body = match.group(2)
        rows = [row.strip() for row in re.split(r"\\\\", body) if row.strip() and row.strip() != "\\hline"]
        if not rows:
            return None
        matrix_data = []
        for row in rows:
            cols = [c.strip() for c in row.split("&")]
            if not cols:
                return None
            parsed_row = []
            for col in cols:
                entry = self._try_parse(col, _depth=_depth + 1)
                if entry is None:
                    try:
                        entry = sympify(self._latex_cleanup(col), rational=True)
                    except Exception:
                        return None
                parsed_row.append(entry)
            matrix_data.append(parsed_row)
        try:
            return sp.Matrix(matrix_data)
        except Exception:
            return None

    def _split_top_level_commas(self, s: str) -> List[str]:
        if not s:
            return []
        if re.search(r"[\(\)\[\]\{\}]", s):
            return [s.strip()]
        return [x.strip() for x in self._TOP_COMMA_SPLIT.split(s) if x.strip()]

    def _variants_pm(self, s: str) -> List[str]:
        return [s.replace("±", "+"), s.replace("±", "-")] if "±" in s else [s]

    # ------------------------- public: extract --------------------------
    def extract_pred(self, text: str) -> str:
        if not text:
            return ""
        boxed = self._extract_all_boxed(text)
        if boxed:
            return ",".join(boxed)
        m = self._ANS_LINE.search(text)
        if m:
            return m.group(1).strip()
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        if lines:
            dollars = self._DOLLAR_INLINE.findall(lines[-1])
            if dollars:
                return ",".join(dollars)
            return lines[-1]
        return ""

    def extract_gold(self, rec: Any) -> str:
        """Extract ground truth string from OlympiadBench record. Priority: final_answer > answer > boxed in solution/solutions > $...$ > last line."""
        if isinstance(rec, dict):
            ans = rec.get("final_answer")
            if ans is None:
                ans = rec.get("answer")
            if isinstance(ans, list):
                if ans:
                    return str(ans[0]).strip()
                return ""
            if isinstance(ans, str) and ans.strip():
                txt = ans.strip()
                if txt.startswith("[") and txt.endswith("]"):
                    try:
                        parsed = ast.literal_eval(txt)
                        if isinstance(parsed, list) and parsed:
                            return str(parsed[0]).strip()
                    except Exception:
                        pass
                return txt

            sol = rec.get("solution") or rec.get("solutions") or ""
            if isinstance(sol, list) and sol:
                sol = sol[-1]
            if isinstance(sol, str) and sol:
                # 1) boxed 우선
                bx = self._extract_all_boxed(sol)
                if bx:
                    return ",".join(bx)
                # 2) $...$
                lines = [ln.strip() for ln in sol.strip().splitlines() if ln.strip()]
                if lines:
                    dollars = self._DOLLAR_INLINE.findall(lines[-1])
                    if dollars:
                        return ",".join(dollars)
                    return lines[-1]
            return ""
        return (str(rec).strip() if rec is not None else "")

    # ----------------------------- public: grade ------------------------
    def grade(self, pred: Union[bool, float, str], gold: Union[float, str]) -> bool:
        if pred is None or gold is None:
            return False
        p = str(pred)
        g = str(gold)
        # quick path: exact ignoring case/space
        if p.strip().lower() == g.strip().lower():
            return True
        # preprocess/clean
        p0 = self._strip_and_replace(p)
        g0 = self._strip_and_replace(g)
        # second quick path after cleanup
        if p0 and g0 and p0.lower() == g0.lower():
            return True
        # split multi-answers by top-level commas (order-insensitive match)
        p_list = self._split_top_level_commas(p0)
        g_list = self._split_top_level_commas(g0)
        if len(p_list) != len(g_list):
            return False
        used = [False] * len(p_list)
        for gx in g_list:
            matched_this = False
            for j, px in enumerate(p_list):
                if used[j]:
                    continue
                ok_pair = False
                for gv in self._variants_pm(gx):
                    for pv in self._variants_pm(px):
                        if gv == pv and gv != "":
                            ok_pair = True
                            break
                        if (self._is_interval(gv) and self._is_interval(pv) and self._interval_equal(gv, pv)):
                            ok_pair = True
                            break
                        if self._numeric_equal(gv, pv) or self._expression_equal(gv, pv) or self._equation_equal(gv, pv):
                            ok_pair = True
                            break
                    if ok_pair:
                        break
                if ok_pair:
                    used[j] = True
                    matched_this = True
                    break
            if not matched_this:
                return False
        return all(used)

    # ------------------------------ equality ---------------------------
    def _is_interval(self, s: str) -> bool:
        return bool(s) and s[0] in "([" and s[-1] in ")]"

    def _numeric_equal(self, a: str, b: str) -> bool:
        try:
            fa = float(a)
            fb = float(b)
        except Exception:
            return False
        for ref in (fa / 100.0, fa, fa * 100.0):
            if abs(ref - fb) <= self.atol * 1.01:
                return True
        return False

    def _try_parse(self, s: str, *, _depth: int = 0):
        if not s or _depth > 8:
            return None
        cleaned = self._latex_cleanup(s)
        has_matrix = bool(re.search(r"\\(begin\{pmatrix\}|begin\{bmatrix\}|begin\{matrix\}|begin\{array\})", s))
        if has_matrix:
            try:
                if parse_latex is not None:
                    return parse_latex(s)
            except Exception:
                pass
            manual = self._parse_matrix_manual(s, _depth=_depth)
            if manual is not None:
                return manual
        if "\\begin" not in cleaned:
            try:
                obj = sympify(cleaned, rational=True)
                if obj is not None:
                    return obj
            except Exception:
                pass
        try:
            if parse_latex is not None and re.search(r"\\[A-Za-z]+", s):
                return parse_latex(s)
        except Exception:
            pass
        try:
            return sympify(cleaned, rational=True)
        except Exception:
            return None

    def _expression_equal(self, a: str, b: str) -> bool:
        if a == b and a != "":
            return True
        A = self._try_parse(a)
        B = self._try_parse(b)
        if A is None or B is None:
            return False
        # Matrix 
        try:
            if isinstance(A, sp.MatrixBase) and isinstance(B, sp.MatrixBase):
                # 모양 같고 원소별로 같으면 True
                if A.shape != B.shape:
                    return False
                # 간단 비교
                if A.equals(B):
                    return True
                # 엄밀 비교: A-B가 영행렬?
                D = A - B
                if getattr(D, "is_zero_matrix", None):
                    return bool(D.is_zero_matrix)
                # 원소별 simplify
                return all(simplify(e) == 0 for e in D)
        except Exception:
            pass
        # direct simplify
        try:
            if simplify(A - B) == 0:
                return True
            if simplify(sp.factor(A) - sp.factor(B)) == 0:
                return True
            if simplify(sp.expand(A - B)) == 0:
                return True
        except Exception:
            pass
        # numeric case (no free symbols)
        try:
            if not (A.free_symbols or B.free_symbols):
                try:
                    av = complex(A.evalf())
                    bv = complex(B.evalf())
                    return abs(av - bv) <= self.atol * 1.01
                except Exception:
                    pass
        except Exception:
            pass
        # sample substitutions if symbols exist
        try:
            vars_ = sorted(list((A.free_symbols | B.free_symbols)), key=lambda s: s.name)
            if not vars_:
                return False
            for val in (-2, -1, 0, 1, 2, 3):
                subs = {v: val for v in vars_}
                if abs(complex(A.evalf(subs=subs)) - complex(B.evalf(subs=subs))) > 1e-8:
                    return False
            return True
        except Exception:
            return False

    def _equation_equal(self, a: str, b: str) -> bool:
        if "=" not in a or "=" not in b:
            return False
        def simp(eq: str):
            lhs, rhs = eq.split("=", 1)
            # 이 시점에서 \boxed, [], (), \mathrm 등은 제거/정규화됨
            return simplify(self._try_parse(lhs) - self._try_parse(rhs))
        try:
            A = simp(a); B = simp(b)
            d1 = simplify(A / B)
            d2 = simplify(B / A)
            return (getattr(d1, 'is_Integer', False) and d1 != 0) or (getattr(d2, 'is_Integer', False) and d2 != 0)
        except Exception:
            return False
        
    def _interval_equal(self, a: str, b: str) -> bool:
        parts_a = [x.strip() for x in a.split("\\cup")]
        parts_b = [x.strip() for x in b.split("\\cup")]
        if len(parts_a) != len(parts_b):
            return False
        for ia, ib in zip(parts_a, parts_b):
            if not (self._is_interval(ia) and self._is_interval(ib)):
                return False
            if ia[0] != ib[0] or ia[-1] != ib[-1]:
                return False
            inner_a = ia.strip('[]()')
            inner_b = ib.strip('[]()')
            ea = [x.strip() for x in inner_a.split(',')]
            eb = [x.strip() for x in inner_b.split(',')]
            if len(ea) != len(eb):
                return False
            for xa, xb in zip(ea, eb):
                if not self._expression_equal(xa, xb):
                    return False
        return True

OBScorer = ObScorer(atol=1e-6)


# Test # 
# if __name__ == "__main__":
#     scorer = ObScorer(atol=1e-6)
#     tests = [
#         ("We have $a=-\\frac 12$ and $b=\\boxed{\\frac 54}$.", "[\\boxed{\\frac{5}{4}}.\\]", True),
#         ("&=\\boxed{\\frac{\\sqrt6}3}.\n\\end{align*}", "[\n\\boxed{\\frac{\\sqrt{6}}{3}}\n\\]", True),
#         ("90 square units", "90", True),
#         ("\\frac{1}{33}", "\\dfrac{1}{33}", True),
#         ("-\\infty, -7)\\cup(-7, 3)\\cup(3, \\infty", "-\\infty, -7) \\cup (-7, 3) \\cup (3, \\infty", True),
#         ("\\frac{68}{3} pounds", "[ \\boxed{\\frac{68}{3}} \\]", True),
#         ("$r^2 + 10r+25 = \\boxed{(r+5)^2}", " the factored form of \\( r^2 + 10r + 25 \\) is \\(\\boxed{(r + 5)^2}\\).", True),
#         ("\\frac{x + 2}{7}}", "x/7 + 2/7", True),
#         (".5", "(\\boxed{\\frac{1}{2}}\\)", True),
#         ("(4,6,14,15)", "(4, 6, 14, 15)", True),
#         ("$x+2$ have opposite signs, so $-2 \\le x \\le 7$ and $\\boxed{x \\in [-2,7]}$.", """The final answer is:\n\n\\[\\boxed{[-2, 7]}\\]""", True),
#         ("the cylindrical coordinates of the point \\((1, -1, -6)\\) are:\n\n\\[ \\boxed{\\left(\\sqrt{2}, \\frac{7\\pi}{4}, -6\\right)} \\]", "so the cylindrical coordinates are $\\boxed{\\left( \\sqrt{2}, \\frac{7 \\pi}{4}, -6 \\right)}.$", True),
#         ("Since $\\cos \\frac{3 \\pi}{4} = -\\frac{1}{\\sqrt{2}},$ $\\arccos \\left( -\\frac{1}{\\sqrt{2}} \\right) = \\boxed{\\frac{3 \\pi}{4}}.", "answer is:\n\\[\n\\boxed{\\frac{3\\pi}{4}}\n\\]", True),
#         ("Thus, the matrix is $\\boxed{\\begin{pmatrix} -4/5 & -3/5 \\\\ -3/5 & 4/5 \\end{pmatrix}}.$", "is:\n\\[\n\\boxed{\\begin{pmatrix} -\\frac{4}{5} & -\\frac{3}{5} \\\\ -\\frac{3}{5} & \\frac{4}{5} \\end{pmatrix}}\n\\]", True),
#         ("\\boxed{5\\sqrt{2}}", "sqrt{50}", True),
#         ("Note that $-16x^4+x^2+2x+1=(x+1)^2-(4x^2)^2=\\boxed{(-4x^2+x+1)(4x^2+x+1)}$, where we have used the difference of squares identity for the second equality.", "final answer is:\n\n\\[\n\\boxed{(-4x^2 + x + 1)(4x^2 + x + 1)}\n\\]", True),
#     ]

#     for i, (p, g, want) in enumerate(tests, 1):
#         p = scorer.extract_pred(p)
#         rec = {"solution": g}
#         g = scorer.extract_gold(rec)
#         got = scorer.grade(p, g)
#         print(f"[{i}] {p!r} vs {g!r} -> {got} (want {want})")
