from typing import Any, Dict, Iterable, List, Optional, Tuple
import torch
from fractions import Fraction
import time, math, re, random
from tqdm import tqdm
from datasets import load_dataset
from run_profile import RunProfiler  # 네가 쓰던 프로파일러
import math
import torch.nn.functional as F

def _peak_mem_gb():
    try:
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024**3)
    except Exception:
        return 0.0

# ---------------- Prompt templates (K=4 default) ----------------
PROMPT_TEMPLATES = [
    (
        "You are a careful math solver. Follow the steps methodically. Keep each step concise.\n"
        "At the end, output exactly one line in the format:\n"
        "The answer is: <final answer>\n\n"
        "Problem:\n{q}\n"
        "Solution: Let's think step by step.\n"
    ),
    (
        "Solve the problem with numbered steps. Be precise. Finally print exactly:\n"
        "The answer is: <final answer>\n"
        "If numeric, use plain digits only (no punctuation).\n\n"
        "Problem:\n{q}\n"
        "Solution (step-by-step):\n"
    ),
    (
        "Work through the solution briefly, then verify and conclude. Conclude with exactly:\n"
        "The answer is: <final answer>\n\n"
        "Problem:\n{q}\n"
        "Solution: Let's proceed carefully.\n"
    ),
    (
        "You are solving a math problem. Compute step by step. End with exactly:\n"
        "The answer is: <final answer>\n\n"
        "Problem:\n{q}\n"
        "Solution: Let's think step by step.\n"
    ),
]

# 접두/기호/형식 정리
_THEANSWER_RE  = re.compile(r"(?i)^\s*the\s+answer\s+is\s*:\s*")
_CURRENCY_RE   = re.compile(r"^[\s\$€£¥₩₹]+\s*")             # 선행 통화기호
_THOUSANDS_RE  = re.compile(r"(?<=\d),(?=\d{3}(\D|$))")      # 천단위 콤마
_FRAC_TEX_RE   = re.compile(r"\\frac\{([-+]?\d+)\}\{([-+]?\d+)\}")
_FRAC_LATEX_RE = re.compile(r"\\frac\s*{([^{}]+)}\s*{([^{}]+)}")
_PLAIN_FRAC_RE = re.compile(r"([-+]?\d+)\s*/\s*([-+]?\d+)")
_UNIT_TAIL_RE  = re.compile(
    r"\s*(?:dollars?|usd|bucks|eur|euro|pounds?|lbs?|kg|g|cm|mm|m|km|percent|per\s*cent|%|원|엔|유로)\.?\s*$",
    flags=re.I
)


class CPMISimpleReward:
    _STEP_RE = re.compile(r"(?:^|\s)(Step\s+\d+\s*:\s*)", flags=re.IGNORECASE)
    _ANS_RE  = re.compile(r"The\s+answer\s+is\s*:\s*(.+?)\s*(?:[+\-]\s*$|\s*$)", flags=re.IGNORECASE | re.DOTALL)
    _ANS_INLINE_RE = re.compile(r"The\s+answer\s+is\s*:\s*(.+?)\s*$", flags=re.IGNORECASE|re.DOTALL)

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self._H_CACHE: Dict[Tuple[str, str, str], float] = {}
        self.trunc_top_k: Optional[int] = None
        self.trunc_top_p: Optional[float] = None
        self.trunc_include_gold: bool = True
        self.neg_enabled: bool = True # False 적용 안함
        self.neg_weight: float = 1.0 # 0.0 적용 안함
        self.prof = RunProfiler()

    # -------------------------- utils --------------------------
    def _clean_spaces(self, s: str) -> str:
        if s is None:
            return ""  
        s = s.replace("\u200b", " ").replace("\xa0", " ")
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"\s+\n", "\n", s)
        return s.strip()
    
    def _strip_units_key(self, s: str) -> str:
        t = self._norm_ans_text(s)                # 네가 쓰던 기본 정규화 선적용
        t = _THEANSWER_RE.sub("", t)              # "The answer is:" 제거
        t = _CURRENCY_RE.sub("", t)               # 선행 통화기호 제거
        t = _THOUSANDS_RE.sub("", t)              # 천단위 콤마 제거
        t = t.strip()

        # 끝단 단위 단어 제거 (원하면 주석 해제 유지)
        t = _UNIT_TAIL_RE.sub("", t)

        # 1) 텍스트 어디에 끼어 있어도 LaTeX 분수 먼저 우선 추출
        m = _FRAC_TEX_RE.search(t)
        if m:
            return m.group(0)  # 원형 유지: '\frac{a}{b}'

        # 2) 일반 분수 a/b 추출
        m = _PLAIN_FRAC_RE.search(t)
        if m:
            a, b = m.group(1), m.group(2)
            return f"{a}/{b}"

        # 3) 숫자 토큰 추출 (첫 번째 것)
        m = re.search(r"[-+]?\d+(?:\.\d+)?", t)
        if m:
            return m.group(0)

        # 4) 못 찾으면 정리된 문자열 반환
        return t

    def _numeric_key(self, s: str) -> str:
        t = s.strip()
        if not t:
            return ""
        # 유니코드 마이너스 정규화
        t = t.replace("−", "-")

        # 1) LaTeX \frac{a}{b}
        m = _FRAC_TEX_RE.fullmatch(t)   # _FRAC_LATEX_RE 대신 이거 써도 됨(동일 패턴이면)
        try:
            if m:
                num = m.group(1).strip()
                den = m.group(2).strip()
                # 둘 다 정수 문자열이니까 int(...)로 캐스팅해서 Fraction(num, den) 사용
                frac = Fraction(int(num), int(den))
                return f"NUM:{frac.numerator}/{frac.denominator}"

            # 2) 간단한 a/b 형태
            #   -> '14/15', '-3/7' 같은 것만 대상으로 하고 싶으면 PLAIN_FRAC_RE 써도 좋고,
            #      최소 수정만 하려면 아래처럼 유지하면서 Fraction 호출만 고치면 됨.
            if "/" in t:
                num, den = t.split("/", 1)
                frac = Fraction(int(num.strip()), int(den.strip()))  # ✅ 여기 수정
                return f"NUM:{frac.numerator}/{frac.denominator}"

            # 3) 정수/소수 (예: 0.9333, 14, -2.5 등)
            frac = Fraction(t)
            return f"NUM:{frac.numerator}/{frac.denominator}"
        except Exception:
            # 수치 파싱 실패하면 텍스트 그대로 key 사용
            return t

    def _canonical_key(self, ans_text: Optional[str]) -> str:
        if not ans_text:
            return ""
        raw = self._extract_ans_from_text(ans_text)
        if not raw:
            # 패턴이 없으면 그냥 전체 문자열 사용
            raw = ans_text
        norm = self._norm_ans_text(raw)
        base = self._strip_units_key(norm)
        if not base:
            return ""
        return self._numeric_key(base)
    
    def _ensure_pad_token(self):
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
    
    def _find_suffix_start(self, full_ids: List[int], tgt_ids: List[int]) -> int:
        if not tgt_ids or len(tgt_ids) > len(full_ids):
            return -1
        Lf, Lt = len(full_ids), len(tgt_ids)
        for start in range(Lf - Lt, -1, -1):
            if full_ids[start:start+Lt] == tgt_ids:
                return start
        return -1

    def build_prompt(self, question: str, template_id: int = 0) -> str:
        tpl = PROMPT_TEMPLATES[template_id % len(PROMPT_TEMPLATES)]
        return tpl.format(q=question)
    
    def _norm_ans_text(self, s: str) -> str:
        s = self._clean_spaces(s)
        s = re.sub(r"[.\s]+$", "", s)
        return s

    def _wrap_ans(self, s: str) -> str:
        return f"The answer is: {self._norm_ans_text(s)}"

    def _extract_ans_from_text(self, s: str) -> Optional[str]:
        m = self._ANS_INLINE_RE.search(self._clean_spaces(s))
        return self._norm_ans_text(m.group(1)) if m else None
    
    def _format_answer_target(self, gold: str) -> str:
        gold = self._clean_spaces(str(gold))
        return f"The answer is: {gold}"
    
    def _prefixes_from_base_and_steps(self, base: str, steps: List[str]) -> List[str]:
        """[base, base+s1, base+s1+s2, ...]"""
        prefixes = [base]
        p = base
        for s in steps:
            p = p + s.rstrip().rstrip("\n") + "\n"
            prefixes.append(p)
        return prefixes

    def _target_len(self, target: str) -> int:
        return len(self.tokenizer(target, add_special_tokens=False)["input_ids"])
    
    def set_truncation(self, *, top_k: Optional[int] = None, top_p: Optional[float] = None, include_gold: bool = True, trunc_top_p_cap: Optional[int] = 3096):
        """Set default truncation used everywhere unless overridden per-call."""
        self.trunc_top_k = top_k
        self.trunc_top_p = top_p
        self.trunc_top_p_cap = trunc_top_p_cap
        self.trunc_include_gold = include_gold
    
    # ---------------------- dataset parsing --------------------
    def parse_math_shepherd_record(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        txt = self._clean_spaces(rec.get("label", ""))
        m_first = re.search(r"Step\s+1\s*:", txt, flags=re.IGNORECASE)
        if m_first:
            question = txt[:m_first.start()].strip()
            tail = txt[m_first.start():].strip()
        else:
            question = re.sub(r"The\s+answer\s+is\s*:.*$", "", txt, flags=re.IGNORECASE).strip()
            tail = ""

        gold_answer = ""
        answer_target = ""
        ma = self._ANS_RE.search(txt)
        if ma:
            body = self._clean_spaces(ma.group(1))
            body = re.sub(r"\s*[+\-]\s*$", "", body)
            gold_answer = body
            answer_target = self._format_answer_target(body)

        ds_gold = rec.get("gold_answer", None)
        if isinstance(ds_gold, str):
            ds_gold = self._clean_spaces(ds_gold)
        if ds_gold:
            gold_answer = ds_gold
            answer_target = self._format_answer_target(ds_gold)

        steps, step_labels_pm = [], []
        parts = self._STEP_RE.split(tail)
        for idx in range(1, len(parts), 2):
            step_tag = parts[idx]
            after = parts[idx+1] if idx+1 < len(parts) else ""
            mnext = self._STEP_RE.search(after); mans = re.search(r"The\s+answer\s+is\s*:", after, flags=re.IGNORECASE)
            if mnext and mans: next_cut = min(mnext.start(), mans.start())
            elif mnext:        next_cut = mnext.start()
            elif mans:         next_cut = mans.start()
            else:              next_cut = len(after)
            step_text = (step_tag + " " + after[:next_cut]).strip()
            pm_in_body = re.search(r"([+\-])\s*$", step_text)
            if pm_in_body:
                pm = pm_in_body.group(1)
                step_text = re.sub(r"[+\-]\s*$", "", step_text).rstrip()
            else:
                suffix = after[next_cut:].lstrip()
                if   suffix.startswith("+"): pm = "+"
                elif suffix.startswith("-"): pm = "-"
                else:                        pm = "+"
            steps.append(step_text); step_labels_pm.append(pm)

        step_values = rec.get("value", [])
        if len(step_values) != len(steps):
            L = min(len(step_values), len(steps))
            steps, step_values = steps[:L], step_values[:L]
        correct_mask = [1 if pm == "+" else 0 for pm in step_values]

        return {
            "question": question,
            "steps": steps,
            "step_pm": step_labels_pm,
            "correct_mask": correct_mask,
            "gold_answer": gold_answer,
            "answer_target": answer_target,
            "task": rec.get("task", None),
            "raw": txt,
        }

    # ----------------- teacher-forced log-prob -----------------
    def _ensure_tensor(self, x, device):
        import torch
        return x if torch.is_tensor(x) else torch.tensor(x, device=device)

    @torch.no_grad()
    def _select_support_indices(self, row_logits: torch.Tensor, gold_id: Optional[int], *,
        top_k: Optional[int] = None, top_p: Optional[float] = None, include_gold: bool = True) -> Optional[torch.Tensor]:
        V = row_logits.numel()
        dev = row_logits.device
        parts = []

        # (1) Top-K (그대로)
        if top_k is not None and top_k > 0:
            k = min(int(top_k), V)
            _, idx_k = torch.topk(row_logits, k=k, dim=-1, largest=True, sorted=False)
            parts.append(idx_k)
        # (2) Approx Top-P within TopK_cap
        if top_p is not None and 0.0 < float(top_p) < 1.0 and V > 0:
            # 상한: 필요시 외부에서 self.trunc_top_p_cap 설정 가능(없으면 2048)
            kcap = int(min(max(1, getattr(self, "trunc_top_p_cap", 3096)), V))
            # 정렬된 상위 kcap만 뽑아서 누적확률 계산
            vals_cap, idx_cap = torch.topk(row_logits, k=kcap, dim=-1, largest=True, sorted=True)  # [kcap]
            # 서브셋 내에서만 정규화(근사 nucleus)
            lse_cap = torch.logsumexp(vals_cap, dim=-1)                 # scalar
            probs_cap = torch.exp(vals_cap - lse_cap)                   # [kcap]
            cumsum = torch.cumsum(probs_cap, dim=-1)
            keep_mask = (cumsum <= float(top_p))
            # 경계 토큰 1개 포함
            first_over_t = torch.searchsorted(cumsum, float(top_p))
            first_over = int(first_over_t.item())
            if first_over < cumsum.numel():
                keep_mask[first_over] = True
            # 극단 케이스 방어: 최소 1개
            if not torch.any(keep_mask):
                keep_mask = torch.zeros_like(cumsum, dtype=torch.bool)
                keep_mask[0] = True
            idx_p = idx_cap[keep_mask]
            parts.append(idx_p)

        # (3) 절단 없음 → 전체 분포 사용
        if not parts:
            return None
        # (4) 합집합 + 정렬
        indices = torch.unique(torch.cat(parts), sorted=True)
        # (5) 골드 강제 포함
        if include_gold and gold_id is not None:
            gold_id_t = (gold_id if torch.is_tensor(gold_id)
                        else torch.tensor(gold_id, device=dev, dtype=indices.dtype)).long()
            if not torch.any(indices == gold_id_t):
                indices = torch.unique(torch.cat([indices, gold_id_t.view(1)]), sorted=True)
        return indices

    @torch.no_grad()
    def _logp_from_support(self, row_logits: torch.Tensor, gold_id: int, support_idx: Optional[torch.Tensor]) -> Tuple[float, int]:
        """
        support_idx가 None이면 전체 분포로 log p_gold 계산. support_idx가 주어지면 해당 서브셋으로 재정규화.
        - 서브셋에 gold가 없으면 확률 0으로 간주하여 -inf 반환(= include_gold=False일 때의 의도된 동작).
        반환: (logp_gold, support_size)
        """
        dev = row_logits.device
        gold_id_t = self._ensure_tensor(gold_id, dev).long()
        # 전체 분포
        if support_idx is None:
            lse = torch.logsumexp(row_logits, dim=-1)
            return float(row_logits[gold_id_t] - lse), int(row_logits.numel())
        # 비정상(빈 집합) 방어
        if support_idx.numel() == 0:
            return float("-inf"), 0
        # 서브셋에 gold 없음 → 확률 0
        if not torch.any(support_idx == gold_id_t):
            return float("-inf"), int(support_idx.numel())
        # 서브셋 재정규화
        sub_logits = row_logits.index_select(dim=-1, index=support_idx)  # [K]
        lse_sub = torch.logsumexp(sub_logits, dim=-1)
        logp_gold = float(row_logits[gold_id_t] - lse_sub)
        return logp_gold, int(sub_logits.numel())

    # ----------------- Single example (prompt, target) -----------------
    @torch.no_grad()
    def _logprob_total_trunc(self, prompt: str, target: str, *, top_k: Optional[int] = None, top_p: Optional[float] = None, include_gold: bool = True) -> Tuple[float, float, int, float]:
        """
        절단(Top-K/Top-p) 후 재정규화한 teacher-forced log-prob.
        반환: (LP_sum, LP_mean, eff_len, avg_support)  # avg_support = 평균 서브셋 크기
        """
        t0 = time.perf_counter()
        full = prompt + target
        full_enc = self.tokenizer(full, return_tensors="pt", add_special_tokens=True).to(self.device)
        tgt_ids  = self.tokenizer(target, add_special_tokens=False)["input_ids"]
        input_ids = full_enc["input_ids"]
        full_ids = input_ids[0].tolist()
        L_full, L_tgt = len(full_ids), len(tgt_ids)
        if L_tgt == 0:
            return 0.0, 0.0, 0, 0.0
        Lp = self._find_suffix_start(full_ids, tgt_ids)
        if Lp < 0:
            Lp = L_full - L_tgt

        logits = self.model(**full_enc).logits.float()        # [1,L,V]
        logits_shifted = logits[:, :-1, :]                   # [1,L-1,V]
        Lm1 = logits_shifted.shape[1]
        start = max(Lp - 1, 0)
        end   = min(start + L_tgt, Lm1)
        eff_len = end - start
        if eff_len <= 0:
            return 0.0, 0.0, 0, 0.0
        
        # ---------- FAST PATH: no truncation ----------
        if (top_k is None) and (top_p is None):
            rows = logits_shifted[0, start:end, :].float()              # [S, V]
            tgt  = torch.tensor(tgt_ids[:eff_len], device=rows.device, dtype=torch.long)  # [S]
            logps = F.log_softmax(rows, dim=-1)                         # [S, V]
            lp_vec = logps.gather(1, tgt.view(-1,1)).squeeze(1)         # [S]
            LP_sum = float(lp_vec.sum().item())
            LP_mean = LP_sum / eff_len
            avg_support = float(rows.shape[-1])  # = vocab size
            try:
                self.prof.log(tag="mi:lp:full", wall_s=time.perf_counter()-t0, gen_tokens=0, prompt_len=L_full, target_len=L_tgt)
            except Exception:
                pass
            return LP_sum, LP_mean, eff_len, avg_support
        # ----------------------------------------------

        LP_sum = 0.0
        support_sizes = 0
        for j in range(eff_len):
            row = logits_shifted[0, start + j, :]            # [V]
            gold = tgt_ids[j]
            support = self._select_support_indices(row, gold, top_k=top_k, top_p=top_p, include_gold=include_gold)
            lpj, supp = self._logp_from_support(row, gold, support)
            LP_sum += lpj
            support_sizes += supp

        LP_mean = LP_sum / max(eff_len, 1)
        avg_support = float(support_sizes) / max(eff_len, 1)
        try:
            self.prof.log(tag="mi:lp:trunc", wall_s=time.perf_counter()-t0, gen_tokens=0, prompt_len=L_full, target_len=L_tgt)
        except Exception:
            pass
        return LP_sum, LP_mean, eff_len, avg_support

    # ----------------- Batch version -----------------
    @torch.no_grad()
    def _logprob_total_batch_trunc(self, prompts: List[str], targets: List[str], batch_size: int = 12, *,
        top_k: Optional[int] = None, top_p: Optional[float] = None, include_gold: bool = True, return_avg_support: bool = False,) -> List[Tuple[float, int]] | Tuple[List[Tuple[float, int]], List[float]]:
        """절단 재정규화 버전의 배치 teacher-forced log-prob. 기본 반환: [(LP_sum, eff_len), ...] return_avg_support=True 이면 avg_support 리스트도 함께 반환."""
        assert len(prompts) == len(targets)
        self._ensure_pad_token()
        self.model.eval()
        if not hasattr(self, "_H_CACHE"):
            self._H_CACHE = {}

        out: List[Optional[Tuple[float, int]]] = [None] * len(prompts)
        avg_supp_list: List[Optional[float]] = [None] * len(prompts)
        miss_idx, miss_prompts, miss_targets = [], [], []

        # 캐시키에 절단 설정을 포함
        def _ckey(p, t):
            return ("LP|pair|trunc", f"k={top_k}", f"p={top_p}", f"gold={include_gold}", p, "\u241E", t)

        for i, (p, t) in enumerate(zip(prompts, targets)):
            key = _ckey(p, t)
            if key in self._H_CACHE:
                out[i] = self._H_CACHE[key]
                if return_avg_support:
                    avg_supp_list[i] = self._H_CACHE.get(key + ("|avg_support",), None)
            else:
                miss_idx.append(i); miss_prompts.append(p); miss_targets.append(t)

        if not miss_idx:
            if return_avg_support:
                return out, avg_supp_list
            return out

        use_amp = torch.cuda.is_available()
        for s in range(0, len(miss_idx), batch_size):
            e = min(s + batch_size, len(miss_idx))
            Ps = miss_prompts[s:e]; Ts = miss_targets[s:e]
            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
                full_batch = [p + t for p, t in zip(Ps, Ts)]
                full_enc = self.tokenizer(full_batch, return_tensors="pt", add_special_tokens=True, padding=True, truncation=False).to(self.device)
                logits = self.model(**full_enc).logits.float()    # [B,L,V]
                logits_shifted = logits[:, :-1, :]                # [B,L-1,V]

            # target 토크나이즈(특수토큰 제외) + 길이
            enc_t = self.tokenizer(Ts, return_tensors="pt", add_special_tokens=False, padding=True, truncation=False)
            full_ids = full_enc["input_ids"]; ids_t = enc_t["input_ids"]
            B, Lm1, V = logits_shifted.shape
            pad_id = self.tokenizer.pad_token_id

            for bi in range(B):
                tgt_ids_list = ids_t[bi].tolist()
                L_tgt = int((ids_t[bi] != pad_id).sum().item()) if pad_id is not None else len(tgt_ids_list)
                if L_tgt <= 0 or Lm1 <= 0:
                    LP_sum, eff_len, avg_support = 0.0, 0, 0.0
                else:
                    tgt_ids = tgt_ids_list[:L_tgt]
                    full_ids_list = full_ids[bi].tolist()
                    L_full = len(full_ids_list)
                    Lp = self._find_suffix_start(full_ids_list, tgt_ids)
                    if Lp < 0:
                        Lp = L_full - L_tgt

                    start = max(Lp - 1, 0)
                    end   = min(start + L_tgt, Lm1)
                    eff_len = end - start
                    if eff_len <= 0:
                        LP_sum, avg_support = 0.0, 0.0
                    else:
                        # -------- FAST PATH (no truncation) --------
                        if (top_k is None) and (top_p is None):
                            rows = logits_shifted[bi, start:end, :].float()        # [S,V]
                            tgt  = torch.tensor(tgt_ids[:eff_len], device=rows.device, dtype=torch.long)
                            logps = F.log_softmax(rows, dim=-1)
                            LP_sum = float(logps.gather(1, tgt.view(-1,1)).sum().item())
                            avg_support = float(rows.shape[-1])  # V
                        else:
                            LP_sum = 0.0
                            supp_acc = 0
                            row_logits = logits_shifted[bi, start:end, :]     # [S,V]
                            for j in range(eff_len):
                                row = row_logits[j, :]
                                gold = tgt_ids[j]
                                support = self._select_support_indices(row, gold,top_k=top_k, top_p=top_p, include_gold=include_gold)
                                lpj, supp = self._logp_from_support(row, gold, support)
                                LP_sum += lpj
                                supp_acc += supp
                            avg_support = float(supp_acc) / max(eff_len, 1)

                gi = miss_idx[s + bi]
                key = _ckey(prompts[gi], targets[gi])
                pair = (LP_sum, eff_len)
                self._H_CACHE[key] = pair
                out[gi] = pair
                if return_avg_support:
                    self._H_CACHE[key + ("|avg_support",)] = avg_support
                    avg_supp_list[gi] = avg_support

        if return_avg_support:
            return out, avg_supp_list
        return out

    # ---------------- logprob -----------------
    @torch.no_grad()
    def _logprob_mean_batch(self, prompts: List[str], targets: List[str], batch_size: int = 12, *, top_k: Optional[int] = None, top_p: Optional[float] = None, include_gold: Optional[bool] = None,) -> List[float]:
        if include_gold is None:
            include_gold = self.trunc_include_gold
        tk = self.trunc_top_k if top_k is None else top_k
        tp = self.trunc_top_p if top_p is None else top_p

        sums_lens = self._logprob_total_batch_trunc(prompts, targets, batch_size, top_k=tk, top_p=tp, include_gold=include_gold)
        means = []
        for (LP_sum, eff_len) in sums_lens:
            means.append(LP_sum / max(eff_len, 1))
        return means

    def _LP_cached_sum(self, prompt: str, target: str,
        *, top_k: Optional[int] = None, top_p: Optional[float] = None, include_gold: Optional[bool] = None):
        if not hasattr(self, "_H_CACHE"): self._H_CACHE = {}
        if include_gold is None: include_gold = self.trunc_include_gold
        tk = self.trunc_top_k if top_k is None else top_k
        tp = self.trunc_top_p if top_p is None else top_p
        k = ("LP|sum|trunc", f"k={tk}", f"p={tp}", f"gold={include_gold}", prompt, "\u241E", target)
        if k in self._H_CACHE: return self._H_CACHE[k]
        s, m, L, _ = self._logprob_total_trunc(prompt, target, top_k=tk, top_p=tp, include_gold=include_gold)
        self._H_CACHE[k] = s
        self._H_CACHE[("LP|mean|trunc", f"k={tk}", f"p={tp}", f"gold={include_gold}", prompt, "\u241E", target)] = m
        self._H_CACHE[("LP|len|trunc",  f"k={tk}", f"p={tp}", f"gold={include_gold}", prompt, "\u241E", target)] = L
        return s

    def _LP_cached_mean(self, prompt: str, target: str,
        *, top_k: Optional[int] = None, top_p: Optional[float] = None, include_gold: Optional[bool] = None):
        if not hasattr(self, "_H_CACHE"): self._H_CACHE = {}
        if include_gold is None: include_gold = self.trunc_include_gold
        tk = self.trunc_top_k if top_k is None else top_k
        tp = self.trunc_top_p if top_p is None else top_p
        k = ("LP|mean|trunc", f"k={tk}", f"p={tp}", f"gold={include_gold}", prompt, "\u241E", target)
        if k in self._H_CACHE: return self._H_CACHE[k]
        s, m, L, _ = self._logprob_total_trunc(prompt, target, top_k=tk, top_p=tp, include_gold=include_gold)
        self._H_CACHE[k] = m
        self._H_CACHE[("LP|sum|trunc",  f"k={tk}", f"p={tp}", f"gold={include_gold}", prompt, "\u241E", target)] = s
        self._H_CACHE[("LP|len|trunc",  f"k={tk}", f"p={tp}", f"gold={include_gold}", prompt, "\u241E", target)] = L
        return m

    # ---------------- negatives (heuristic + sampled) -----------
    def _heuristic_negatives(self, gold: str, max_h: int = 8) -> List[str]:
        """
        gold: 정답 텍스트 (latex/일반 숫자 상관 X)
        heuristic하게 hard-negative 숫자 후보를 생성.
        지금은 숫자/분수/√ 형태 위주로만 교란.
        """
        raw  = self._norm_ans_text(gold)      # 정규화된 원본
        base = self._strip_units_key(raw)     # 단위/텍스트 제거한 숫자 부분
        gkey = self._numeric_key(base) if base else ""   # 수치 canonical key

        cands = set()
        seen_keys = set()
        if gkey:
            seen_keys.add(gkey)

        def add(s: str):
            """
            s: 후보 표현식 (예: '\\frac{15}{14}', '15/14', '0.9', '\\sqrt{5}' 등)
            - norm + strip_units + numeric_key 를 거쳐 gold/중복 필터링
            - 저장은 항상 canonical wrapped 표현 (base) 로만.
            """
            norm = self._norm_ans_text(s)
            base = self._strip_units_key(norm)
            if not base:
                return
            key = self._numeric_key(base)
            if not key or key in seen_keys:
                return
            wrapped = self._wrap_ans(base)
            if not self._is_valid_wrapped(wrapped):
                return
            seen_keys.add(key)
            cands.add(wrapped)

        # A) LaTeX 분수: 원본으로 판단 (구조 유지)
        m_tex = _FRAC_TEX_RE.fullmatch(raw)
        if m_tex:
            a, b = int(m_tex.group(1)), int(m_tex.group(2))
            for da, db in [(-1,0),(1,0),(0,-1),(0,1),(-2,0),(0,2)]:
                nb, na = b + db, a + da
                if nb != 0:
                    add(f"\\frac{{{na}}}{{{nb}}}")
            if a != 0:
                add(f"\\frac{{{b}}}{{{a}}}")
            add(f"\\frac{{{-a}}}{{{b}}}")

        else:
            # B) 일반 분수: 키로 판단 (a/b)
            m_pf = _PLAIN_FRAC_RE.fullmatch(base or "")
            if m_pf:
                a, b = int(m_pf.group(1)), int(m_pf.group(2))
                for da, db in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nb, na = b + db, a + da
                    if nb != 0:
                        add(f"{na}/{nb}")
                if a != 0:
                    add(f"{b}/{a}")

            else:
                # C) 순수 숫자: 키로 판단
                if base and re.fullmatch(r"-?\d+(?:\.\d+)?", base):
                    x = float(base)
                    pool = [
                        x - 2, x - 10, x + 10, x + 2,
                        x * 0.9, x * 1.1,
                        math.floor(x), math.ceil(x),
                    ]
                    for v in pool:
                        if abs(v - round(v)) < 1e-9:
                            add(str(int(round(v))))
                        else:
                            add(f"{v:.6g}")
                else:
                    # D) √형태: 원본으로 판단
                    m_sq = re.fullmatch(r"\\sqrt\{(\d+)\}", raw)
                    if m_sq:
                        n = int(m_sq.group(1))
                        for dn in [-4, -1, 1, 4]:
                            m = n + dn
                            if m > 0:
                                add(f"\\sqrt{{{m}}}")
                        add(f"-\\sqrt{{{n}}}")
                    # 그 외 복합식은 교란하지 않음 (키 기반 dedup와 충돌 방지)

        outs = list(cands)
        random.shuffle(outs)
        return outs[:max_h]
    
    @torch.no_grad()
    def _sample_model_negatives(self, question: str, steps: List[str], gold: str, n: int = 8, max_new_tokens: int = 24, temperature: float = 1.05, top_p: float = 0.9, log_ctx: Optional[Dict[str, Any]] = None) -> List[str]:
        t0 = time.perf_counter()
        prompt = (self.build_prompt(question) + f"\n(Gold answer: {gold})\n\n"
        "Review the steps below. Identify any wrong step and take that step’s wrong computed result or the mistaken number used in it as a hard-negative example. Try to include all **earlier steps’ wrong intermediate outputs** if possible. "
        "If all steps seem correct, return a plausible wrong number (±1, ±2, or ±10%).\nOutput exactly one short line after 'The answer is:' no units or words.\n\n"
        + "".join(s.rstrip().rstrip('\n') + '\n\n' for s in steps) + "The answer is: ")
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = enc["input_ids"].shape[1]

        out = self.model.generate(**enc, do_sample=True, temperature=temperature, top_p=top_p, num_return_sequences=n, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.eos_token_id)

        total_new = sum(max(0, int(seq.shape[0]) - input_len) for seq in out)
        texts = self.tokenizer.batch_decode(out, skip_special_tokens=True)

        cands = set()
        g_key = self._strip_units_key(self._norm_ans_text(gold))
        for t in texts:
            tail = t[len(prompt):]
            tail = tail.split("\n")[0]
            tail = re.split(r"(?:The\s+answer\s+is\s*:)", tail, flags=re.IGNORECASE)[0]
            ans_raw = self._norm_ans_text(tail)
            ans_key = self._strip_units_key(ans_raw) 
            if ans_key and len(ans_key) <= 32 and ans_key != g_key:
                cands.add(self._wrap_ans(ans_key)) 
        
        try: self.prof.log(tag="neg:sample", wall_s=time.perf_counter() - t0, gen_tokens=total_new, num_samples=n, **(log_ctx or {}))
        except Exception: pass
        return list(cands)
    
    def _is_valid_wrapped(self, a_wrapped: str) -> bool:
        ans = self._extract_ans_from_text(a_wrapped)
        return bool(ans) and (len(ans) <= 32)

    def _build_wrong_answer_candidates(self, question: str, steps: List[str], gold_answer: str, max_candidates: int = 6, n_sample: int = 8, log_ctx: Optional[Dict[str, Any]] = None) -> List[str]:
        if not getattr(self, "neg_enabled", True) or max_candidates <= 0 or n_sample <= 0:
            return []
        
        gold_norm = self._norm_ans_text(gold_answer)
        gold_key  = self._canonical_key(gold_norm)
        pool = []
        # 1) model-sampled (logged with gen_tokens above)
        try: pool += self._sample_model_negatives(question, steps, gold_norm, n=n_sample, log_ctx=log_ctx)
        except Exception: pass
        # 2) heuristics (no generation)
        pool += self._heuristic_negatives(gold_norm, max_h=max_candidates)
        # 3) 중복/골드 제거 + 간단 유효성 필터
        uniq, seen_keys = [], set()
        if gold_key:
            seen_keys.add(gold_key)
        def _wrap_if_valid_from_any(s: str) -> Optional[str]:
            raw = self._extract_ans_from_text(s)
            norm = self._norm_ans_text(raw)
            base = self._strip_units_key(norm)
            if not base:
                return None
            key = self._numeric_key(base)
            if not key or key in seen_keys:
                return None
            wrapped = self._wrap_ans(base)  # 진짜로 저장할 표현
            if not self._is_valid_wrapped(wrapped):
                return None
            # 여기까지 왔으면 seen_keys에 등록
            seen_keys.add(key)
            return wrapped
        
        for a in pool:
            wrapped = _wrap_if_valid_from_any(a)
            if wrapped:
                # seen_keys.add(self._strip_units_key(self._extract_ans_from_text(wrapped)))
                uniq.append(wrapped)
                if len(uniq) >= max_candidates:
                    break

        if len(uniq) < max_candidates:
            more = self._heuristic_negatives(gold_norm, max_h=max_candidates*4)
            for a in more:
                wrapped = _wrap_if_valid_from_any(a)
                if wrapped:
                    # seen_keys.add(self._strip_units_key(self._extract_ans_from_text(wrapped)))
                    uniq.append(wrapped)
                    if len(uniq) >= max_candidates:
                        break
        return uniq[:max_candidates]
    
    # --------------- Contrastive + Ensemble (Batch) ---------------
    def _bases_for_templates(self, question: str, template_ids: List[int]) -> List[str]:
        return [self.build_prompt(question, template_id=t) + "\n" for t in template_ids]
    
    def compute_step_contrastive_cmi_ensemble_batch(self, question: str, steps: List[str], gold_target: str, neg_targets: List[str],
        template_ids: List[int], batch_size: int = 12, normalize: bool = True, neg_weight: Optional[float] = None,) -> Tuple[List[float], List[float], List[float], int]:
        """ Δ_i^cont (CMI): [LP(A*|<=i)-LP(A*|<i)] - mean_m [LP(Ã|<=i)-LP(Ã|<i)]"""
        w = self.neg_weight if neg_weight is None else float(neg_weight)
        use_negs = (w != 0.0) and bool(neg_targets)

        bases = self._bases_for_templates(question, template_ids)
        all_prev, all_with = [], []
        for b in bases:
            prefixes = self._prefixes_from_base_and_steps(b, steps)
            all_prev += prefixes[:-1]   # K * S
            all_with += prefixes[1:]    # K * S
        S, K = len(steps), len(template_ids)

        # truncation settings (fallback-safe)
        tk = getattr(self, "trunc_top_k", None)
        tp = getattr(self, "trunc_top_p", None)
        ig = getattr(self, "trunc_include_gold", True)

        # helper: pairs -> means
        def _to_means(pairs: List[Tuple[float, int]]) -> List[float]:
            return [s / max(L, 1) for (s, L) in pairs]

        # ----- gold -----
        if normalize:
            prev_pairs = self._logprob_total_batch_trunc(all_prev, [gold_target]*len(all_prev), batch_size, top_k=tk, top_p=tp, include_gold=ig)
            with_pairs = self._logprob_total_batch_trunc(all_with, [gold_target]*len(all_with), batch_size, top_k=tk, top_p=tp, include_gold=ig)
            LP_prev_g = _to_means(prev_pairs)
            LP_with_g = _to_means(with_pairs)
        else:
            LP_prev_g_pairs = self._logprob_total_batch_trunc(all_prev, [gold_target]*len(all_prev), batch_size, top_k=tk, top_p=tp, include_gold=ig)
            LP_with_g_pairs = self._logprob_total_batch_trunc(all_with, [gold_target]*len(all_with), batch_size, top_k=tk, top_p=tp, include_gold=ig)
            LP_prev_g = [s for (s, L) in LP_prev_g_pairs]
            LP_with_g = [s for (s, L) in LP_with_g_pairs]

        out = [0.0]*S
        cur = 0
        for _ in range(K):
            for i in range(S):
                out[i] += (LP_with_g[cur+i] - LP_prev_g[cur+i])
            cur += S
        gold_terms = [v / K for v in out]

        # ----- negatives -----
        if use_negs:
            neg_acc = [0.0]*S
            for neg in neg_targets:
                if normalize:
                    n_prev_pairs = self._logprob_total_batch_trunc(all_prev, [neg]*len(all_prev), batch_size, top_k=tk, top_p=tp, include_gold=ig)
                    n_with_pairs = self._logprob_total_batch_trunc(all_with, [neg]*len(all_with), batch_size, top_k=tk, top_p=tp, include_gold=ig)
                    n_prev = _to_means(n_prev_pairs)
                    n_with = _to_means(n_with_pairs)
                else:
                    n_prev_pairs = self._logprob_total_batch_trunc(all_prev, [neg]*len(all_prev), batch_size, top_k=tk, top_p=tp, include_gold=ig)
                    n_with_pairs = self._logprob_total_batch_trunc(all_with, [neg]*len(all_with), batch_size,top_k=tk, top_p=tp, include_gold=ig)
                    n_prev = [s for (s, L) in n_prev_pairs]
                    n_with = [s for (s, L) in n_with_pairs]
                cur = 0
                for _ in range(K):
                    for i in range(S):
                        neg_acc[i] += (n_with[cur+i] - n_prev[cur+i])
                    cur += S
            M = float(len(neg_targets))
            neg_terms = [neg_acc[i] / (K * M) for i in range(S)]
        else:
            neg_terms = [0.0] * S

        reward_terms = [gold_terms[i] - w * neg_terms[i] for i in range(S)]
        return reward_terms, gold_terms, neg_terms, 0

    def compute_step_contrastive_marginal_ensemble_batch(self, question: str, steps: List[str], gold_target: str, neg_targets: List[str],
        template_ids: List[int], batch_size: int = 12, normalize: bool = True, neg_weight: Optional[float] = None,) -> Tuple[List[float], List[float], List[float], int]:
        """ Δ_i^cont (Marginal): [LP(A*|base+S_i)-LP(A*|base)] - mean_m[LP(Ã|base+S_i)-LP(Ã|base)]"""
        w = self.neg_weight if neg_weight is None else float(neg_weight)
        use_negs = (w != 0.0) and bool(neg_targets)
        
        bases = self._bases_for_templates(question, template_ids)
        S, K = len(steps), len(template_ids)
        all_base_unique = bases
        all_with_i = []
        for b in bases:
            all_with_i += [b + s.rstrip().rstrip("\n") + "\n\n" for s in steps]  # len K*S

        tk = getattr(self, "trunc_top_k", None)
        tp = getattr(self, "trunc_top_p", None)
        ig = getattr(self, "trunc_include_gold", True)

        def _to_means(pairs: List[Tuple[float, int]]) -> List[float]:
            return [s / max(L, 1) for (s, L) in pairs]

        if normalize:
            base_pairs = self._logprob_total_batch_trunc(all_base_unique, [gold_target]*K, batch_size, top_k=tk, top_p=tp, include_gold=ig )
            with_pairs = self._logprob_total_batch_trunc(all_with_i, [gold_target]*(K*S), batch_size, top_k=tk, top_p=tp, include_gold=ig)
            LP_base_g = _to_means(base_pairs)
            LP_with_g = _to_means(with_pairs)
        else:
            base_pairs = self._logprob_total_batch_trunc(all_base_unique, [gold_target]*K, batch_size, top_k=tk, top_p=tp, include_gold=ig)
            with_pairs = self._logprob_total_batch_trunc(all_with_i, [gold_target]*(K*S), batch_size, top_k=tk, top_p=tp, include_gold=ig)
            LP_base_g = [s for (s, L) in base_pairs]
            LP_with_g = [s for (s, L) in with_pairs]

        out = [0.0]*S
        cur = 0
        for e in range(K):
            base_e = LP_base_g[e]
            for i in range(S):
                out[i] += (LP_with_g[cur+i] - base_e)
            cur += S
        gold_terms = [v / K for v in out]

        if use_negs:
            neg_acc = [0.0]*S
            for neg in neg_targets:
                if normalize:
                    n_base_pairs = self._logprob_total_batch_trunc(all_base_unique, [neg]*K, batch_size, top_k=tk, top_p=tp, include_gold=ig)
                    n_with_pairs = self._logprob_total_batch_trunc(all_with_i, [neg]*(K*S), batch_size,top_k=tk, top_p=tp, include_gold=ig)
                    n_base = _to_means(n_base_pairs)
                    n_with = _to_means(n_with_pairs)
                else:
                    n_base_pairs = self._logprob_total_batch_trunc(all_base_unique, [neg]*K, batch_size, top_k=tk, top_p=tp, include_gold=ig)
                    n_with_pairs = self._logprob_total_batch_trunc(all_with_i, [neg]*(K*S), batch_size, top_k=tk, top_p=tp, include_gold=ig)
                    n_base = [s for (s, L) in n_base_pairs]
                    n_with = [s for (s, L) in n_with_pairs]
                cur = 0
                for e in range(K):
                    base_e = n_base[e]
                    for i in range(S):
                        neg_acc[i] += (n_with[cur+i] - base_e)
                    cur += S
            M = float(len(neg_targets))
            neg_terms = [neg_acc[i] / (K * M) for i in range(S)]
        else:
            neg_terms = [0.0] * S
        
        reward_terms = [gold_terms[i] - w * neg_terms[i] for i in range(S)]
        return reward_terms, gold_terms, neg_terms, 0

    def compute_step_contrastive_loo_ensemble_batch(self, question: str, steps: List[str], gold_target: str, neg_targets: List[str],
        template_ids: List[int], batch_size: int = 12, normalize: bool = True, neg_weight: Optional[float] = None) -> Tuple[List[float], List[float], List[float], int]:
        """ Δ_i^cont (LOO): [LP(A*|all)-LP(A*|without_i)] - mean_m[LP(Ã|all)-LP(Ã|without_i)]"""
        w = self.neg_weight if neg_weight is None else float(neg_weight)
        use_negs = (w != 0.0) and bool(neg_targets)
        
        bases = self._bases_for_templates(question, template_ids)
        all_all_ctx, all_wo_ctx = [], []
        block_starts = []
        for b in bases:
            block_starts.append(len(all_wo_ctx))
            with_all = b + "".join(s.rstrip().rstrip("\n") + "\n\n" for s in steps)
            all_all_ctx.append(with_all)
            for i in range(len(steps)):
                wo = b + "".join(steps[j].rstrip().rstrip("\n") + "\n\n" for j in range(len(steps)) if j != i)
                all_wo_ctx.append(wo)
        S, K = len(steps), len(template_ids)

        tk = getattr(self, "trunc_top_k", None)
        tp = getattr(self, "trunc_top_p", None)
        ig = getattr(self, "trunc_include_gold", True)

        def _to_means(pairs: List[Tuple[float, int]]) -> List[float]:
            return [s / max(L, 1) for (s, L) in pairs]

        # ----- gold -----
        if normalize:
            all_pairs = self._logprob_total_batch_trunc(all_all_ctx, [gold_target]*len(all_all_ctx), batch_size, top_k=tk, top_p=tp, include_gold=ig)
            wo_pairs  = self._logprob_total_batch_trunc(all_wo_ctx,  [gold_target]*len(all_wo_ctx), batch_size, top_k=tk, top_p=tp, include_gold=ig)
            LP_all_g = _to_means(all_pairs)
            LP_wo_g  = _to_means(wo_pairs)
        else:
            LP_all_pairs = self._logprob_total_batch_trunc(all_all_ctx, [gold_target]*len(all_all_ctx), batch_size, top_k=tk, top_p=tp, include_gold=ig)
            LP_wo_pairs  = self._logprob_total_batch_trunc(all_wo_ctx,  [gold_target]*len(all_wo_ctx), batch_size, top_k=tk, top_p=tp, include_gold=ig)
            LP_all_g = [s for (s, L) in LP_all_pairs]
            LP_wo_g  = [s for (s, L) in LP_wo_pairs]

        out = [0.0]*S
        for e in range(K):
            o = block_starts[e]
            base_all = LP_all_g[e]
            for i in range(S):
                out[i] += (base_all - LP_wo_g[o + i])
        gold_terms = [v / K for v in out]

        # ----- negatives -----
        if use_negs:
            neg_acc = [0.0]*S
            for neg in neg_targets:
                if normalize:
                    all_pairs_n = self._logprob_total_batch_trunc(all_all_ctx, [neg]*len(all_all_ctx), batch_size, top_k=tk, top_p=tp, include_gold=ig)
                    wo_pairs_n  = self._logprob_total_batch_trunc(all_wo_ctx,  [neg]*len(all_wo_ctx),  batch_size, top_k=tk, top_p=tp, include_gold=ig)
                    LP_all_n = _to_means(all_pairs_n)
                    LP_wo_n  = _to_means(wo_pairs_n)
                else:
                    LP_all_pairs = self._logprob_total_batch_trunc(all_all_ctx, [neg]*len(all_all_ctx), batch_size, top_k=tk, top_p=tp, include_gold=ig)
                    LP_wo_pairs  = self._logprob_total_batch_trunc(all_wo_ctx,  [neg]*len(all_wo_ctx),  batch_size, top_k=tk, top_p=tp, include_gold=ig)
                    LP_all_n = [s for (s, L) in LP_all_pairs]
                    LP_wo_n  = [s for (s, L) in LP_wo_pairs]
                for e in range(K):
                    o = block_starts[e]
                    base_all = LP_all_n[e]
                    for i in range(S):
                        neg_acc[i] += (base_all - LP_wo_n[o + i])
            M = float(len(neg_targets))
            neg_terms = [neg_acc[i] / (K * M) for i in range(S)]
        else:
            neg_terms = [0.0] * S

        reward_terms = [gold_terms[i] - w * neg_terms[i] for i in range(S)]
        return reward_terms, gold_terms, neg_terms, 0

    # --------- Naive (robust) variants used as fallbacks ----------
    def _cmi_from_base_naive(self, base: str, steps: List[str], gold_target: str, neg_targets: List[str], normalize: bool = True, neg_weight: Optional[float] = None) -> Tuple[List[float], List[float], List[float]]:
        w = self.neg_weight if neg_weight is None else float(neg_weight)
        use_negs = (w != 0.0) and bool(neg_targets)

        gold_terms: List[float] = []
        neg_terms:  List[float] = []

        prompt = base
        gold_prev = self._LP_cached_mean(prompt, gold_target) if normalize else self._LP_cached_sum(prompt, gold_target)
        for s in steps:
            prompt_with = prompt + s.rstrip().rstrip("\n") + "\n\n"
            gold_with = self._LP_cached_mean(prompt_with, gold_target) if normalize else self._LP_cached_sum(prompt_with, gold_target)
            gold_delta = gold_with - gold_prev
            gold_terms.append(gold_delta)
            if use_negs:
                diffs = []
                for neg in neg_targets:
                    n_prev = self._LP_cached_mean(prompt, neg) if normalize else self._LP_cached_sum(prompt, neg)
                    n_with = self._LP_cached_mean(prompt_with, neg) if normalize else self._LP_cached_sum(prompt_with, neg)
                    diffs.append(n_with - n_prev)
                neg_terms.append(sum(diffs) / float(len(neg_targets)))
            else:
                neg_terms.append(0.0)

            prompt = prompt_with; gold_prev = gold_with
        reward_terms = [g - w * n for g, n in zip(gold_terms, neg_terms)]
        return reward_terms, gold_terms, neg_terms

    def _marginal_from_base_naive(self, base: str, steps: List[str], gold_target: str, neg_targets: List[str], normalize: bool = True, neg_weight: Optional[float] = None) -> Tuple[List[float], List[float], List[float]]:
        w = self.neg_weight if neg_weight is None else float(neg_weight)
        use_negs = (w != 0.0) and bool(neg_targets)

        gold_base = self._LP_cached_mean(base, gold_target) if normalize else self._LP_cached_sum(base, gold_target)
        gold_terms: List[float] = []
        neg_terms:  List[float] = []

        neg_base_cache: Dict[str, float] = {}
        if use_negs:
            for neg in neg_targets:
                neg_base_cache[neg] = self._LP_cached_mean(base, neg) if normalize else self._LP_cached_sum(base, neg)

        for s in steps:
            with_i = base + s.rstrip().rstrip("\n") + "\n\n"
            gold_with = self._LP_cached_mean(with_i, gold_target) if normalize else self._LP_cached_sum(with_i, gold_target)
            gold_terms.append(gold_with - gold_base)
            if use_negs:
                diffs = []
                for neg in neg_targets:
                    n_base = neg_base_cache[neg]
                    n_with = self._LP_cached_mean(with_i, neg) if normalize else self._LP_cached_sum(with_i, neg)
                    diffs.append(n_with - n_base)
                neg_terms.append(sum(diffs) / float(len(neg_targets)))
            else:
                neg_terms.append(0.0)
        reward_terms = [g - w * n for g, n in zip(gold_terms, neg_terms)]
        return reward_terms, gold_terms, neg_terms

    def _loo_from_base_naive(self, base: str, steps: List[str], gold_target: str, neg_targets: List[str], normalize: bool = True, neg_weight: Optional[float] = None) -> Tuple[List[float], List[float], List[float]]:
        w = self.neg_weight if neg_weight is None else float(neg_weight)
        use_negs = (w != 0.0) and bool(neg_targets)

        with_all = base + "".join(s.rstrip().rstrip("\n") + "\n\n" for s in steps)
        gold_all = self._LP_cached_mean(with_all, gold_target) if normalize else self._LP_cached_sum(with_all, gold_target)

        neg_all_cache: Dict[str, float] = {}
        if use_negs:
            for neg in neg_targets:
                neg_all_cache[neg] = self._LP_cached_mean(with_all, neg) if normalize else self._LP_cached_sum(with_all, neg)
        
        gold_terms: List[float] = []
        neg_terms:  List[float] = []

        for i in range(len(steps)):
            wo = base + "".join(steps[j].rstrip().rstrip("\n") + "\n\n" for j in range(len(steps)) if j != i)
            gold_wo = self._LP_cached_mean(wo, gold_target) if normalize else self._LP_cached_sum(wo, gold_target)
            gold_terms.append(gold_all - gold_wo)
            if use_negs:
                diffs = []
                for neg in neg_targets:
                    n_all = neg_all_cache[neg]
                    n_wo  = self._LP_cached_mean(wo, neg) if normalize else self._LP_cached_sum(wo, neg)
                    diffs.append(n_all - n_wo)
                neg_terms.append(sum(diffs) / float(len(neg_targets)))
            else:
                neg_terms.append(0.0)
        
        reward_terms = [g - w * n for g, n in zip(gold_terms, neg_terms)]
        return reward_terms, gold_terms, neg_terms

    def compute_step_contrastive_cmi_ensemble_naive(self, question: str, steps: List[str], gold_target: str, neg_targets: List[str], template_ids: List[int], normalize: bool = True, neg_weight: Optional[float] = None) -> Tuple[List[float], List[float], List[float]]:
        S = len(steps)
        agg_r = [0.0]*S; agg_g = [0.0]*S; agg_n = [0.0]*S
        for t in template_ids:
            base = self.build_prompt(question, template_id=t) + "\n"
            r, g, n = self._cmi_from_base_naive(base, steps, gold_target, neg_targets, normalize=normalize, neg_weight=neg_weight)
            for i in range(S):
                agg_r[i] += r[i]; agg_g[i] += g[i]; agg_n[i] += n[i]
        K = float(len(template_ids)) if template_ids else 1.0
        return [v/K for v in agg_r], [v/K for v in agg_g], [v/K for v in agg_n]

    def compute_step_contrastive_marginal_ensemble_naive(self, question: str, steps: List[str], gold_target: str, neg_targets: List[str], template_ids: List[int], normalize: bool = True, neg_weight: Optional[float] = None) -> Tuple[List[float], List[float], List[float]]:
        S = len(steps)
        agg_r = [0.0]*S; agg_g = [0.0]*S; agg_n = [0.0]*S
        for t in template_ids:
            base = self.build_prompt(question, template_id=t) + "\n"
            r, g, n = self._marginal_from_base_naive(base, steps, gold_target, neg_targets, normalize=normalize, neg_weight=neg_weight)
            for i in range(S):
                agg_r[i] += r[i]; agg_g[i] += g[i]; agg_n[i] += n[i]
        K = float(len(template_ids)) if template_ids else 1.0
        return [v/K for v in agg_r], [v/K for v in agg_g], [v/K for v in agg_n]

    def compute_step_contrastive_loo_ensemble_naive(self, question: str, steps: List[str], gold_target: str, neg_targets: List[str], template_ids: List[int], normalize: bool = True, neg_weight: Optional[float] = None) -> Tuple[List[float], List[float], List[float]]:
        S = len(steps)
        agg_r = [0.0]*S; agg_g = [0.0]*S; agg_n = [0.0]*S
        for t in template_ids:
            base = self.build_prompt(question, template_id=t) + "\n"
            r, g, n = self._loo_from_base_naive(base, steps, gold_target, neg_targets, normalize=normalize, neg_weight=neg_weight)
            for i in range(S):
                agg_r[i] += r[i]; agg_g[i] += g[i]; agg_n[i] += n[i]
        K = float(len(template_ids)) if template_ids else 1.0
        return [v/K for v in agg_r], [v/K for v in agg_g], [v/K for v in agg_n]
    
    # ----------------- public: labeling (stream) -----------------
    def mi_labelling(self, *, ds, ds_task_tag: Optional[str] = None, template_ids: Optional[List[int]] = None, batch_size: int = 12, normalize: bool = True,
        use_negatives: Optional[bool] = None, neg_weight: Optional[float] = None, n_sample_negs: int = 8, max_negatives: int = 4) -> Iterable[Dict[str, Any]]:
        
        neg_on  = self.neg_enabled if use_negatives is None else bool(use_negatives)
        w       = self.neg_weight  if neg_weight  is None else float(neg_weight)
        
        if template_ids is None:
            template_ids = [0, 1, 2, 3]
        K = len(template_ids)

        t_ds0 = time.perf_counter()
        for si, rec in tqdm(enumerate(ds)):
            t0 = time.perf_counter()
            parsed = self.parse_math_shepherd_record(rec)
            question      = parsed["question"]
            steps         = parsed["steps"]
            gold_answer   = parsed["gold_answer"]
            answer_target = self._format_answer_target(gold_answer) if gold_answer else ""
            correct_mask  = parsed["correct_mask"]
            task          = parsed.get("task") or ds_task_tag or "mathshepherd"

            if not question or not steps or not answer_target:
                try: self.prof.log(tag="skip_empty", backend="meta", dataset=task, sample_idx=si)
                except Exception: pass
                continue

            # negatives
            if neg_on and w != 0.0:
                neg_targets = self._build_wrong_answer_candidates(question, steps, gold_answer, max_candidates=max_negatives, n_sample=n_sample_negs, log_ctx={"dataset": task, "sample_idx": si})
            else:
                neg_targets = []
                try: self.prof.log(tag="neg:disabled", dataset=task, sample_idx=si, wall_s=0.0, gen_tokens=0)  # optional
                except Exception: pass

            # ---- Ensemble CMI ----
            # try:
            #     t1 = time.perf_counter()
            #     cmi_reward, cmi_gold, cmi_neg, cmi_tok = self.compute_step_contrastive_cmi_ensemble_batch(question, steps, answer_target, neg_targets, template_ids=template_ids, batch_size=batch_size, normalize=normalize, neg_weight=w)
            #     wall_cmi = time.perf_counter() - t1
            #     num_prompts_cmi = K * 2 * len(steps) * (1 + len(neg_targets))
            #     self.prof.log(tag="cont:cmi.ensemble", wall_s=wall_cmi, num_prompts=num_prompts_cmi, gen_tokens=cmi_tok, n=K, dataset=task, sample_idx=si, peak_mem_gb=_peak_mem_gb())
            # except Exception:
            #     # robust fallback
            #     t1 = time.perf_counter()
            #     cmi_reward, cmi_gold, cmi_neg = self.compute_step_contrastive_cmi_ensemble_naive(question, steps, answer_target, neg_targets, template_ids=template_ids, normalize=normalize, neg_weight=w)
            #     wall_cmi = time.perf_counter() - t1
            #     num_prompts_cmi = K * 2 * len(steps) * (1 + len(neg_targets))
            #     self.prof.log(tag="cont:cmi.ensemble.naive", wall_s=wall_cmi, num_prompts=num_prompts_cmi, gen_tokens=0, n=K, dataset=task, sample_idx=si, peak_mem_gb=_peak_mem_gb())

            # ---- Ensemble Marginal ----
            try:
                t1 = time.perf_counter()
                marg_reward, marg_gold, marg_neg, marg_tok = self.compute_step_contrastive_marginal_ensemble_batch(question, steps, answer_target, neg_targets, template_ids=template_ids, batch_size=batch_size, normalize=normalize, neg_weight=w)
                wall_marg = time.perf_counter() - t1
                num_prompts_marg = K * (len(steps) + 1) * (1 + len(neg_targets))
                self.prof.log(tag="cont:marginal.ensemble", wall_s=wall_marg, num_prompts=num_prompts_marg, gen_tokens=marg_tok, n=K, dataset=task, sample_idx=si, peak_mem_gb=_peak_mem_gb())
            except Exception:
                t1 = time.perf_counter()
                marg_reward, marg_gold, marg_neg = self.compute_step_contrastive_marginal_ensemble_naive(question, steps, answer_target, neg_targets, template_ids=template_ids, normalize=normalize, neg_weight=w)
                wall_marg = time.perf_counter() - t1
                num_prompts_marg = K * (len(steps) + 1) * (1 + len(neg_targets))
                self.prof.log(tag="cont:marginal.ensemble.naive", wall_s=wall_marg, num_prompts=num_prompts_marg, gen_tokens=0, n=K, dataset=task, sample_idx=si, peak_mem_gb=_peak_mem_gb())

            # ---- Ensemble LOO ----
            # try:
            #     t1 = time.perf_counter()
            #     loo_reward, loo_gold, loo_neg, loo_tok = self.compute_step_contrastive_loo_ensemble_batch(question, steps, answer_target, neg_targets, template_ids=template_ids, batch_size=batch_size, normalize=normalize, neg_weight=w)
            #     wall_loo = time.perf_counter() - t1
            #     num_prompts_loo = K * (len(steps) + 1) * (1 + len(neg_targets))
            #     self.prof.log(tag="cont:loo.ensemble", wall_s=wall_loo, num_prompts=num_prompts_loo, gen_tokens=loo_tok, n=K, dataset=task, sample_idx=si, peak_mem_gb=_peak_mem_gb())
            # except Exception:
            #     t1 = time.perf_counter()
            #     loo_reward, loo_gold, loo_neg = self.compute_step_contrastive_loo_ensemble_naive(question, steps, answer_target, neg_targets, template_ids=template_ids, normalize=normalize, neg_weight=w)
            #     wall_loo = time.perf_counter() - t1
            #     num_prompts_loo = K * (len(steps) + 1) * (1 + len(neg_targets))
            #     self.prof.log(tag="cont:loo.ensemble.naive", wall_s=wall_loo, num_prompts=num_prompts_loo, gen_tokens=0, n=K, dataset=task, sample_idx=si, peak_mem_gb=_peak_mem_gb())

            entry = {
                "question": question, "completion": steps,
                "original_answer": gold_answer, "answer_target": answer_target,
                # "cpmi_cmi": cmi_reward,
                # "cpmi_loo": loo_reward,
                "cpmi_marg": marg_reward,
                "correct_mask": correct_mask,
                # "cmi_gold":  cmi_gold,   "cmi_neg":  cmi_neg,   "cmi_reward":  cmi_reward,
                # "loo_gold":  loo_gold,   "loo_neg":  loo_neg,   "loo_reward":  loo_reward,
                "marg_gold": marg_gold,  "marg_neg": marg_neg,  "marg_reward": marg_reward,
                "task": task,
            }
            yield entry

            try: self.prof.log(tag="sample_total", dataset=task, sample_idx=si, wall_s=time.perf_counter() - t0)
            except Exception: pass

        try: self.prof.log(tag="dataset_total", wall_s=time.perf_counter() - t_ds0)
        except Exception: pass
