from typing import Any, Dict, Iterable, List, Optional, Tuple
import torch
import time, math, re, random
from tqdm import tqdm
from datasets import load_dataset
import traceback
from run_profile import RunProfiler  # 네가 쓰던 프로파일러

class CPMIHardRewardBatch:
    _STEP_RE = re.compile(r"(?:^|\s)(Step\s+\d+\s*:\s*)", flags=re.IGNORECASE)
    _ANS_RE  = re.compile(r"The\s+answer\s+is\s*:\s*(.+?)\s*(?:[+\-]\s*$|\s*$)", flags=re.IGNORECASE | re.DOTALL)

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self._H_CACHE: Dict[Tuple[str, str, str], float] = {}
        self.prof = RunProfiler()

    def _clean_spaces(self, s: str) -> str:
        s = s.replace("\u200b", " ").replace("\xa0", " ")
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"\s+\n", "\n", s)
        return s.strip()
    
    def _find_suffix_start(self, full_ids: List[int], tgt_ids: List[int]) -> int:
        if not tgt_ids or len(tgt_ids) > len(full_ids):
            return -1
        Lf, Lt = len(full_ids), len(tgt_ids)
        for start in range(Lf - Lt, -1, -1):
            if full_ids[start:start+Lt] == tgt_ids:
                return start
        return -1

    def build_prompt(self, question: str, tokenizer=None) -> str:
        return f"Problem:\n{question}\nSolution (step-by-step):\n"
    
    def _format_answer_target(self, gold: str) -> str:
        gold = self._clean_spaces(str(gold))
        return f"The answer is: {gold}"
    
    def _ensure_pad_token(self):
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    # ----------------- Math-Shepherd parsing -----------------
    def parse_math_shepherd_record(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        txt = rec.get("label")
        txt = self._clean_spaces(txt)

        # 문제/풀이 분리
        m_first = re.search(r"Step\s+1\s*:", txt, flags=re.IGNORECASE)
        if m_first:
            question = txt[:m_first.start()].strip()
            tail = txt[m_first.start():].strip()
        else:
            question = re.sub(r"The\s+answer\s+is\s*:.*$", "", txt, flags=re.IGNORECASE).strip()
            tail = ""

        # 정답 파싱
        gold_answer = ""
        answer_target = ""
        ma = self._ANS_RE.search(txt)
        if ma:
            body = self._clean_spaces(ma.group(1))
            body = re.sub(r"\s*[+\-]\s*$", "", body)  # 끝의 +/- 정리
            gold_answer = body
            answer_target = self._format_answer_target(body)

        # ---------- NEW: dataset에 붙인 gold_answer가 있으면 무조건 우선 ----------
        ds_gold = rec.get("gold_answer", None)
        if isinstance(ds_gold, str):
            ds_gold = self._clean_spaces(ds_gold)
        if ds_gold:  # gold가 있으면 라벨 기반 추출을 덮어씀
            gold_answer = ds_gold
            answer_target = self._format_answer_target(ds_gold)

        steps: List[str] = []
        step_labels_pm: List[str] = []

        parts = self._STEP_RE.split(tail)
        for idx in range(1, len(parts), 2):
            step_tag = parts[idx]
            after = parts[idx+1] if idx+1 < len(parts) else ""

            # 이번 스텝 본문 경계
            mnext = self._STEP_RE.search(after)
            mans  = re.search(r"The\s+answer\s+is\s*:", after, flags=re.IGNORECASE)
            if mnext and mans:
                next_cut = min(mnext.start(), mans.start())
            elif mnext:
                next_cut = mnext.start()
            elif mans:
                next_cut = mans.start()
            else:
                next_cut = len(after)

            # 본문 후보
            step_text = (step_tag + " " + after[:next_cut]).strip()

            # 1) 스텝 본문 끝에서 우선 +/- 탐지
            pm_in_body = re.search(r"([+\-])\s*$", step_text)
            if pm_in_body:
                pm = pm_in_body.group(1)
                step_text = re.sub(r"[+\-]\s*$", "", step_text).rstrip()
            else:
                # 2) 본문 뒤 suffix의 맨 앞에서 +/- 탐지 (기존 로직)
                suffix = after[next_cut:].lstrip()
                if   suffix.startswith("+"): pm = "+"
                elif suffix.startswith("-"): pm = "-"
                else:                        pm = "+"

            steps.append(step_text)     # <-- 이미 +/− 제거된 클린 스텝
            step_labels_pm.append(pm)

        step_values = rec.get("value")
        if len(step_values) != len(steps):
            L = min(len(step_values), len(steps))
            steps = steps[:L]
            step_values = step_values[:L]
        correct_mask = [1 if pm == "+" else 0 for pm in step_values]
        
        return {
            "question": question,
            "steps": steps,                 # +/− 제거된 본문만
            "step_pm": step_labels_pm,      # 원본 라벨
            "correct_mask": correct_mask,   # +→1, −→0
            "gold_answer": gold_answer,     # ★ 여기 최종 gold (우선순위: rec.gold → 라벨 추출)
            "answer_target": answer_target, # "The answer is: <gold>"
            "task": rec.get("task", None),
            "raw": txt,
        }

    # === Answer log-probabilities (teacher-forced) ==========================
    @torch.no_grad()
    def _logprob_total(self, prompt: str, target: str) -> Tuple[float, float, int]:
        """
        Return (LP_sum, LP_mean, eff_len) where LP_sum = sum_t log p_theta(target_t | prompt + target_<t>)
        This is teacher-forced log-prob of the target suffix given the prompt.
        """
        t0 = time.perf_counter()
        full = prompt + target
        full_enc = self.tokenizer(full, return_tensors="pt", add_special_tokens=True).to(self.device)
        tgt_ids  = self.tokenizer(target, add_special_tokens=False)["input_ids"]
        input_ids = full_enc["input_ids"]
        full_ids = input_ids[0].tolist()
        L_full, L_tgt = len(full_ids), len(tgt_ids)

        if L_tgt == 0:
            return 0.0, 0.0, 0

        # Align the target suffix inside the full sequence
        Lp = self._find_suffix_start(full_ids, tgt_ids)
        if Lp < 0:
            Lp = L_full - L_tgt

        logits = self.model(**full_enc).logits.float()   # [1, L, V]
        logits_shifted = logits[:, :-1, :]               # [1, L-1, V]
        Lm1 = logits_shifted.shape[1]

        start = max(Lp - 1, 0)                           # first token predicting target[0]
        end   = min(start + L_tgt, Lm1)                  # exclusive
        eff_len = end - start
        if eff_len <= 0:
            return 0.0, 0.0, 0

        # gather log p at the gold target tokens
        lp = torch.log_softmax(logits_shifted[0, start:end, :], dim=-1)    # [eff_len, V]
        tgt_tensor = torch.tensor(tgt_ids[:eff_len], device=lp.device, dtype=torch.long)
        lp_gold = lp.gather(dim=-1, index=tgt_tensor.view(-1, 1)).squeeze(-1)  # [eff_len]
        LP_sum = float(lp_gold.sum().item())
        LP_mean = float(LP_sum / eff_len)
        try:
            self.prof.log(tag="mi:lp:call", wall_s=time.perf_counter()-t0,
                        gen_tokens=eff_len, prompt_len=L_full, target_len=L_tgt)
        except Exception:
            pass
        return LP_sum, LP_mean, eff_len

    @torch.no_grad()
    def _logprob_total_batch(self, prompts: List[str], targets: List[str], batch_size: int = 16) -> List[float]:
        """
        Batched LP_sum list for each (prompt, target).
        """
        assert len(prompts) == len(targets)
        self._ensure_pad_token()
        self.model.eval()
        if not hasattr(self, "_H_CACHE"):
            self._H_CACHE = {}

        out = [None] * len(prompts)
        miss_idx, miss_prompts, miss_targets = [], [], []

        for i, (p, t) in enumerate(zip(prompts, targets)):
            key = ("LP|sum", p, "\u241E", t)
            if key in self._H_CACHE:
                out[i] = self._H_CACHE[key]
            else:
                miss_idx.append(i); miss_prompts.append(p); miss_targets.append(t)

        if not miss_idx:
            return out

        use_amp = torch.cuda.is_available()

        for s in range(0, len(miss_idx), batch_size):
            e = min(s + batch_size, len(miss_idx))
            Ps = miss_prompts[s:e]; Ts = miss_targets[s:e]

            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
                full_enc = self.tokenizer(
                    [p + t for p, t in zip(Ps, Ts)],
                    return_tensors="pt", add_special_tokens=True, padding=True, truncation=False
                ).to(self.device)
                logits = self.model(**full_enc).logits.float()    # [B,L,V]
                logits_shifted = logits[:, :-1, :]                # [B,L-1,V]

            # tokenize targets (no specials) for suffix alignment + gather indices
            enc_t = self.tokenizer(Ts, return_tensors="pt", add_special_tokens=False, padding=True, truncation=False)

            B, Lm1, V = logits_shifted.shape
            full_ids = full_enc["input_ids"]
            ids_t = enc_t["input_ids"]

            for bi in range(B):
                tgt_ids_list = ids_t[bi].tolist()
                L_tgt = int((ids_t[bi] != self.tokenizer.pad_token_id).sum().item()) if self.tokenizer.pad_token_id is not None else len(tgt_ids_list)
                if L_tgt <= 0 or Lm1 <= 0:
                    LP_sum = 0.0
                else:
                    tgt_ids = tgt_ids_list[:L_tgt]
                    full_ids_list = full_ids[bi].tolist()
                    Lp = self._find_suffix_start(full_ids_list, tgt_ids)
                    if Lp < 0:  # fallback
                        enc_p_nospec = self.tokenizer(Ps[bi], add_special_tokens=False)
                        Lp = max(len(enc_p_nospec["input_ids"]), 1)

                    start = max(Lp - 1, 0)
                    end   = min(start + L_tgt, Lm1)
                    eff_len = end - start
                    if eff_len <= 0:
                        LP_sum = 0.0
                    else:
                        lp = torch.log_softmax(logits_shifted[bi, start:end, :], dim=-1)   # [eff_len,V]
                        tgt_tensor = torch.tensor(tgt_ids[:eff_len], device=lp.device, dtype=torch.long)
                        lp_gold = lp.gather(dim=-1, index=tgt_tensor.view(-1, 1)).squeeze(-1)
                        LP_sum = float(lp_gold.sum().item())

                gi = miss_idx[s + bi]
                key = ("LP|sum", prompts[gi], "\u241E", targets[gi])
                self._H_CACHE[key] = LP_sum
                out[gi] = LP_sum
        return out
    
    @torch.no_grad()
    def _logprob_mean_batch(self, prompts: List[str], targets: List[str], batch_size: int = 16) -> List[float]:
        """Return list of LP_mean (= per-target-token average log-prob)."""
        sums = self._logprob_total_batch(prompts, targets, batch_size=batch_size)
        # We cached eff_len in entropy path; for PMI 쪽은 길이=tokenizer(target).len (pad 제외)
        means = []
        for p, t, sm in zip(prompts, targets, sums):
            enc_t = self.tokenizer(t, add_special_tokens=False)
            eff_len = len(enc_t["input_ids"])
            means.append(sm / eff_len if eff_len > 0 else 0.0)
        return means

    def _LP_cached_sum(self, prompt: str, target: str):
        if not hasattr(self, "_H_CACHE"):
            self._H_CACHE = {}
        k = ("LP|sum", prompt, "\u241E", target)
        if k in self._H_CACHE:
            return self._H_CACHE[k]
        s, m, L = self._logprob_total(prompt, target)
        self._H_CACHE[k] = s
        self._H_CACHE[("LP|mean", prompt, "\u241E", target)] = m
        self._H_CACHE[("LP|len",  prompt, "\u241E", target)] = L
        return s

    def _LP_cached_mean(self, prompt: str, target: str):
        if not hasattr(self, "_H_CACHE"):
            self._H_CACHE = {}
        k = ("LP|mean", prompt, "\u241E", target)
        if k in self._H_CACHE:
            return self._H_CACHE[k]
        s, m, L = self._logprob_total(prompt, target)
        self._H_CACHE[k] = m
        self._H_CACHE[("LP|sum",  prompt, "\u241E", target)] = s
        self._H_CACHE[("LP|len",  prompt, "\u241E", target)] = L
        return m
    
    # ----------------- MI algorithms -----------------
    # === NEW: PMI/pointwise-CMI based on answer log-prob =========================
    def compute_step_pmi_cmi(self, question: str, steps: List[str], answer_target: str, tokenizer, normalize: bool = False) -> List[float]:
        """
        Sequential pointwise CMI (PMI-style):
        r_i = log p(A | base + prefix_<=i) - log p(A | base + prefix_<i)
        If normalize=True, use per-token mean log-prob instead of sum.
        """
        t0 = time.perf_counter()
        question = re.sub(r' +', ' ', question)
        base = self.build_prompt(question, tokenizer=tokenizer) + "\n"

        vals: List[float] = []
        prompt = base
        LP_prev = self._LP_cached_mean(prompt, answer_target) if normalize \
                else self._LP_cached_sum(prompt, answer_target)
        for s in steps:
            prompt_with = prompt + s.rstrip().rstrip("\n") + "\n"
            LP_with = self._LP_cached_mean(prompt_with, answer_target) if normalize \
                    else self._LP_cached_sum(prompt_with, answer_target)
            vals.append(LP_with - LP_prev)     # <-- PMI difference
            prompt = prompt_with
            LP_prev = LP_with
        try:
            self.prof.log(tag="pmi:cmi:call", num_prompts=len(steps)+1, n=1,
                        wall_s=time.perf_counter()-t0, num_steps=len(steps))
        except Exception:
            pass
        return vals

    def compute_step_pmi_cmi_batch(self, question: str, steps: List[str], answer_target: str, tokenizer, batch_size: int = 16, normalize: bool = False):
        question = re.sub(r' +', ' ', question)
        base = self.build_prompt(question, tokenizer=tokenizer) + "\n"
        prefixes = [base]
        prompt = base
        for s in steps:
            prompt = prompt + s.rstrip().rstrip("\n") + "\n"
            prefixes.append(prompt)
        if normalize:
            LP_prev_list = self._logprob_mean_batch(prefixes[:-1], [answer_target]*len(steps), batch_size=batch_size)
            LP_with_list = self._logprob_mean_batch(prefixes[1:],  [answer_target]*len(steps), batch_size=batch_size)
        else:
            LP_prev_list = self._logprob_total_batch(prefixes[:-1], [answer_target]*len(steps), batch_size=batch_size)
            LP_with_list = self._logprob_total_batch(prefixes[1:],  [answer_target]*len(steps), batch_size=batch_size)

        cmi = [LP_with_list[i] - LP_prev_list[i] for i in range(len(steps))]
        return cmi

    # === PMI-LOO ================================================================
    def compute_step_pmi_loo(self, question: str, steps: List[str], answer_target: str, tokenizer, normalize: bool = False) -> List[float]:
        """
        Leave-one-out contribution under PMI: φ_i = LP(all steps) - LP(all steps without i)
        (정답 로그우도 관점에서 '제거 시 성능감소'를 기여도로 봄)
        """
        question = re.sub(r' +', ' ', question)
        base = self.build_prompt(question, tokenizer=tokenizer) + "\n"
        with_all = base + "".join(s.rstrip().rstrip("\n") + "\n" for s in steps)

        LP_all = self._LP_cached_mean(with_all, answer_target) if normalize \
                else self._LP_cached_sum(with_all, answer_target)

        contribs = []
        for i in range(len(steps)):
            without_i = base + "".join(steps[j].rstrip().rstrip("\n") + "\n"
                                    for j in range(len(steps)) if j != i)
            LP_wo = self._LP_cached_mean(without_i, answer_target) if normalize \
                    else self._LP_cached_sum(without_i, answer_target)
            # with_all - without_i  (크면 i가 유익)
            contribs.append(LP_all - LP_wo)
        return contribs

    def compute_step_pmi_loo_batch(self, question: str, steps: List[str], answer_target: str, tokenizer, batch_size: int = 16, normalize: bool = False) -> List[float]:
        question = re.sub(r' +', ' ', question)
        base = self.build_prompt(question, tokenizer=tokenizer) + "\n"
        with_all = base + "".join(s.rstrip().rstrip("\n") + "\n" for s in steps)

        prompts = [with_all]
        targets = [answer_target]
        for i in range(len(steps)):
            without_i = base + "".join(steps[j].rstrip().rstrip("\n") + "\n"
                                    for j in range(len(steps)) if j != i)
            prompts.append(without_i); targets.append(answer_target)

        if normalize:
            L = self._logprob_mean_batch(prompts, targets, batch_size=batch_size)
        else:
            L = self._logprob_total_batch(prompts, targets, batch_size=batch_size)

        LP_all = L[0]
        return [LP_all - L[i+1] for i in range(len(steps))]

    # === PMI-Marginal ===========================================================
    def compute_step_pmi_marginal(self, question: str, steps: List[str], answer_target: str, tokenizer, normalize: bool = False) -> List[float]:
        """
        Marginal effect of a single step on top of base only: m_i = LP(base + S_i) - LP(base) (순서 의존, 간단 비교용)
        """
        question = re.sub(r' +', ' ', question)
        base = self.build_prompt(question, tokenizer=tokenizer) + "\n"

        LP_base = self._LP_cached_mean(base, answer_target) if normalize \
                else self._LP_cached_sum(base, answer_target)

        out = []
        for s in steps:
            with_i = base + s.rstrip().rstrip("\n") + "\n"
            LP_with = self._LP_cached_mean(with_i, answer_target) if normalize \
                    else self._LP_cached_sum(with_i, answer_target)
            out.append(LP_with - LP_base)
        return out

    def compute_step_pmi_marginal_batch(self, question: str, steps: List[str], answer_target: str, tokenizer, batch_size: int = 16, normalize: bool = False) -> List[float]:
        question = re.sub(r' +', ' ', question)
        base = self.build_prompt(question, tokenizer=tokenizer) + "\n"

        prompts = [base]
        targets = [answer_target]
        for s in steps:
            with_i = base + s.rstrip().rstrip("\n") + "\n"
            prompts.append(with_i); targets.append(answer_target)

        if normalize:
            L = self._logprob_mean_batch(prompts, targets, batch_size=batch_size)
        else:
            L = self._logprob_total_batch(prompts, targets, batch_size=batch_size)

        LP_base = L[0]
        return [L[i+1] - LP_base for i in range(len(steps))]

    # === PMI-Shapley (순열 평균) ================================================
    def compute_step_pmi_shapley(self, question: str, steps: List[str], answer_target: str, tokenizer, n_perm: int = 10, seed: int = 42, normalize: bool = False) -> List[float]:
        """
        Shapley approximation under PMI: φ_i ≈ E_π[ LP(base + prefix_before_i ∪ {i}) - LP(base + prefix_before_i) ]
        """
        question = re.sub(r' +', ' ', question)
        base = self.build_prompt(question, tokenizer=tokenizer) + "\n"
        N = len(steps); rng = random.Random(seed)
        shap = [0.0] * N

        for _ in range(n_perm):
            idxs = list(range(N)); rng.shuffle(idxs)
            prompt = base
            LP_prev = self._LP_cached_mean(prompt, answer_target) if normalize \
                    else self._LP_cached_sum(prompt, answer_target)
            for idx in idxs:
                prompt_with = prompt + steps[idx].rstrip().rstrip("\n") + "\n"
                LP_with = self._LP_cached_mean(prompt_with, answer_target) if normalize \
                        else self._LP_cached_sum(prompt_with, answer_target)
                shap[idx] += (LP_with - LP_prev)
                prompt, LP_prev = prompt_with, LP_with
        return [v / n_perm for v in shap]

    def compute_step_pmi_shapley_batch(self, question: str, steps: List[str], answer_target: str,  tokenizer, n_perm: int = 10, seed: int = 42, batch_size: int = 16, normalize: bool = False) -> List[float]:
        question = re.sub(r' +', ' ', question)
        base = self.build_prompt(question, tokenizer=tokenizer) + "\n"
        N = len(steps); rng = random.Random(seed)

        prompts_before, prompts_after = [], []
        perms = []
        for _ in range(n_perm):
            idxs = list(range(N)); rng.shuffle(idxs)
            perms.append(idxs)
            prompt = base
            for idx in idxs:
                prompts_before.append(prompt)
                prompt = prompt + steps[idx].rstrip().rstrip("\n") + "\n"
                prompts_after.append(prompt)

        targets = [answer_target] * len(prompts_before)
        if normalize:
            LP_before = self._logprob_mean_batch(prompts_before, targets, batch_size=batch_size)
            LP_after  = self._logprob_mean_batch(prompts_after,  targets, batch_size=batch_size)
        else:
            LP_before = self._logprob_total_batch(prompts_before, targets, batch_size=batch_size)
            LP_after  = self._logprob_total_batch(prompts_after,  targets, batch_size=batch_size)

        shap = [0.0] * N
        k = 0
        for idxs in perms:
            for idx in idxs:
                shap[idx] += (LP_after[k] - LP_before[k])
                k += 1
        return [v / n_perm for v in shap]
    
    # ----------------- public: labeling (stream) -----------------
    def mi_labelling(self, *, ds, n_shapley_perm: int = 10, seed: int = 42, ds_task_tag: Optional[str] = None) -> Iterable[Dict[str, Any]]:
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
                try:
                    self.prof.log(tag="skip_empty", backend="meta", dataset=task, sample_idx=si)
                except Exception:
                    pass
                continue

            # PMI versions
            # --------- CMI ---------
            try:
                if torch.cuda.is_available():
                    try: torch.cuda.reset_peak_memory_stats()
                    except Exception: pass
                t1 = time.perf_counter()
                pll_cmi = self.compute_step_pmi_cmi_batch(question, steps, answer_target, tokenizer=self.tokenizer, batch_size=16, normalize=True)
                wall = time.perf_counter() - t1
                peak = None
                if torch.cuda.is_available():
                    try: peak = torch.cuda.max_memory_allocated() / (1024**3)
                    except Exception: pass
                # prompts ≈ 2 * len(steps)  (prefix_<i>, prefix_<=i)
                self.prof.log(tag="pmi:cmi:sample", wall_s=wall, peak_mem_gb=peak, dataset=task, sample_idx=si, num_steps=len(steps), num_prompts=2 * len(steps), n=1)
            except Exception as e:
                print("[PMI-CMI batch] exception:", repr(e))
                traceback.print_exc(limit=1)
                try:
                    t1 = time.perf_counter()
                    pll_cmi = self.compute_step_pmi_cmi(question, steps, answer_target, tokenizer=self.tokenizer, normalize=True)
                    wall = time.perf_counter() - t1
                    self.prof.log(tag="pmi:cmi:sample", wall_s=wall, peak_mem_gb=None, dataset=task, sample_idx=si, num_steps=len(steps), num_prompts=2 * len(steps), n=1)
                except Exception as e2:
                    print("[PMI-CMI naive] exception:", repr(e2))
                    traceback.print_exc(limit=1)

            # --------- Marginal ---------
            try:
                if torch.cuda.is_available():
                    try: torch.cuda.reset_peak_memory_stats()
                    except Exception: pass
                t1 = time.perf_counter()
                pll_marginal = self.compute_step_pmi_marginal_batch(question, steps, answer_target, tokenizer=self.tokenizer, batch_size=16, normalize=True)
                wall = time.perf_counter() - t1
                peak = None
                if torch.cuda.is_available():
                    try: peak = torch.cuda.max_memory_allocated() / (1024**3)
                    except Exception: pass
                # prompts ≈ len(steps) + 1 (base + each with_i)
                self.prof.log(tag="pmi:marginal:sample", wall_s=wall, peak_mem_gb=peak, dataset=task, sample_idx=si, num_steps=len(steps), num_prompts=len(steps) + 1, n=1)
            except Exception as e:
                print("[PMI-Marginal batch] exception:", repr(e))
                traceback.print_exc(limit=1)
                try:
                    t1 = time.perf_counter()
                    pll_marginal = self.compute_step_pmi_marginal( question, steps, answer_target, tokenizer=self.tokenizer, normalize=True)
                    wall = time.perf_counter() - t1
                    self.prof.log(tag="pmi:marginal:sample", wall_s=wall, peak_mem_gb=None, dataset=task, sample_idx=si, num_steps=len(steps),num_prompts=len(steps) + 1, n=1)
                except Exception as e2:
                    print("[PMI-Marginal naive] exception:", repr(e2))
                    traceback.print_exc(limit=1)

            # --------- LOO ---------
            try:
                if torch.cuda.is_available():
                    try: torch.cuda.reset_peak_memory_stats()
                    except Exception: pass
                t1 = time.perf_counter()
                pll_loo = self.compute_step_pmi_loo_batch(question, steps, answer_target, tokenizer=self.tokenizer,batch_size=16, normalize=True)
                wall = time.perf_counter() - t1
                peak = None
                if torch.cuda.is_available():
                    try: peak = torch.cuda.max_memory_allocated() / (1024**3)
                    except Exception: pass
                # prompts ≈ len(steps) + 1 (with_all + each without_i)
                self.prof.log(tag="pmi:loo:sample", wall_s=wall, peak_mem_gb=peak, dataset=task, sample_idx=si, num_steps=len(steps), num_prompts=len(steps) + 1, n=1)
            except Exception as e:
                print("[PMI-LOO batch] exception:", repr(e))
                traceback.print_exc(limit=1)
                try:
                    t1 = time.perf_counter()
                    pll_loo = self.compute_step_pmi_loo(question, steps, answer_target, tokenizer=self.tokenizer, normalize=True)
                    wall = time.perf_counter() - t1
                    self.prof.log(tag="pmi:loo:sample", wall_s=wall, peak_mem_gb=None, dataset=task, sample_idx=si, num_steps=len(steps), num_prompts=len(steps) + 1, n=1)
                except Exception as e2:
                    print("[PMI-LOO naive] exception:", repr(e2))
                    traceback.print_exc(limit=1)

            # --------- Shapley ---------
            try:
                if torch.cuda.is_available():
                    try: torch.cuda.reset_peak_memory_stats()
                    except Exception: pass
                t1 = time.perf_counter()
                pll_shapley = self.compute_step_pmi_shapley_batch(question, steps, answer_target, tokenizer=self.tokenizer, n_perm=n_shapley_perm, seed=seed, batch_size=16, normalize=True)
                wall = time.perf_counter() - t1
                peak = None
                if torch.cuda.is_available():
                    try: peak = torch.cuda.max_memory_allocated() / (1024**3)
                    except Exception: pass
                # prompts ≈ n_perm * 2 * len(steps) (before/after)
                self.prof.log(tag="pmi:shapley:sample", wall_s=wall, peak_mem_gb=peak, dataset=task, sample_idx=si, num_steps=len(steps), num_prompts=n_shapley_perm * 2 * len(steps), n=1)
            except Exception as e:
                print("[PMI-Shapley batch] exception:", repr(e))
                traceback.print_exc(limit=1)
                try:
                    t1 = time.perf_counter()
                    pll_shapley = self.compute_step_pmi_shapley(question, steps, answer_target, tokenizer=self.tokenizer,n_perm=n_shapley_perm, seed=seed, normalize=True)
                    wall = time.perf_counter() - t1
                    self.prof.log(tag="pmi:shapley:sample", wall_s=wall, peak_mem_gb=None, dataset=task, sample_idx=si, num_steps=len(steps), num_prompts=n_shapley_perm * 2 * len(steps), n=1)
                except Exception as e2:
                    print("[PMI-Shapley naive] exception:", repr(e2))
                    traceback.print_exc(limit=1)

            entry = {
                "question": question,
                "completion": steps,
                "original_answer": gold_answer,
                "answer_target": answer_target,
                "pll_cmi": pll_cmi,
                "pll_loo": pll_loo,
                "pll_shapley": pll_shapley,
                "pll_marginal": pll_marginal,
                "correct_mask": correct_mask,
                "task": task,
            }
            yield entry

            try:
                self.prof.log(tag="sample_total", dataset=task, sample_idx=si, wall_s=time.perf_counter() - t0)
            except Exception:
                pass

        # final profile log
        try:
            self.prof.log(tag="dataset_total", wall_s=time.perf_counter() - t_ds0)
        except Exception:
            pass 
