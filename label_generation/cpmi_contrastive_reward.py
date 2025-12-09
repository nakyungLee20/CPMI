from typing import Any, Dict, Iterable, List, Optional, Tuple
import torch
import time, math, re, random
from tqdm import tqdm
from datasets import load_dataset
import traceback
from run_profile import RunProfiler  # 네가 쓰던 프로파일러

def _peak_mem_gb():
    try:
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024**3)
    except Exception:
        return 0.0

class CPMIContrastiveReward:
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
        return f"Problem:\n{question}\nSolution: Let's think step by step.\n\n"
    
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
    
    # === NEW: util ─ answer text normalize/extract ============================
    _ANS_INLINE_RE = re.compile(r"The\s+answer\s+is\s*:\s*(.+?)\s*$", flags=re.IGNORECASE|re.DOTALL)

    def _norm_ans_text(self, s: str) -> str:
        s = self._clean_spaces(s)
        # 흔한 말꼬리 잘라내기
        s = re.sub(r"[.\s]+$", "", s)
        return s

    def _wrap_ans(self, s: str) -> str:
        return f"The answer is: {self._norm_ans_text(s)}"

    def _extract_ans_from_text(self, s: str) -> Optional[str]:
        m = self._ANS_INLINE_RE.search(self._clean_spaces(s))
        return self._norm_ans_text(m.group(1)) if m else None
    
    # === NEW: heuristic wrong answer candidates ===============================
    def _heuristic_negatives(self, gold: str, max_h: int = 8) -> List[str]:
        """
        숫자형 정답을 가정하지 않고 '약간 틀린' 후보를 여러 유형으로 만든다.
        - 정수/실수/분수/√표기/부호 반전/인접값/역수 등
        - 수치가 없으면 단순 토큰 교란/순서 뒤집기 등 가벼운 변형
        """
        g = self._norm_ans_text(gold)
        cands = set()

        def add(x):
            x = self._norm_ans_text(x)
            if x and x != g:
                cands.add(self._wrap_ans(x))

        # 1) 분수 a/b 형태
        mf = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", g)
        if mf:
            a, b = int(mf.group(1)), int(mf.group(2))
            # 인접 분자/분모, 약간의 스케일링
            for da, db in [(-1,0),(1,0),(0,-1),(0,1),(-2,0),(0,2)]:
                nb, na = b+db, a+da
                if nb != 0:
                    add(f"\\frac{{{na}}}{{{nb}}}")
            # 역수/부호
            if a!=0:
                add(f"\\frac{{{b}}}{{{a}}}")
            add(f"\\frac{{{-a}}}{{{b}}}")
        else:
            # 2) 정수/실수
            mn = re.fullmatch(r"-?\d+(?:\.\d+)?", g)
            if mn:
                x = float(g)
                pool = [x-2, x-1, x+1, x+2, -x, x*0.9, x*1.1, math.floor(x), math.ceil(x)]
                # 깔끔한 표현(정수면 정수로)
                for v in pool:
                    if abs(v - round(v)) < 1e-9:
                        add(str(int(round(v))))
                    else:
                        add(f"{v:.6g}")
            else:
                # 3) √, 기호치환
                msq = re.fullmatch(r"\\sqrt\{(\d+)\}", g)
                if msq:
                    n = int(msq.group(1))
                    for dn in [-4,-1,1,4]:
                        m = n+dn
                        if m>0: add(f"\\sqrt{{{m}}}")
                    add(f"-\\sqrt{{{n}}}")
                # 4) 단순 토큰 교란
                if "/" in g:
                    a,b = g.split("/",1)
                    add(b+"/"+a)
                if "+" in g:
                    add(g.replace("+","-"))
                if "-" in g and not g.startswith("-"):
                    add(g.replace("-","+"))
                add(g+"+1")
                add(g+"-1")

        # 5) 포맷 통일 및 상한
        outs = list(cands)
        random.shuffle(outs)
        return outs[:max_h]
    
    # === NEW: model-sampled negatives =========================================
    @torch.no_grad()
    def _sample_model_negatives(self, question: str, steps: List[str], gold: str, n: int = 4,
                                max_new_tokens: int = 24, temperature: float = 1.1, top_p: float = 0.9) -> List[str]:
        """
        풀 prefix로 '정답 문장'을 샘플링하여 강한 오답을 수집.
        """
        prompt = self.build_prompt(question, tokenizer=self.tokenizer) \
                 + "".join(s.rstrip().rstrip("\n") + "\n" for s in steps) \
                 + "\nThe answer is: "
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # 여러 샘플 생성
        out = self.model.generate(
            **enc,
            do_sample=True, temperature=temperature, top_p=top_p,
            num_return_sequences=n, max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        texts = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        cands = set()
        g = self._norm_ans_text(gold)
        for t in texts:
            # 마지막 생성분에서 정답 추출
            tail = t[len(prompt):]
            # 줄바꿈/문장 경계에서 잘라 짧게
            tail = tail.split("\n")[0]
            tail = re.split(r"(?:The\s+answer\s+is\s*:)", tail, flags=re.IGNORECASE)[0]
            ans = self._norm_ans_text(tail)
            if ans and ans != g and len(ans) <= 32:
                cands.add(self._wrap_ans(ans))
        return list(cands)
    
    # === NEW: public ─ make negative pool =====================================
    def _is_valid_wrapped(self, a_wrapped: str) -> bool:
        ans = self._extract_ans_from_text(a_wrapped)
        return bool(ans) and (len(ans) <= 32)

    def _build_wrong_answer_candidates(self, question: str, steps: List[str], gold_answer: str,  max_candidates: int = 8, n_sample: int = 4) -> List[str]:
        """heuristic + model-sampled를 합치고 gold와 동일/중복 제거."""
        gold_answer = self._norm_ans_text(gold_answer)
        pool = []
        # 1) 모델 샘플 (우선 사용)
        try:
            pool += self._sample_model_negatives(question, steps, gold_answer, n=n_sample)
        except Exception:
            pass
        # 2) 휴리스틱 보충
        pool += self._heuristic_negatives(gold_answer, max_h=max_candidates)
        # 3) 중복/골드 제거 + 간단 유효성 필터
        uniq, seen = [], set([self._wrap_ans(gold_answer)])
        for a in pool:
            if a not in seen and self._is_valid_wrapped(a):
                seen.add(a); uniq.append(a)
                if len(uniq) >= max_candidates:
                    break
        # 4) 안전망(부족하면 휴리스틱 더)
        if len(uniq) < max_candidates:
            more = self._heuristic_negatives(gold_answer, max_h=max_candidates*4)
            for a in more:
                if a not in seen and self._is_valid_wrapped(a):
                    seen.add(a); uniq.append(a)
                    if len(uniq) >= max_candidates:
                        break
        return uniq[:max_candidates]
    
    # === NEW: contrastive (batch) =========================================
    def compute_step_contrastive_cmi_batch(
        self, question: str, steps: List[str], gold_target: str,
        neg_targets: List[str], batch_size: int = 16, normalize: bool = True
    ) -> List[float]:
        """
        Δ_i^cont (CMI): [LP(A*|<=i)-LP(A*|<i)] - mean_m [LP(Ã|<=i)-LP(Ã|<i)]
        """
        q = re.sub(r' +', ' ', question)
        base = self.build_prompt(q, tokenizer=self.tokenizer) + "\n"

        # 정확한 페어링: prev_i = prefix_<i>, with_i = prefix_<=i
        prefixes_prev, prefixes_with = [], []
        p = base
        for s in steps:
            prev = p
            p = p + s.rstrip().rstrip("\n") + "\n"   # with
            prefixes_prev.append(prev)
            prefixes_with.append(p)

        # gold path
        if normalize:
            LP_prev = self._logprob_mean_batch(prefixes_prev, [gold_target]*len(steps), batch_size)
            LP_with = self._logprob_mean_batch(prefixes_with, [gold_target]*len(steps), batch_size)
        else:
            LP_prev = self._logprob_total_batch(prefixes_prev, [gold_target]*len(steps), batch_size)
            LP_with = self._logprob_total_batch(prefixes_with, [gold_target]*len(steps), batch_size)
        gold_diff = [w - v for w, v in zip(LP_with, LP_prev)]

        # negatives
        if not neg_targets:
            return gold_diff

        neg_diffs_sum = [0.0] * len(steps)
        for neg in neg_targets:
            if normalize:
                n_prev = self._logprob_mean_batch(prefixes_prev, [neg]*len(steps), batch_size)
                n_with = self._logprob_mean_batch(prefixes_with, [neg]*len(steps), batch_size)
            else:
                n_prev = self._logprob_total_batch(prefixes_prev, [neg]*len(steps), batch_size)
                n_with = self._logprob_total_batch(prefixes_with, [neg]*len(steps), batch_size)
            for i in range(len(steps)):
                neg_diffs_sum[i] += (n_with[i] - n_prev[i])

        M = float(len(neg_targets))
        return [gold_diff[i] - (neg_diffs_sum[i] / M) for i in range(len(steps))]
    
    def compute_step_contrastive_marginal_batch(
        self, question: str, steps: List[str], gold_target: str,
        neg_targets: List[str], batch_size: int = 16, normalize: bool = True
    ) -> List[float]:
        """
        Δ_i^cont (Marginal): [LP(A*|base+S_i)-LP(A*|base)] - mean_m[LP(Ã|base+S_i)-LP(Ã|base)]
        """
        q = re.sub(r' +', ' ', question)
        base = self.build_prompt(q, tokenizer=self.tokenizer) + "\n"
        with_i_prompts = [base + s.rstrip().rstrip("\n") + "\n" for s in steps]

        if normalize:
            LP_base_g = self._logprob_mean_batch([base]*len(steps), [gold_target]*len(steps), batch_size)
            LP_with_g = self._logprob_mean_batch(with_i_prompts,     [gold_target]*len(steps), batch_size)
        else:
            LP_base_g = self._logprob_total_batch([base]*len(steps), [gold_target]*len(steps), batch_size)
            LP_with_g = self._logprob_total_batch(with_i_prompts,     [gold_target]*len(steps), batch_size)
        gold_diff = [LP_with_g[i]-LP_base_g[i] for i in range(len(steps))]

        if not neg_targets:
            return gold_diff

        neg_sum = [0.0]*len(steps)
        for neg in neg_targets:
            if normalize:
                LP_base_n = self._logprob_mean_batch([base]*len(steps), [neg]*len(steps), batch_size)
                LP_with_n = self._logprob_mean_batch(with_i_prompts,    [neg]*len(steps), batch_size)
            else:
                LP_base_n = self._logprob_total_batch([base]*len(steps), [neg]*len(steps), batch_size)
                LP_with_n = self._logprob_total_batch(with_i_prompts,    [neg]*len(steps), batch_size)
            for i in range(len(steps)):
                neg_sum[i] += (LP_with_n[i]-LP_base_n[i])
        M = float(len(neg_targets))
        return [gold_diff[i] - (neg_sum[i]/M) for i in range(len(steps))]

    def compute_step_contrastive_loo_batch(
        self, question: str, steps: List[str], gold_target: str,
        neg_targets: List[str], batch_size: int = 16, normalize: bool = True
    ) -> List[float]:
        """
        Δ_i^cont (LOO): [LP(A*|all)-LP(A*|without_i)] - mean_m[LP(Ã|all)-LP(Ã|without_i)]
        """
        q = re.sub(r' +', ' ', question)
        base = self.build_prompt(q, tokenizer=self.tokenizer) + "\n"
        with_all = base + "".join(s.rstrip().rstrip("\n") + "\n" for s in steps)

        without_prompts = []
        for i in range(len(steps)):
            p = base + "".join(steps[j].rstrip().rstrip("\n") + "\n"
                               for j in range(len(steps)) if j != i)
            without_prompts.append(p)

        # gold
        if normalize:
            LP_all_g = self._logprob_mean_batch([with_all]*len(steps), [gold_target]*len(steps), batch_size)
            LP_wo_g  = self._logprob_mean_batch(without_prompts,       [gold_target]*len(steps), batch_size)
        else:
            LP_all_g = self._logprob_total_batch([with_all]*len(steps), [gold_target]*len(steps), batch_size)
            LP_wo_g  = self._logprob_total_batch(without_prompts,       [gold_target]*len(steps), batch_size)
        gold_diff = [LP_all_g[i]-LP_wo_g[i] for i in range(len(steps))]

        if not neg_targets:
            return gold_diff

        neg_sum = [0.0]*len(steps)
        for neg in neg_targets:
            if normalize:
                LP_all_n = self._logprob_mean_batch([with_all]*len(steps), [neg]*len(steps), batch_size)
                LP_wo_n  = self._logprob_mean_batch(without_prompts,       [neg]*len(steps), batch_size)
            else:
                LP_all_n = self._logprob_total_batch([with_all]*len(steps), [neg]*len(steps), batch_size)
                LP_wo_n  = self._logprob_total_batch(without_prompts,       [neg]*len(steps), batch_size)
            for i in range(len(steps)):
                neg_sum[i] += (LP_all_n[i]-LP_wo_n[i])
        M = float(len(neg_targets))
        return [gold_diff[i] - (neg_sum[i]/M) for i in range(len(steps))]
    
    # === NEW: contrastive (naive) =========================================
    def compute_step_contrastive_cmi(
        self, question: str, steps: List[str], gold_target: str,
        neg_targets: List[str], normalize: bool = True
    ) -> List[float]:
        q = re.sub(r' +', ' ', question)
        base = self.build_prompt(q, tokenizer=self.tokenizer) + "\n"
        vals = []
        prompt = base
        # gold prev
        gold_prev = self._LP_cached_mean(prompt, gold_target) if normalize \
                    else self._LP_cached_sum(prompt, gold_target)
        for s in steps:
            prompt_with = prompt + s.rstrip().rstrip("\n") + "\n"
            gold_with = self._LP_cached_mean(prompt_with, gold_target) if normalize \
                        else self._LP_cached_sum(prompt_with, gold_target)
            gold_diff = gold_with - gold_prev

            if neg_targets:
                acc = 0.0
                for neg in neg_targets:
                    n_prev = self._LP_cached_mean(prompt, neg) if normalize else self._LP_cached_sum(prompt, neg)
                    n_with = self._LP_cached_mean(prompt_with, neg) if normalize else self._LP_cached_sum(prompt_with, neg)
                    acc += (n_with - n_prev)
                gold_diff -= acc / float(len(neg_targets))
            vals.append(gold_diff)
            prompt = prompt_with
            gold_prev = gold_with
        return vals

    def compute_step_contrastive_marginal(
        self, question: str, steps: List[str], gold_target: str,
        neg_targets: List[str], normalize: bool = True
    ) -> List[float]:
        q = re.sub(r' +', ' ', question)
        base = self.build_prompt(q, tokenizer=self.tokenizer) + "\n"
        gold_base = self._LP_cached_mean(base, gold_target) if normalize else self._LP_cached_sum(base, gold_target)
        out = []
        for s in steps:
            with_i = base + s.rstrip().rstrip("\n") + "\n"
            gold_with = self._LP_cached_mean(with_i, gold_target) if normalize else self._LP_cached_sum(with_i, gold_target)
            diff = gold_with - gold_base
            if neg_targets:
                acc = 0.0
                for neg in neg_targets:
                    n_base = self._LP_cached_mean(base, neg) if normalize else self._LP_cached_sum(base, neg)
                    n_with = self._LP_cached_mean(with_i, neg) if normalize else self._LP_cached_sum(with_i, neg)
                    acc += (n_with - n_base)
                diff -= acc / float(len(neg_targets))
            out.append(diff)
        return out

    def compute_step_contrastive_loo(self, question: str, steps: List[str], gold_target: str, neg_targets: List[str], normalize: bool = True) -> List[float]:
        q = re.sub(r' +', ' ', question)
        base = self.build_prompt(q, tokenizer=self.tokenizer) + "\n"
        with_all = base + "".join(s.rstrip().rstrip("\n") + "\n" for s in steps)
        gold_all = self._LP_cached_mean(with_all, gold_target) if normalize else self._LP_cached_sum(with_all, gold_target)
        out = []
        for i in range(len(steps)):
            wo = base + "".join(steps[j].rstrip().rstrip("\n") + "\n" for j in range(len(steps)) if j != i)
            gold_wo = self._LP_cached_mean(wo, gold_target) if normalize else self._LP_cached_sum(wo, gold_target)
            diff = gold_all - gold_wo
            if neg_targets:
                acc = 0.0
                for neg in neg_targets:
                    n_all = self._LP_cached_mean(with_all, neg) if normalize else self._LP_cached_sum(with_all, neg)
                    n_wo  = self._LP_cached_mean(wo, neg)        if normalize else self._LP_cached_sum(wo, neg)
                    acc += (n_all - n_wo)
                diff -= acc / float(len(neg_targets))
            out.append(diff)
        return out

    # ----------------- public: labeling (stream) -----------------
    def mi_labelling(self, *, ds, n_shapley_perm: int = 10, seed: int = 42, ds_task_tag: Optional[str] = None) -> Iterable[Dict[str, Any]]:
        t_ds0 = time.perf_counter()
        for si, rec in tqdm(enumerate(ds)):
            t_sample = time.perf_counter()
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

            neg_targets = self._build_wrong_answer_candidates(question, steps, gold_answer, max_candidates=8, n_sample=5)
            M = len(neg_targets)

            # Contrastive CMI
            try:
                t0 = time.perf_counter()
                pll_cmi_cont = self.compute_step_contrastive_cmi_batch(question, steps, answer_target, neg_targets, batch_size=12, normalize=True)
                wall = time.perf_counter() - t0
                num_prompts = 2 * len(steps) * (1 + M)   # prev+with, gold+negs
                self.prof.log(tag="cont:cmi", wall_s=wall, num_prompts=num_prompts, n=1, dataset=task, sample_idx=si, peak_mem_gb=_peak_mem_gb())
            except Exception as e:
                print("[Contrastive-CMI batch] exception:", repr(e))
                traceback.print_exc(limit=1)
                try:
                    t0 = time.perf_counter()
                    pll_cmi_cont = self.compute_step_contrastive_cmi(question, steps, answer_target, neg_targets, normalize=True)
                    wall = time.perf_counter() - t0
                    num_prompts = 2 * len(steps) * (1 + M)
                    self.prof.log(tag="cont:cmi.naive", wall_s=wall, num_prompts=num_prompts, n=1, dataset=task, sample_idx=si, peak_mem_gb=_peak_mem_gb())
                except Exception as e2:
                    print("[Contrastive-CMI naive] exception:", repr(e2))
                    pll_cmi_cont = None

            # Contrastive Marginal
            try:
                t0 = time.perf_counter()
                pll_marginal_cont = self.compute_step_contrastive_marginal_batch(question, steps, answer_target, neg_targets, batch_size=12, normalize=True)
                wall = time.perf_counter() - t0
                num_prompts = 2 * len(steps) * (1 + M)   # base+with_i
                self.prof.log(tag="cont:marginal", wall_s=wall, num_prompts=num_prompts, n=1, dataset=task, sample_idx=si, peak_mem_gb=_peak_mem_gb())
            except Exception as e:
                print("[Contrastive-Marginal batch] exception:", repr(e))
                traceback.print_exc(limit=1)
                try:
                    t0 = time.perf_counter()
                    pll_marginal_cont = self.compute_step_contrastive_marginal(question, steps, answer_target, neg_targets, normalize=True)
                    wall = time.perf_counter() - t0
                    num_prompts = 2 * len(steps) * (1 + M)
                    self.prof.log(tag="cont:marginal.naive", wall_s=wall, num_prompts=num_prompts, n=1,dataset=task, sample_idx=si, peak_mem_gb=_peak_mem_gb())
                except Exception as e2:
                    print("[Contrastive-Marginal naive] exception:", repr(e2))
                    pll_marginal_cont = None

            # Contrastive LOO
            try:
                t0 = time.perf_counter()
                pll_loo_cont = self.compute_step_contrastive_loo_batch(question, steps, answer_target, neg_targets, batch_size=12, normalize=True)
                wall = time.perf_counter() - t0
                num_prompts = 2 * len(steps) * (1 + M)   # all + without_i
                self.prof.log(tag="cont:loo", wall_s=wall, num_prompts=num_prompts, n=1, dataset=task, sample_idx=si, peak_mem_gb=_peak_mem_gb())
            except Exception as e:
                print("[Contrastive-LOO batch] exception:", repr(e))
                traceback.print_exc(limit=1)
                try:
                    t0 = time.perf_counter()
                    pll_loo_cont = self.compute_step_contrastive_loo(
                        question, steps, answer_target, neg_targets, normalize=True
                    )
                    wall = time.perf_counter() - t0
                    num_prompts = 2 * len(steps) * (1 + M)
                    self.prof.log(tag="cont:loo.naive", wall_s=wall, num_prompts=num_prompts, n=1, dataset=task, sample_idx=si, peak_mem_gb=_peak_mem_gb())
                except Exception as e2:
                    print("[Contrastive-LOO naive] exception:", repr(e2))
                    pll_loo_cont = None

            entry = {
                "question": question,
                "completion": steps,
                "original_answer": gold_answer,
                "answer_target": answer_target,
                "pll_cmi_cont": pll_cmi_cont,
                "pll_loo_cont": pll_loo_cont,
                "pll_marginal_cont": pll_marginal_cont,
                "correct_mask": correct_mask,
                "task": task,
            }
            yield entry

            try:
                self.prof.log(tag="sample_total", dataset=task, sample_idx=si, wall_s=time.perf_counter() - t_sample)
            except Exception:
                pass

        # final profile log
        try:
            self.prof.log(tag="dataset_total", wall_s=time.perf_counter() - t_ds0, dataset=task)
        except Exception:
            pass 
