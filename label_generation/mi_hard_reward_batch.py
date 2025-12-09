from typing import Any, Dict, Iterable, List, Optional, Tuple
import torch
import time, math, re, random
from tqdm import tqdm
from datasets import load_dataset
import traceback
from run_profile import RunProfiler  # 네가 쓰던 프로파일러

class MIHardRewardBatch:
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

    # ----------------- entropy / MI (SUM version) -----------------
    @torch.no_grad()
    def _entropy_bits_total(self, prompt: str, target: str) -> Tuple[float, float, int]:
        """Return (H_total_bits, H_bits_per_token, target_eff_len) for H(A | prompt)."""
        t0 = time.perf_counter()
        full = prompt + target
        full_enc = self.tokenizer(full, return_tensors="pt", add_special_tokens=True).to(self.device)
        tgt_ids  = self.tokenizer(target, add_special_tokens=False)["input_ids"]
        input_ids = full_enc["input_ids"]
        full_ids = input_ids[0].tolist()
        L_full, L_tgt = len(full_ids), len(tgt_ids)

        if L_tgt == 0:
            wall = time.perf_counter() - t0
            try:
                self.prof.log(tag="mi:entropy:call", wall_s=wall, gen_tokens=0,
                            prompt_len=L_full if 'L_full' in locals() else 0, target_len=0)
            except Exception:
                pass
            return 0.0, 0.0, 0

        Lp = self._find_suffix_start(full_ids, tgt_ids)
        if Lp < 0:
            Lp = L_full - L_tgt

        logits = self.model(**full_enc).logits.float()    # [1, L, V]
        logits_shifted = logits[:, :-1, :]                # [1, L-1, V]
        Lm1 = logits_shifted.shape[1]

        start = max(Lp - 1, 0)
        end   = min(start + L_tgt, Lm1)
        eff_len = end - start
        if eff_len <= 0:
            wall = time.perf_counter() - t0
            try:
                self.prof.log(tag="mi:entropy:call", wall_s=wall, gen_tokens=0,
                            prompt_len=L_full, target_len=L_tgt, num_prompts=1, n=1)
            except Exception:
                pass
            return 0.0, 0.0, 0

        LOG2E = 1.0 / math.log(2.0)
        lp = torch.log_softmax(logits_shifted[0, start:end, :], dim=-1)  # [eff_len, V]
        H_nats_per_step = -(lp.exp() * lp).sum(dim=-1)                   # [eff_len]
        H_bits_sum  = float(H_nats_per_step.sum().item() * LOG2E)
        H_bits_mean = float(H_bits_sum / eff_len)

        wall = time.perf_counter() - t0
        try:
            self.prof.log(tag="mi:entropy:call", wall_s=wall, gen_tokens=eff_len,
                        prompt_len=L_full, target_len=L_tgt, num_prompts=1, n=1)
        except Exception:
            pass
        return H_bits_sum, H_bits_mean, eff_len

    @torch.no_grad()
    def _entropy_bits_total_batch(self, prompts: List[str], targets: List[str], batch_size: int = 16) -> List[float]:
        assert len(prompts) == len(targets)
        self._ensure_pad_token()
        self.model.eval()
        if not hasattr(self, "_H_CACHE"):
            self._H_CACHE = {}

        out = [None] * len(prompts)
        miss_idx, miss_prompts, miss_targets = [], [], []

        for i, (p, t) in enumerate(zip(prompts, targets)):
            key = ("H|sum", p, "\u241E", t)
            if key in self._H_CACHE:
                out[i] = self._H_CACHE[key]
            else:
                miss_idx.append(i); miss_prompts.append(p); miss_targets.append(t)

        if not miss_idx:
            return out

        LOG2E = 1.0 / math.log(2.0)
        use_amp = torch.cuda.is_available()

        for s in range(0, len(miss_idx), batch_size):
            e = min(s + batch_size, len(miss_idx))
            Ps = miss_prompts[s:e]
            Ts = miss_targets[s:e]

            # (1) full: prompt+target with specials → logits 기준이 되는 시퀀스
            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
                full_enc = self.tokenizer(
                    [p + t for p, t in zip(Ps, Ts)],
                    return_tensors="pt", add_special_tokens=True, padding=True, truncation=False
                ).to(self.device)
                logits = self.model(**full_enc).logits.float()   # [B,L,V]
                logits_shifted = logits[:, :-1, :]               # [B,L-1,V]

            # (2) 타깃: specials 없이 (suffix 매칭에 사용)
            enc_t = self.tokenizer(
                Ts, return_tensors="pt", add_special_tokens=False, padding=True, truncation=False
            )  # CPU 텐서면 충분

            B, Lm1, V = logits_shifted.shape
            full_ids = full_enc["input_ids"]  # on device; copy to cpu list when needed
            ids_t = enc_t["input_ids"]

            for bi in range(B):
                tgt_ids_list = ids_t[bi].tolist()
                # 타깃 유효 길이
                L_tgt = int((ids_t[bi] != self.tokenizer.pad_token_id).sum().item()) if self.tokenizer.pad_token_id is not None else len(tgt_ids_list)
                if L_tgt <= 0 or Lm1 <= 0:
                    H_sum = 0.0
                else:
                    tgt_ids = tgt_ids_list[:L_tgt]

                    # --- 핵심: suffix alignment로 타깃 시작 위치 찾기 ---
                    full_ids_list = full_ids[bi].tolist()
                    Lp = self._find_suffix_start(full_ids_list, tgt_ids)

                    # 안전장치: 못 찾으면 (예외적) 휴리스틱 fallback
                    if Lp < 0:
                        # prompt만 토크나이즈(스페셜 제거)해서 길이 휴리스틱 사용
                        enc_p_nospec = self.tokenizer(Ps[bi], add_special_tokens=False)
                        Lp_heur = len(enc_p_nospec["input_ids"])
                        Lp = max(Lp_heur, 1)

                    start = max(Lp - 1, 0)
                    end   = min(start + L_tgt, Lm1)
                    eff_len = end - start

                    if eff_len <= 0:
                        H_sum = 0.0
                    else:
                        lp = torch.log_softmax(logits_shifted[bi, start:end, :], dim=-1)    # [eff_len, V]
                        H_nats_per_step = -(lp.exp() * lp).sum(dim=-1)                      # [eff_len]
                        H_sum = float(H_nats_per_step.sum().item() * LOG2E)

                gi = miss_idx[s + bi]
                key = ("H|sum", prompts[gi], "\u241E", targets[gi])
                self._H_CACHE[key] = H_sum
                out[gi] = H_sum

        return out
    
    @torch.no_grad()
    def _entropy_bits_mean_batch(self, prompts: List[str], targets: List[str], batch_size: int = 16) -> List[float]:
        """bits-per-token(=mean) 리스트를 반환."""
        assert len(prompts) == len(targets)
        self._ensure_pad_token()
        self.model.eval()
        if not hasattr(self, "_H_CACHE"):
            self._H_CACHE = {}

        out = [None] * len(prompts)
        miss_idx, miss_prompts, miss_targets = [], [], []

        for i, (p, t) in enumerate(zip(prompts, targets)):
            kmean = ("H|mean", p, "\u241E", t)
            if kmean in self._H_CACHE:
                out[i] = self._H_CACHE[kmean]
            else:
                miss_idx.append(i); miss_prompts.append(p); miss_targets.append(t)

        if not miss_idx:
            return out

        LOG2E = 1.0 / math.log(2.0)
        use_amp = torch.cuda.is_available()

        for s in range(0, len(miss_idx), batch_size):
            e = min(s + batch_size, len(miss_idx))
            Ps = miss_prompts[s:e]
            Ts = miss_targets[s:e]

            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
                full_enc = self.tokenizer(
                    [p + t for p, t in zip(Ps, Ts)],
                    return_tensors="pt", add_special_tokens=True, padding=True, truncation=False
                ).to(self.device)
                logits = self.model(**full_enc).logits.float()   # [B,L,V]
                logits_shifted = logits[:, :-1, :]

            enc_t = self.tokenizer(Ts, return_tensors="pt", add_special_tokens=False, padding=True, truncation=False)

            B, Lm1, V = logits_shifted.shape
            full_ids = full_enc["input_ids"]
            ids_t = enc_t["input_ids"]

            for bi in range(B):
                tgt_ids_list = ids_t[bi].tolist()
                L_tgt = int((ids_t[bi] != self.tokenizer.pad_token_id).sum().item()) if self.tokenizer.pad_token_id is not None else len(tgt_ids_list)
                if L_tgt <= 0 or Lm1 <= 0:
                    H_sum = 0.0
                    eff_len = 0
                else:
                    tgt_ids = tgt_ids_list[:L_tgt]
                    full_ids_list = full_ids[bi].tolist()
                    Lp = self._find_suffix_start(full_ids_list, tgt_ids)
                    if Lp < 0:
                        enc_p_nospec = self.tokenizer(Ps[bi], add_special_tokens=False)
                        Lp = max(len(enc_p_nospec["input_ids"]), 1)
                    start = max(Lp - 1, 0)
                    end   = min(start + L_tgt, Lm1)
                    eff_len = end - start

                    if eff_len <= 0:
                        H_sum = 0.0
                    else:
                        lp = torch.log_softmax(logits_shifted[bi, start:end, :], dim=-1)
                        H_nats_per_step = -(lp.exp() * lp).sum(dim=-1)
                        H_sum = float(H_nats_per_step.sum().item() * LOG2E)

                H_mean = (H_sum / eff_len) if eff_len > 0 else 0.0

                gi = miss_idx[s + bi]
                key_sum  = ("H|sum",  prompts[gi], "\u241E", targets[gi])
                key_mean = ("H|mean", prompts[gi], "\u241E", targets[gi])
                key_len  = ("H|len",  prompts[gi], "\u241E", targets[gi])
                self._H_CACHE[key_sum]  = H_sum
                self._H_CACHE[key_mean] = H_mean
                self._H_CACHE[key_len]  = eff_len
                out[gi] = H_mean

        return out
    
    def _H_cached(self, prompt: str, target: str) -> float:
        """SUM version: cache and return H_bits_sum (total bits)."""
        if not hasattr(self, "_H_CACHE"):
            self._H_CACHE = {}
        ksum = ("H|sum", prompt, "\u241E", target)
        if ksum in self._H_CACHE:
            return self._H_CACHE[ksum]
        H_sum, H_mean, eff_len = self._entropy_bits_total(prompt, target)
        self._H_CACHE[ksum] = H_sum
        # (optional) also cache mean and length for other callers:
        self._H_CACHE[("H|mean", prompt, "\u241E", target)] = H_mean
        self._H_CACHE[("H|len",  prompt, "\u241E", target)] = eff_len
        return H_sum
    
    def _H_cached_mean(self, prompt: str, target: str) -> float:
        """Return H_bits_mean (bits per token)."""
        if not hasattr(self, "_H_CACHE"):
            self._H_CACHE = {}
        kmean = ("H|mean", prompt, "\u241E", target)
        if kmean in self._H_CACHE:
            return self._H_CACHE[kmean]
        H_sum, H_mean, eff_len = self._entropy_bits_total(prompt, target)
        self._H_CACHE[kmean] = H_mean
        self._H_CACHE[("H|sum",  prompt, "\u241E", target)] = H_sum
        self._H_CACHE[("H|len",  prompt, "\u241E", target)] = eff_len
        return H_mean

    def _H_cached_sum(self, prompt: str, target: str) -> float:
        return self._H_cached(prompt, target)

    # ----------------- MI algorithms -----------------
    def compute_step_mi_loo(self, question: str, steps: List[str], answer_target: str, tokenizer):
        """Leave-one-out contribution: Δ_i = H(all without i) - H(all). Returns a List[float] with each step's Δ_i."""
        t0 = time.perf_counter()
        question = re.sub(r' +', ' ', question)
        base = self.build_prompt(question, tokenizer=tokenizer) + "\n"
        with_all = base + "".join(s.rstrip().rstrip("\n") + "\n" for s in steps)
        H_all = self._H_cached_mean(with_all, answer_target)

        contribs = []
        for i in range(len(steps)):
            without_i = base + "".join(steps[j].rstrip().rstrip("\n") + "\n" for j in range(len(steps)) if j != i)
            H_wo = self._H_cached_mean(without_i, answer_target)
            contribs.append(H_wo - H_all)
        
        try: # profile
            self.prof.log(tag="mi:loo:call", num_prompts=len(steps) + 1, n=1, wall_s=time.perf_counter() - t0, num_steps=len(steps))
        except Exception:
            pass
        return contribs

    def compute_step_mi_loo_batch(self, question: str, steps: List[str], answer_target: str, tokenizer, batch_size: int = 16):
        # base
        question = re.sub(r' +', ' ', question)
        base = self.build_prompt(question, tokenizer=tokenizer) + "\n"
        with_all = base + "".join(s.rstrip().rstrip("\n") + "\n" for s in steps)

        # 1) 한 번에 필요한 프롬프트/타깃 수집
        prompts = [with_all]  # H_all 먼저
        targets = [answer_target]
        for i in range(len(steps)):
            without_i = base + "".join(steps[j].rstrip().rstrip("\n") + "\n" for j in range(len(steps)) if j != i)
            prompts.append(without_i)
            targets.append(answer_target)

        # 2) 배치로 H들 계산
        Hs = self._entropy_bits_mean_batch(prompts, targets, batch_size=batch_size)
        H_all = Hs[0]
        contribs = [Hs[i+1] - H_all for i in range(len(steps))]
        return contribs
    
    def compute_step_mi_marginal(self, question: str, steps: List[str], answer_target: str, tokenizer):
        """marginal effect of step: MI_i = H(base) - H(base + S_i) order dependent ↓: List[float]"""
        t0 = time.perf_counter()
        question = re.sub(r' +', ' ', question)
        base = self.build_prompt(question, tokenizer=tokenizer) + "\n"
        H_base = self._H_cached_mean(base, answer_target)

        mis = []
        for s in steps:
            with_i = base + s.rstrip().rstrip("\n") + "\n"
            H_with = self._H_cached_mean(with_i, answer_target)
            mis.append(H_base - H_with)
        try:
            self.prof.log(tag="mi:marginal:call", num_prompts=len(steps) + 1, n=1, wall_s=time.perf_counter() - t0, num_steps=len(steps))
        except Exception:
            pass
        return mis

    def compute_step_mi_marginal_batch(self, question: str, steps: List[str], answer_target: str, tokenizer, batch_size: int = 16):
        question = re.sub(r' +', ' ', question)
        base = self.build_prompt(question, tokenizer=tokenizer) + "\n"
        prompts = [base]  # H_base 먼저
        targets = [answer_target]
        for s in steps:
            with_i = base + s.rstrip().rstrip("\n") + "\n"
            prompts.append(with_i)
            targets.append(answer_target)

        Hs = self._entropy_bits_mean_batch(prompts, targets, batch_size=batch_size)
        H_base = Hs[0]
        mis = [H_base - Hs[i+1] for i in range(len(steps))]
        return mis

    def compute_step_mi_shapley(self, question: str, steps: List[str], answer_target: str, tokenizer, n_perm: int = 10, seed: int = 42):
        """Shapley approximation: various random permutation π's average Δ_MI  φ_i ≈ E_π[ H(base + prefix_before_i) - H(base + prefix_before_i + S_i) ]"""
        t0 = time.perf_counter()
        question = re.sub(r' +', ' ', question)
        base = self.build_prompt(question, tokenizer=tokenizer) + "\n"

        N = len(steps)
        rng = random.Random(seed)
        shap = [0.0] * N
        for _ in range(n_perm):
            idxs = list(range(N))
            rng.shuffle(idxs)
            prompt = base
            H_prev = self._H_cached_mean(prompt, answer_target)
            for idx in idxs:
                prompt_with = prompt + steps[idx].rstrip().rstrip("\n") + "\n"
                H_with = self._H_cached_mean(prompt_with, answer_target)
                shap[idx] += (H_prev - H_with)
                prompt, H_prev = prompt_with, H_with
        shap = [v / n_perm for v in shap]
        try:
            self.prof.log(tag="mi:shapley:call", num_prompts=n_perm * (N + 1), n=1, wall_s=time.perf_counter() - t0, num_steps=N, n_perm=n_perm)
        except Exception:
            pass
        return shap

    def compute_step_mi_shapley_batch(self, question: str, steps: List[str], answer_target: str, tokenizer, n_perm: int = 10, seed: int = 42, batch_size: int = 16):
        question = re.sub(r' +', ' ', question)
        base = self.build_prompt(question, tokenizer=tokenizer) + "\n"
        N = len(steps)
        rng = random.Random(seed)
        prompts_before, prompts_after = [], []
        perms = []
        for _ in range(n_perm):
            idxs = list(range(N))
            rng.shuffle(idxs)
            perms.append(idxs)
            prompt = base
            for idx in idxs:
                prompts_before.append(prompt)
                prompt = prompt + steps[idx].rstrip().rstrip("\n") + "\n"
                prompts_after.append(prompt)

        targets = [answer_target] * len(prompts_before)
        H_before = self._entropy_bits_mean_batch(prompts_before, targets, batch_size=batch_size)
        H_after  = self._entropy_bits_mean_batch(prompts_after,  targets, batch_size=batch_size)

        shap = [0.0] * N
        k = 0
        for idxs in perms:
            for idx in idxs:
                shap[idx] += (H_before[k] - H_after[k])
                k += 1
        shap = [v / n_perm for v in shap]
        return shap

    def compute_step_mi_cmi(self, question: str, steps: List[str], answer_target: str, tokenizer) -> List[float]:
        """Sequential conditional MI (no permutations): For each step i in the given order, compute CMI_i = H(base + prefix_<i>) - H(base + prefix_≤i) where H(·) is the total bits of H(A | prompt), and prefix_<i> is the concatenation of steps up to but not including i."""
        t0 = time.perf_counter()
        question = re.sub(r' +', ' ', question)
        base = self.build_prompt(question, tokenizer=tokenizer) + "\n"

        vals: List[float] = []
        prompt = base
        H_prev = self._H_cached_mean(prompt, answer_target)
        for s in steps:
            prompt_with = prompt + s.rstrip().rstrip("\n") + "\n"
            H_with = self._H_cached_mean(prompt_with, answer_target)
            vals.append(H_prev - H_with)
            prompt = prompt_with
            H_prev = H_with
        try:
            self.prof.log(tag="mi:cmi:call", num_prompts=len(steps) + 1, n=1, wall_s=time.perf_counter() - t0, num_steps=len(steps))
        except Exception:
            pass
        return vals
    
    def compute_step_mi_cmi_batch(self, question: str, steps: List[str], answer_target: str, tokenizer, batch_size: int = 16):
        question = re.sub(r' +', ' ', question)
        base = self.build_prompt(question, tokenizer=tokenizer) + "\n"
        prefixes = [base]  # prefix_0
        prompt = base
        for s in steps:
            prompt = prompt + s.rstrip().rstrip("\n") + "\n"
            prefixes.append(prompt)  # prefix_≤i

        # H(prefix_<i>)는 prefixes[:-1], H(prefix_≤i)는 prefixes[1:]
        H_prev_list = self._entropy_bits_mean_batch(prefixes[:-1], [answer_target]*len(steps), batch_size=batch_size)
        H_with_list = self._entropy_bits_mean_batch(prefixes[1:],  [answer_target]*len(steps), batch_size=batch_size)
        cmi = [H_prev_list[i] - H_with_list[i] for i in range(len(steps))]
        return cmi

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

            # LOO
            try:
                if torch.cuda.is_available():
                    try: torch.cuda.reset_peak_memory_stats()
                    except Exception: pass
                t1 = time.perf_counter()
                mi_loo = self.compute_step_mi_loo_batch(question, steps, answer_target, tokenizer=self.tokenizer)
                wall = time.perf_counter() - t1
                peak = None
                if torch.cuda.is_available():
                    try: peak = torch.cuda.max_memory_allocated() / (1024**3)
                    except Exception: pass
                self.prof.log(tag="mi:loo:sample",  wall_s=wall, peak_mem_gb=peak, dataset=task, sample_idx=si, num_steps=len(steps), num_prompts=len(steps) + 1, n=1)
            except Exception as e:
                print("[batch-LOO] exception:", repr(e))
                traceback.print_exc(limit=1)
                print("Fail batch process. Convert to naive.")
                mi_loo = self.compute_step_mi_loo(question, steps, answer_target, tokenizer=self.tokenizer)

            # Shapley
            try:
                if torch.cuda.is_available():
                    try: torch.cuda.reset_peak_memory_stats()
                    except Exception: pass
                t1 = time.perf_counter()
                mi_shapley = self.compute_step_mi_shapley_batch(question, steps, answer_target, tokenizer=self.tokenizer, n_perm=n_shapley_perm, seed=seed)
                wall = time.perf_counter() - t1
                peak = None
                if torch.cuda.is_available():
                    try: peak = torch.cuda.max_memory_allocated() / (1024**3)
                    except Exception: pass
                self.prof.log(tag="mi:shapley:sample", wall_s=wall, peak_mem_gb=peak, dataset=task, sample_idx=si, num_steps=len(steps), num_prompts=n_shapley_perm * 2 * len(steps), n=1)
            except Exception as e:
                print("[batch-Shapley] exception:", repr(e))
                traceback.print_exc(limit=1)
                print("Fail batch process. Convert to naive.")
                mi_shapley = self.compute_step_mi_shapley(question, steps, answer_target, tokenizer=self.tokenizer, n_perm=n_shapley_perm, seed=seed)

            # CMI
            try:
                if torch.cuda.is_available():
                    try: torch.cuda.reset_peak_memory_stats()
                    except Exception: pass
                t1 = time.perf_counter()
                mi_cmi = self.compute_step_mi_cmi_batch(question, steps, answer_target, tokenizer=self.tokenizer)
                wall = time.perf_counter() - t1
                peak = None
                if torch.cuda.is_available():
                    try: peak = torch.cuda.max_memory_allocated() / (1024**3)
                    except Exception: pass
                self.prof.log(tag="mi:cmi:sample", wall_s=wall, peak_mem_gb=peak, dataset=task, sample_idx=si, num_steps=len(steps), num_prompts=2 * len(steps), n=1)
            except Exception as e:
                print("[batch-CMI] exception:", repr(e))
                traceback.print_exc(limit=1)
                print("Fail batch process. Convert to naive.")
                mi_cmi = self.compute_step_mi_cmi(question, steps, answer_target, tokenizer=self.tokenizer)

            # Marginal
            try:
                if torch.cuda.is_available():
                    try: torch.cuda.reset_peak_memory_stats()
                    except Exception: pass
                t1 = time.perf_counter()
                mi_margin = self.compute_step_mi_marginal_batch(question, steps, answer_target, tokenizer=self.tokenizer)
                wall = time.perf_counter() - t1
                peak = None
                if torch.cuda.is_available():
                    try: peak = torch.cuda.max_memory_allocated() / (1024**3)
                    except Exception: pass
                self.prof.log(tag="mi:margin:sample", wall_s=wall, peak_mem_gb=peak, dataset=task, sample_idx=si, num_steps=len(steps), num_prompts=len(steps) + 1, n=1)
            except Exception as e:
                print("[batch-Marginal] exception:", repr(e))
                traceback.print_exc(limit=1)
                print("Fail batch process. Convert to naive.")
                mi_margin = self.compute_step_mi_marginal(question, steps, answer_target, tokenizer=self.tokenizer)

            entry = {
                "question": question,
                "completion": steps,            # 원본 step들
                "original_answer": gold_answer,     # 본문만(참고용)
                "answer_target": answer_target, # "The answer is: …" (MI 타깃)
                "mi_loo": mi_loo,
                "mi_shapley": mi_shapley,
                "mi_cmi": mi_cmi,
                "mi_margin": mi_margin,
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
    
