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

class CPMIEnsembleReward:
    """
    Unifies:
      - prompt-ensemble PMI/CMI/LOO/Marginal machinery (Version 1),
      - contrastive (gold vs negatives) CPMI (Version 2),
    with shared parsing + teacher-forced log-prob routines.
    """
    _STEP_RE = re.compile(r"(?:^|\s)(Step\s+\d+\s*:\s*)", flags=re.IGNORECASE)
    _ANS_RE  = re.compile(r"The\s+answer\s+is\s*:\s*(.+?)\s*(?:[+\-]\s*$|\s*$)", flags=re.IGNORECASE | re.DOTALL)
    _ANS_INLINE_RE = re.compile(r"The\s+answer\s+is\s*:\s*(.+?)\s*$", flags=re.IGNORECASE|re.DOTALL)

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self._H_CACHE: Dict[Tuple[str, str, str], float] = {}
        self.prof = RunProfiler()

    # -------------------------- utils --------------------------
    def _clean_spaces(self, s: str) -> str:
        s = s.replace("\u200b", " ").replace("\xa0", " ")
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"\s+\n", "\n", s)
        return s.strip()
    
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
        Lp = self._find_suffix_start(full_ids, tgt_ids)
        if Lp < 0:
            Lp = L_full - L_tgt

        logits = self.model(**full_enc).logits.float()   # [1,L,V]
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
            self.prof.log(tag="mi:lp:call", wall_s=time.perf_counter()-t0, gen_tokens=0, prompt_len=L_full, target_len=L_tgt)
        except Exception:
            pass
        return LP_sum, LP_mean, eff_len

    @torch.no_grad()
    def _logprob_total_batch(self, prompts: List[str], targets: List[str], batch_size: int = 12) -> List[Tuple[float, int]]:
        assert len(prompts) == len(targets)
        self._ensure_pad_token()
        self.model.eval()
        if not hasattr(self, "_H_CACHE"):
            self._H_CACHE = {}

        out = [None] * len(prompts)
        miss_idx, miss_prompts, miss_targets = [], [], []
        for i, (p, t) in enumerate(zip(prompts, targets)):
            key = ("LP|pair", p, "\u241E", t) 
            if key in self._H_CACHE: out[i] = self._H_CACHE[key]
            else:
                miss_idx.append(i); miss_prompts.append(p); miss_targets.append(t)
        if not miss_idx:
            return out

        use_amp = torch.cuda.is_available()
        for s in range(0, len(miss_idx), batch_size):
            e = min(s + batch_size, len(miss_idx))
            Ps = miss_prompts[s:e]; Ts = miss_targets[s:e]
            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
                full_enc = self.tokenizer([p + t for p, t in zip(Ps, Ts)],
                    return_tensors="pt", add_special_tokens=True, padding=True, truncation=False).to(self.device)
                logits = self.model(**full_enc).logits.float()    # [B,L,V]
                logits_shifted = logits[:, :-1, :]                # [B,L-1,V]

            # tokenize targets (no specials) for suffix alignment + gather indices
            enc_t = self.tokenizer(Ts, return_tensors="pt", add_special_tokens=False, padding=True, truncation=False)
            B, Lm1, V = logits_shifted.shape
            full_ids = full_enc["input_ids"]; ids_t = enc_t["input_ids"]

            for bi in range(B):
                tgt_ids_list = ids_t[bi].tolist()
                L_tgt = int((ids_t[bi] != self.tokenizer.pad_token_id).sum().item()) if self.tokenizer.pad_token_id is not None else len(tgt_ids_list)
                if L_tgt <= 0 or Lm1 <= 0:
                    LP_sum, eff_len = 0.0, 0 
                else:
                    tgt_ids = tgt_ids_list[:L_tgt]
                    full_ids_list = full_ids[bi].tolist()
                    Lp = self._find_suffix_start(full_ids_list, tgt_ids)
                    if Lp < 0:  # fallback
                        # enc_p_spec = self.tokenizer(Ps[bi], add_special_tokens=True)
                        # Lp = max(len(enc_p_spec["input_ids"]), 1)
                        Lp = len(full_ids_list) - L_tgt

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
                key = ("LP|pair", prompts[gi], "\u241E", targets[gi])
                pair = (LP_sum, eff_len)
                self._H_CACHE[key] = pair
                out[gi] = pair
        return out
    
    @torch.no_grad()
    def _logprob_mean_batch(self, prompts: List[str], targets: List[str], batch_size: int = 12) -> List[float]:
        sums_lens = self._logprob_total_batch(prompts, targets, batch_size)
        means = []
        for (LP_sum, eff_len) in sums_lens:
            means.append(LP_sum / max(eff_len, 1))
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
    
    # ---------------- negatives (heuristic + sampled) -----------
    def _heuristic_negatives(self, gold: str, max_h: int = 8) -> List[str]:
        g = self._norm_ans_text(gold)
        cands = set()
        def add(x):
            x = self._norm_ans_text(x)
            if x and x != g: cands.add(self._wrap_ans(x))

        # 1) 분수 a/b 형태
        mf = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", g)
        if mf:
            a, b = int(mf.group(1)), int(mf.group(2))
            for da, db in [(-1,0),(1,0),(0,-1),(0,1),(-2,0),(0,2)]:
                nb, na = b+db, a+da
                if nb != 0: add(f"\\frac{{{na}}}{{{nb}}}")
            if a!=0: add(f"\\frac{{{b}}}{{{a}}}")
            add(f"\\frac{{{-a}}}{{{b}}}")
        else:
            # 2) 정수/실수
            mn = re.fullmatch(r"-?\d+(?:\.\d+)?", g)
            if mn:
                x = float(g)
                pool = [x-2, x-1, x+1, x+2, -x, x*0.9, x*1.1, math.floor(x), math.ceil(x)]
                for v in pool:
                    if abs(v - round(v)) < 1e-9: add(str(int(round(v))))
                    else: add(f"{v:.6g}")
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
                    a,b = g.split("/",1); add(b+"/"+a)
                if "+" in g: add(g.replace("+","-"))
                if "-" in g and not g.startswith("-"): add(g.replace("-","+"))
                add(g+"+1"); add(g+"-1")
        # 5) 포맷 통일 및 상한
        outs = list(cands); random.shuffle(outs)
        return outs[:max_h]
    
    @torch.no_grad()
    def _sample_model_negatives(self, question: str, steps: List[str], gold: str, n: int = 4, max_new_tokens: int = 24, temperature: float = 1.1, top_p: float = 0.9, log_ctx: Optional[Dict[str, Any]] = None) -> List[str]:
        t0 = time.perf_counter()
        prompt = self.build_prompt(question) + "".join(s.rstrip().rstrip("\n") + "\n" for s in steps) + "\nThe answer is: "
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = enc["input_ids"].shape[1]

        out = self.model.generate(**enc, do_sample=True, temperature=temperature, top_p=top_p,
            num_return_sequences=n, max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.eos_token_id)

        total_new = 0
        for seq in out:
            total_new += max(0, int(seq.shape[0]) - input_len)

        texts = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        cands = set()
        g = self._norm_ans_text(gold)
        for t in texts:
            tail = t[len(prompt):]
            tail = tail.split("\n")[0]
            tail = re.split(r"(?:The\s+answer\s+is\s*:)", tail, flags=re.IGNORECASE)[0]
            ans = self._norm_ans_text(tail)
            if ans and ans != g and len(ans) <= 32:
                cands.add(self._wrap_ans(ans))
        
        try: self.prof.log(tag="neg:sample", wall_s=time.perf_counter() - t0, gen_tokens=total_new, num_samples=n, **(log_ctx or {}))
        except Exception: pass
        return list(cands)
    
    def _is_valid_wrapped(self, a_wrapped: str) -> bool:
        ans = self._extract_ans_from_text(a_wrapped)
        return bool(ans) and (len(ans) <= 32)

    def _build_wrong_answer_candidates(self, question: str, steps: List[str], gold_answer: str, max_candidates: int = 8, n_sample: int = 4, log_ctx: Optional[Dict[str, Any]] = None) -> List[str]:
        gold_answer = self._norm_ans_text(gold_answer)
        pool = []
        # 1) model-sampled (logged with gen_tokens above)
        try: pool += self._sample_model_negatives(question, steps, gold_answer, n=n_sample, log_ctx=log_ctx)
        except Exception: pass
        # 2) heuristics (no generation)
        pool += self._heuristic_negatives(gold_answer, max_h=max_candidates)
        # 3) 중복/골드 제거 + 간단 유효성 필터
        uniq, seen = [], set([self._wrap_ans(gold_answer)])
        for a in pool:
            if a not in seen and self._is_valid_wrapped(a):
                seen.add(a); uniq.append(a)
                if len(uniq) >= max_candidates: break
        if len(uniq) < max_candidates:
            more = self._heuristic_negatives(gold_answer, max_h=max_candidates*4)
            for a in more:
                if a not in seen and self._is_valid_wrapped(a):
                    seen.add(a); uniq.append(a)
                    if len(uniq) >= max_candidates: break
        return uniq[:max_candidates]
    
    # --------------- Contrastive + Ensemble (Batch) ---------------
    def _bases_for_templates(self, question: str, template_ids: List[int]) -> List[str]:
        return [self.build_prompt(question, template_id=t) + "\n" for t in template_ids]
    
    def compute_step_contrastive_cmi_ensemble_batch(self, question: str, steps: List[str], gold_target: str, neg_targets: List[str],
        template_ids: List[int], batch_size: int = 12, normalize: bool = True) -> Tuple[List[float], int]:
        """ Δ_i^cont (CMI): [LP(A*|<=i)-LP(A*|<i)] - mean_m [LP(Ã|<=i)-LP(Ã|<i)]"""
        bases = self._bases_for_templates(question, template_ids)
        all_prev, all_with = [], []
        for b in bases:
            prefixes = self._prefixes_from_base_and_steps(b, steps)
            all_prev += prefixes[:-1]   # K * S
            all_with += prefixes[1:]    # K * S
        S, K = len(steps), len(template_ids)

        # gold
        if normalize:
            LP_prev_g = self._logprob_mean_batch(all_prev, [gold_target]*len(all_prev), batch_size)
            LP_with_g = self._logprob_mean_batch(all_with, [gold_target]*len(all_with), batch_size)
        else:
            LP_prev_g_pairs = self._logprob_total_batch(all_prev, [gold_target]*len(all_prev), batch_size)
            LP_with_g_pairs = self._logprob_total_batch(all_with, [gold_target]*len(all_with), batch_size)
            LP_prev_g = [s for (s, L) in LP_prev_g_pairs]
            LP_with_g = [s for (s, L) in LP_with_g_pairs]

        out = [0.0]*S
        cur = 0
        for _ in range(K):
            for i in range(S):
                out[i] += (LP_with_g[cur+i] - LP_prev_g[cur+i])
            cur += S

        # negatives
        if neg_targets:
            neg_acc = [0.0]*S
            for neg in neg_targets:
                if normalize:
                    n_prev = self._logprob_mean_batch(all_prev, [neg]*len(all_prev), batch_size)
                    n_with = self._logprob_mean_batch(all_with, [neg]*len(all_with), batch_size)
                else:
                    n_prev_pairs = self._logprob_total_batch(all_prev, [neg]*len(all_prev), batch_size)
                    n_with_pairs = self._logprob_total_batch(all_with, [neg]*len(all_with), batch_size)
                    n_prev = [s for (s, L) in n_prev_pairs]
                    n_with = [s for (s, L) in n_with_pairs]
                cur = 0
                for _ in range(K):
                    for i in range(S):
                        neg_acc[i] += (n_with[cur+i] - n_prev[cur+i])
                    cur += S
            M = float(len(neg_targets))
            for i in range(S):
                out[i] = (out[i]/K) - (neg_acc[i]/(K*M))
        else:
            out = [v / K for v in out]

        # gen_tokens estimate: each pair uses 2 contexts per step
        Lg = self._target_len(gold_target)
        Ln_sum = sum(self._target_len(n) for n in neg_targets) if neg_targets else 0
        gen_tokens = K * S * 2 * (Lg + Ln_sum)
        return out, 0
    
    def compute_step_contrastive_marginal_ensemble_batch(self, question: str, steps: List[str], gold_target: str, neg_targets: List[str],
        template_ids: List[int], batch_size: int = 12, normalize: bool = True) -> Tuple[List[float], int]:
        """ Δ_i^cont (Marginal): [LP(A*|base+S_i)-LP(A*|base)] - mean_m[LP(Ã|base+S_i)-LP(Ã|base)]"""
        bases = self._bases_for_templates(question, template_ids)
        S, K = len(steps), len(template_ids)
        all_base_unique = bases
        all_with_i = []
        for b in bases:
            all_with_i += [b + s.rstrip().rstrip("\n") + "\n" for s in steps]  # len K*S

        if normalize:
            LP_base_g = self._logprob_mean_batch(all_base_unique, [gold_target]*K, batch_size)
            LP_with_g = self._logprob_mean_batch(all_with_i,     [gold_target]*(K*S), batch_size)
        else:
            base_pairs = self._logprob_total_batch(all_base_unique, [gold_target]*K, batch_size)
            with_pairs = self._logprob_total_batch(all_with_i,      [gold_target]*(K*S), batch_size)
            LP_base_g = [s for (s, L) in base_pairs]
            LP_with_g = [s for (s, L) in with_pairs]

        out = [0.0]*S
        cur = 0
        for e in range(K):
            base_e = LP_base_g[e]
            for i in range(S):
                out[i] += (LP_with_g[cur+i] - base_e)
            cur += S

        if neg_targets:
            neg_acc = [0.0]*S
            for neg in neg_targets:
                if normalize:
                    n_base = self._logprob_mean_batch(all_base_unique, [neg]*K, batch_size)
                    n_with = self._logprob_mean_batch(all_with_i,      [neg]*(K*S), batch_size)
                else:
                    n_base_pairs = self._logprob_total_batch(all_base_unique, [neg]*K, batch_size)
                    n_with_pairs = self._logprob_total_batch(all_with_i,      [neg]*(K*S), batch_size)
                    n_base = [s for (s, L) in n_base_pairs]
                    n_with = [s for (s, L) in n_with_pairs]
                cur = 0
                for e in range(K):
                    base_e = n_base[e]
                    for i in range(S):
                        neg_acc[i] += (n_with[cur+i] - base_e)
                    cur += S
            M = float(len(neg_targets))
            out = [(v/K) - (neg_acc[i]/(K*M)) for i, v in enumerate(out)]
        else:
            out = [v / K for v in out]

        # gen_tokens estimate: per template (S + 1) contexts per target
        Lg = self._target_len(gold_target)
        Ln_sum = sum(self._target_len(n) for n in neg_targets) if neg_targets else 0
        gen_tokens = (K*(1+M) + K*S*(1+M)) * (Lg) + (K*(1+M) + K*S*(1+M)) * (Ln_sum)
        return out, 0

    def compute_step_contrastive_loo_ensemble_batch(self, question: str, steps: List[str], gold_target: str, neg_targets: List[str],
        template_ids: List[int], batch_size: int = 12, normalize: bool = True) -> Tuple[List[float], int]:
        """ Δ_i^cont (LOO): [LP(A*|all)-LP(A*|without_i)] - mean_m[LP(Ã|all)-LP(Ã|without_i)]"""
        bases = self._bases_for_templates(question, template_ids)
        all_all_ctx, all_wo_ctx = [], []
        block_starts = []
        for b in bases:
            block_starts.append(len(all_wo_ctx))
            with_all = b + "".join(s.rstrip().rstrip("\n") + "\n" for s in steps)
            all_all_ctx.append(with_all)
            for i in range(len(steps)):
                wo = b + "".join(steps[j].rstrip().rstrip("\n") + "\n" for j in range(len(steps)) if j != i)
                all_wo_ctx.append(wo)
        S, K = len(steps), len(template_ids)

        # gold
        if normalize:
            LP_all_g = self._logprob_mean_batch(all_all_ctx, [gold_target]*len(all_all_ctx), batch_size)
            LP_wo_g  = self._logprob_mean_batch(all_wo_ctx,  [gold_target]*len(all_wo_ctx),  batch_size)
        else:
            LP_all_pairs = self._logprob_total_batch(all_all_ctx, [gold_target]*len(all_all_ctx), batch_size)
            LP_wo_pairs  = self._logprob_total_batch(all_wo_ctx,  [gold_target]*len(all_wo_ctx),  batch_size)
            LP_all_g = [s for (s, L) in LP_all_pairs]
            LP_wo_g  = [s for (s, L) in LP_wo_pairs]

        out = [0.0]*S
        for e in range(K):
            o = block_starts[e]
            base_all = LP_all_g[e]
            for i in range(S):
                out[i] += (base_all - LP_wo_g[o + i])

        if neg_targets:
            neg_acc = [0.0]*S
            for neg in neg_targets:
                if normalize:
                    LP_all_n = self._logprob_mean_batch(all_all_ctx, [neg]*len(all_all_ctx), batch_size)
                    LP_wo_n  = self._logprob_mean_batch(all_wo_ctx,  [neg]*len(all_wo_ctx),  batch_size)
                else:
                    LP_all_pairs = self._logprob_total_batch(all_all_ctx, [neg]*len(all_all_ctx), batch_size)
                    LP_wo_pairs  = self._logprob_total_batch(all_wo_ctx,  [neg]*len(all_wo_ctx),  batch_size)
                    LP_all_n = [s for (s, L) in LP_all_pairs]
                    LP_wo_n  = [s for (s, L) in LP_wo_pairs]
                for e in range(K):
                    o = block_starts[e]
                    base_all = LP_all_n[e]
                    for i in range(S):
                        neg_acc[i] += (base_all - LP_wo_n[o + i])
            M = float(len(neg_targets))
            for i in range(S):
                out[i] = (out[i]/K) - (neg_acc[i]/(K*M))
        else:
            out = [v / K for v in out]

        # gen_tokens estimate: per template (S + 1) contexts per target
        Lg = self._target_len(gold_target)
        Ln_sum = sum(self._target_len(n) for n in neg_targets) if neg_targets else 0
        gen_tokens = K * (S + 1) * (Lg + Ln_sum)
        return out, 0
    
    # --------- Naive (robust) variants used as fallbacks ----------
    def _cmi_from_base_naive(self, base: str, steps: List[str], gold_target: str, neg_targets: List[str], normalize: bool = True) -> List[float]:
        vals = []
        prompt = base
        gold_prev = self._LP_cached_mean(prompt, gold_target) if normalize else self._LP_cached_sum(prompt, gold_target)
        for s in steps:
            prompt_with = prompt + s.rstrip().rstrip("\n") + "\n"
            gold_with = self._LP_cached_mean(prompt_with, gold_target) if normalize else self._LP_cached_sum(prompt_with, gold_target)
            diff = gold_with - gold_prev
            if neg_targets:
                acc = 0.0
                for neg in neg_targets:
                    n_prev = self._LP_cached_mean(prompt, neg) if normalize else self._LP_cached_sum(prompt, neg)
                    n_with = self._LP_cached_mean(prompt_with, neg) if normalize else self._LP_cached_sum(prompt_with, neg)
                    acc += (n_with - n_prev)
                diff -= acc / float(len(neg_targets))
            vals.append(diff)
            prompt = prompt_with; gold_prev = gold_with
        return vals

    def _marginal_from_base_naive(self, base: str, steps: List[str], gold_target: str, neg_targets: List[str], normalize: bool = True) -> List[float]:
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

    def _loo_from_base_naive(self, base: str, steps: List[str], gold_target: str, neg_targets: List[str], normalize: bool = True) -> List[float]:
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

    def compute_step_contrastive_cmi_ensemble_naive(self, question: str, steps: List[str], gold_target: str, neg_targets: List[str], template_ids: List[int], normalize: bool = True) -> List[float]:
        S = len(steps); agg = [0.0]*S
        for t in template_ids:
            base = self.build_prompt(question, template_id=t) + "\n"
            vals = self._cmi_from_base_naive(base, steps, gold_target, neg_targets, normalize=normalize)
            for i in range(S): agg[i] += vals[i]
        return [v/len(template_ids) for v in agg]

    def compute_step_contrastive_marginal_ensemble_naive(self, question: str, steps: List[str], gold_target: str, neg_targets: List[str], template_ids: List[int], normalize: bool = True) -> List[float]:
        S = len(steps); agg = [0.0]*S
        for t in template_ids:
            base = self.build_prompt(question, template_id=t) + "\n"
            vals = self._marginal_from_base_naive(base, steps, gold_target, neg_targets, normalize=normalize)
            for i in range(S): agg[i] += vals[i]
        return [v/len(template_ids) for v in agg]

    def compute_step_contrastive_loo_ensemble_naive(self, question: str, steps: List[str], gold_target: str, neg_targets: List[str], template_ids: List[int], normalize: bool = True) -> List[float]:
        S = len(steps); agg = [0.0]*S
        for t in template_ids:
            base = self.build_prompt(question, template_id=t) + "\n"
            vals = self._loo_from_base_naive(base, steps, gold_target, neg_targets, normalize=normalize)
            for i in range(S): agg[i] += vals[i]
        return [v/len(template_ids) for v in agg]
    
    # ----------------- public: labeling (stream) -----------------
    def mi_labelling(self, *, ds, ds_task_tag: Optional[str] = None, template_ids: Optional[List[int]] = None, batch_size: int = 12, normalize: bool = True,
        n_sample_negs: int = 5, max_negatives: int = 8) -> Iterable[Dict[str, Any]]:
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
            neg_targets = self._build_wrong_answer_candidates(question, steps, gold_answer, max_candidates=max_negatives, n_sample=n_sample_negs, log_ctx={"dataset": task, "sample_idx": si})
            M = len(neg_targets)

            # ---- Ensemble CMI ----
            try:
                t1 = time.perf_counter()
                cmi_vals, cmi_tok = self.compute_step_contrastive_cmi_ensemble_batch(question, steps, answer_target, neg_targets, template_ids=template_ids, batch_size=batch_size, normalize=normalize)
                wall_cmi = time.perf_counter() - t1
                num_prompts_cmi = K * 2 * len(steps) * (1 + len(neg_targets))
                self.prof.log(tag="cont:cmi.ensemble", wall_s=wall_cmi, num_prompts=num_prompts_cmi, gen_tokens=cmi_tok, n=K, dataset=task, sample_idx=si, peak_mem_gb=_peak_mem_gb())
            except Exception:
                # robust fallback
                t1 = time.perf_counter()
                cmi_vals = self.compute_step_contrastive_cmi_ensemble_naive(question, steps, answer_target, neg_targets, template_ids=template_ids, normalize=normalize)
                wall_cmi = time.perf_counter() - t1
                num_prompts_cmi = K * 2 * len(steps) * (1 + len(neg_targets))
                self.prof.log(tag="cont:cmi.ensemble.naive", wall_s=wall_cmi, num_prompts=num_prompts_cmi, gen_tokens=0, n=K, dataset=task, sample_idx=si, peak_mem_gb=_peak_mem_gb())

            # ---- Ensemble Marginal ----
            try:
                t1 = time.perf_counter()
                marg_vals, marg_tok = self.compute_step_contrastive_marginal_ensemble_batch(question, steps, answer_target, neg_targets, template_ids=template_ids, batch_size=batch_size, normalize=normalize)
                wall_marg = time.perf_counter() - t1
                num_prompts_marg = K * (len(steps) + 1) * (1 + len(neg_targets))
                self.prof.log(tag="cont:marginal.ensemble", wall_s=wall_marg, num_prompts=num_prompts_marg, gen_tokens=marg_tok, n=K, dataset=task, sample_idx=si, peak_mem_gb=_peak_mem_gb())
            except Exception:
                t1 = time.perf_counter()
                marg_vals = self.compute_step_contrastive_marginal_ensemble_naive(question, steps, answer_target, neg_targets, template_ids=template_ids, normalize=normalize)
                wall_marg = time.perf_counter() - t1
                num_prompts_marg = K * (len(steps) + 1) * (1 + len(neg_targets))
                self.prof.log(tag="cont:marginal.ensemble.naive", wall_s=wall_marg, num_prompts=num_prompts_marg, gen_tokens=0, n=K, dataset=task, sample_idx=si, peak_mem_gb=_peak_mem_gb())

            # ---- Ensemble LOO ----
            try:
                t1 = time.perf_counter()
                loo_vals, loo_tok = self.compute_step_contrastive_loo_ensemble_batch(question, steps, answer_target, neg_targets, template_ids=template_ids, batch_size=batch_size, normalize=normalize)
                wall_loo = time.perf_counter() - t1
                num_prompts_loo = K * (len(steps) + 1) * (1 + len(neg_targets))
                self.prof.log(tag="cont:loo.ensemble", wall_s=wall_loo, num_prompts=num_prompts_loo, gen_tokens=loo_tok, n=K, dataset=task, sample_idx=si, peak_mem_gb=_peak_mem_gb())
            except Exception:
                t1 = time.perf_counter()
                loo_vals = self.compute_step_contrastive_loo_ensemble_naive(question, steps, answer_target, neg_targets, template_ids=template_ids, normalize=normalize)
                wall_loo = time.perf_counter() - t1
                num_prompts_loo = K * (len(steps) + 1) * (1 + len(neg_targets))
                self.prof.log(tag="cont:loo.ensemble.naive", wall_s=wall_loo, num_prompts=num_prompts_loo, gen_tokens=0, n=K, dataset=task, sample_idx=si, peak_mem_gb=_peak_mem_gb())

            entry = {
                "question": question,
                "completion": steps,
                "original_answer": gold_answer,
                "answer_target": answer_target,
                "pll_cmi_cont": cmi_vals,
                "pll_loo_cont": loo_vals,
                "pll_marginal_cont": marg_vals,
                "correct_mask": correct_mask,
                "task": task,
            }
            yield entry

            try: self.prof.log(tag="sample_total", dataset=task, sample_idx=si, wall_s=time.perf_counter() - t0)
            except Exception: pass

        try: self.prof.log(tag="dataset_total", wall_s=time.perf_counter() - t_ds0)
        except Exception: pass
