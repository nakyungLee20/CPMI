import multiprocessing as mp, os
try:
    mp.set_start_method("spawn", force=True)  # or "forkserver"
except RuntimeError:
    pass
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("SYMPY_GROUND_TYPES", "python")  # gmpy2 회피
os.environ.setdefault("SYMPY_USE_CACHE", "no")
import re, random, math, json, time, sys, pathlib, threading
import numpy as np
from utils import numeric_equiv_fallback
from typing import List, Optional, Any, Dict
import torch
from tqdm import tqdm
import time
from run_profile import RunProfiler
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from math_scorer import MATHScorer

def _grade_worker(pred: str, gold: str) -> bool:
    from math_scorer import MATHScorer as _Scorer
    return bool(_Scorer.grade(pred, gold))

class BaseHardReward:
    _STEP_RE = re.compile(r"(?:^|\s)(Step\s+\d+\s*:\s*)", flags=re.IGNORECASE)
    _ANS_RE  = re.compile(r"The\s+answer\s+is\s*:\s*(.+?)\s*(?:[+\-]\s*$|\s*$)", flags=re.IGNORECASE | re.DOTALL)
    ANSWER_PATTERN = re.compile(
        r"""
        (?:
            \\boxed\s*\{([^{}]+)\}           |   # \boxed{...}
            \\fbox\s*\{([^{}]+)\}            |   # \fbox{...}
            [\(\[\{]?\s*\\boxed\s*\{([^{}]+)\}\s*[\)\]\}]? |
            [\$\s]*([^\n$]+?)\s*[\$]*$           # 마지막 라인 형태
        )
        """,
        re.IGNORECASE | re.VERBOSE | re.DOTALL,
    )
    _ANSWER_RE = re.compile(r"^#{1,6}\s*answer\s*[:\-]?\s*(.+)$", re.IGNORECASE)
    
    def __init__(self, config, model_name: str = "Qwen/Qwen3-4B-base", base_type: str = "qval", prover_model_name: Optional[str] = "Qwen/Qwen2.5-1.5B"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.base_type = base_type
        self.prover_model_name = prover_model_name

        from vllm import LLM, SamplingParams  # 지연 임포트
        self._SamplingParams = SamplingParams
        self.llm = LLM(
            model=model_name,
            download_dir="/data/hub",
            trust_remote_code=True,
            dtype="bfloat16",
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_mem_util,
            max_model_len=self.config.max_model_len,
            max_num_seqs=24,
        )
        self.tokenizer = self.llm.get_tokenizer()
        # default rollout params
        self.rollout_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            truncate_prompt_tokens=max(0, self.config.max_model_len - self.config.max_new_tokens - 16),
            max_tokens=self.config.max_new_tokens,
            n=self.config.num_rollouts,
            repetition_penalty=1.05,
        )
        print(f"vLLM model loaded: {model_name}")

        if self.base_type == "pav" and self.prover_model_name:
            self._ensure_prover(self.prover_model_name)
            print(f"vLLM prover model loaded: {prover_model_name}")

        self._init_grade_pool()
        self._grade_timeout_s = 300 
        self._grade_sem = threading.Semaphore(value=max(2, min(8, (os.cpu_count() or 8)//2)))  # 동시 채점 제한
        self._grade_fail_streak = 0
        self._grade_fail_trip = 24
        self._circuit_open_until = 0.0
        self.prof = RunProfiler()

    # ---- Optional separate prover backend (HF or vLLM) helpers ----
    class _SimplePiece:
        def __init__(self, text: str):
            self.text = text
    class _SimpleResult:
        def __init__(self, texts: List[str]):
            self.outputs = [BaseHardReward._SimplePiece(t) for t in texts]

    # ------------------------ 워커 풀 ------------------------
    def _init_grade_pool(self):
        workers = max(4, min(16, (os.cpu_count() or 8)))
        self._grade_pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="grader")

    def _reset_grade_pool(self):
        try:
            self._grade_pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        self._init_grade_pool()

    # ------------------------ Core prompting helpers ------------------------
    def build_prompt_with_prefix(self, question: str, prefix_steps: List[str], tokenizer=None) -> str:
        tok = tokenizer or self.tokenizer
        sys_prompt = ("You are a math tutor. Continue the reasoning from the given partial steps and finish with 'The answer is: <final answer>'. Keep the same 'Step k: ...' format.")
        prefix_txt = "\n".join(prefix_steps) if prefix_steps else ""
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Problem: {question}"},
        ]
        base = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if prefix_txt:
            base += f"\n\n{prefix_txt}\n"
        return base
    
    def _batched_generate(self, prompts: List[str], params: Any, *, tag: str = "policy"):
        B = getattr(self.config, "max_prompts_per_call", 32)
        if torch.cuda.is_available():
            try: torch.cuda.reset_peak_memory_stats()
            except Exception: pass
        
        t0 = time.perf_counter()
        # results = self.llm.generate(prompts, params)
        all_results = []
        for start in range(0, len(prompts), B):
            chunk = prompts[start:start+B]
            # 청크별 generate
            chunk_results = self.llm.generate(chunk, params)
            all_results.extend(chunk_results)
        t1 = time.perf_counter()
        wall = t1 - t0
        # best-effort 토큰 카운팅
        gen_tokens = 0
        for r in all_results:
            for o in getattr(r, "outputs", []):
                if hasattr(o, "token_ids") and o.token_ids is not None:
                    gen_tokens += len(o.token_ids)
                else:
                    gen_tokens += len(o.text.split())
        peak = None
        if torch.cuda.is_available():
            try: peak = torch.cuda.max_memory_allocated() / (1024**3)
            except Exception: pass
        self.prof.log(tag=tag, num_prompts=len(prompts), n=getattr(params, "n", 1), wall_s=wall, gen_tokens=gen_tokens, peak_mem_gb=peak)
        return all_results
    
    # ------------------------ prover model loading helpers ------------------------
    def _ensure_prover(self, model_name: str = "Qwen/Qwen2.5-1.5B"):
        from vllm import LLM, SamplingParams 
        if hasattr(self, "_prover") and getattr(self, "_prover", None) is not None:
            if getattr(self, "_prover_name", None) == model_name:
                return 
            else:
                pass
        self._prover = LLM(
            model=model_name,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=0.15,
            max_model_len=8000,
            tensor_parallel_size=getattr(self.config, "tensor_parallel_size", 1),
        )
        self._prover_tok = self._prover.get_tokenizer()
        self._prover_name = model_name
        rp = self.rollout_params
        self._prover_params = SamplingParams(
            temperature=getattr(rp, "temperature", 0.6),
            top_p=getattr(rp, "top_p", 0.95),
            top_k=getattr(rp, "top_k", 20),
            truncate_prompt_tokens=getattr(rp, "truncate_prompt_tokens", max(0, 8000 - 1024 - 16)),
            max_tokens=getattr(rp, "max_tokens", 1024),
            n=4,
            repetition_penalty=getattr(rp, "repetition_penalty", 1.05),
        )
    
    def _prover_generate(self, prompts: List[str], params: Any, *, tag: str = "prover"):
        sp = params or getattr(self, "_prover_params", None) or self.rollout_params
        if torch.cuda.is_available():
            try: torch.cuda.reset_peak_memory_stats()
            except Exception: pass
        t0 = time.perf_counter()
        results = self._prover.generate(prompts, sp)
        wall = time.perf_counter() - t0
        peak = None
        if torch.cuda.is_available():
            try: peak = torch.cuda.max_memory_allocated() / (1024**3)
            except Exception: pass
        gen_tokens = 0
        for r in results:
            for o in getattr(r, "outputs", []):
                gen_tokens += len(getattr(o, "token_ids", []) or o.text.split())
        self.prof.log(tag=tag, num_prompts=len(prompts), n=getattr(sp, "n", 1), wall_s=wall, gen_tokens=gen_tokens, peak_mem_gb=peak)
        return results

    # ------------------------ Parsing helpers ------------------------
    def _clean_spaces(self, s: str) -> str:
        s = s.replace("\u200b", " ").replace("\xa0", " ")
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"\s+\n", "\n", s)
        return s.strip()
    
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
        ma = self._ANS_RE.search(txt)
        if ma:
            body = self._clean_spaces(ma.group(1))
            body = re.sub(r"\s*[+\-]\s*$", "", body)  # 끝의 +/- 정리
            gold_answer = body

        # ---------- NEW: dataset에 붙인 gold_answer가 있으면 무조건 우선 ----------
        ds_gold = rec.get("gold_answer", None)
        if isinstance(ds_gold, str):
            ds_gold = self._clean_spaces(ds_gold)
        if ds_gold:  # gold가 있으면 라벨 기반 추출을 덮어씀
            gold_answer = ds_gold

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
            "task": rec.get("task", None),
            "raw": txt,
        }
 
    # ------------------------ 안전 채점 (회로차단 포함) ------------------------
    def _grade_safe(self, pred: str, gold: str) -> bool:
        now = time.time()
        # 회로 차단: 최근 오류가 많으면 잠깐 동기/폴백만 사용
        if now < self._circuit_open_until:
            return numeric_equiv_fallback(pred, gold)

        with self._grade_sem:  # 동시 채점 제한
            fut = self._grade_pool.submit(_grade_worker, pred, gold)
            try:
                ok = bool(fut.result(timeout=self._grade_timeout_s))
                # 성공 시 연속 실패 카운터 리셋
                self._grade_fail_streak = max(0, self._grade_fail_streak - 1)
                return ok
            except TimeoutError:
                self._grade_fail_streak += 1
                self._reset_grade_pool()
                pass
            except Exception:
                self._grade_fail_streak += 1
                self._reset_grade_pool()
                pass

        # 연속 실패가 임계 넘으면 60초간 회로 오픈 (폴백만)
        if self._grade_fail_streak >= self._grade_fail_trip:
            self._circuit_open_until = time.time() + 60.0
        return numeric_equiv_fallback(pred, gold)
    
    def _score_one_prompt_outputs(self, result, gold_answer: str) -> float:
        correct = 0
        total = max(1, len(result.outputs))
        for comp in result.outputs:
            ans = MATHScorer.extract_pred(comp.text)
            if ans and self._grade_safe(ans, gold_answer):
                correct += 1
        return correct / float(total)

    # ── Baselines Reward Generation Algorithms ────────────────────────────────────────────────
    def compute_qval_step_rewards(self, question: str, steps: List[str], gold_answer: str, tokenizer=None) -> List[float]:
        """Monte Carlo Q-value baseline (Math-Shepherd style). For each prefix s_{1:i}, rollout n completions and take success frequency. Returns a list of length len(steps)."""
        prompts = []
        for i in range(len(steps)):
            prefix = steps[: i + 1]
            prompts.append(self.build_prompt_with_prefix(question, prefix, tokenizer))
        results = self._batched_generate(prompts, self.rollout_params, tag="policy:qval")
        rewards = [self._score_one_prompt_outputs(res, gold_answer) for res in results]
        return rewards

    def compute_pav_step_rewards(self, question: str, steps: List[str], gold_answer: str, tokenizer=None, *, alpha: float = 1.0, prover_model_name: Optional[str] = "Qwen/Qwen3-1.7B") -> List[float]:
        """Monte Carlo Advantage (Rewarding-Progress). We first estimate Q for each prefix and for the empty prefix Q[-1].
        If `prover_model_name` is None, uses the POLICY model for both Q and A (original behavior).
        If provided, computes Q^π with the policy (self.llm) and Advantage A^μ with a separate prover μ:  A_i^μ = Q_i^μ - Q_{i-1}^μ, and returns Q_i^π + α * A_i^μ."""
        # ---- Build prompts for prefixes ----
        empty_prompt = self.build_prompt_with_prefix(question, [], tokenizer)
        policy_prompts = [empty_prompt] + [self.build_prompt_with_prefix(question, steps[: i + 1], tokenizer) for i in range(len(steps))]

        # Q^π with rollout (policy)
        policy_results = self._batched_generate(policy_prompts, self.rollout_params, tag="policy:pav-Q")
        q_pi = [self._score_one_prompt_outputs(res, gold_answer) for res in policy_results]  # len L+1

        if prover_model_name:  # separate prover for A-term
            prover_prompts = policy_prompts  # identical prefixes
            prover_results = self._prover_generate(prover_prompts, None, tag="prover:pav-A")
            q_mu = [self._score_one_prompt_outputs(res, gold_answer) for res in prover_results]
        else:
            q_mu = q_pi  # fallback: same-model advantage

        q_pi_empty, q_mu_empty = q_pi[0], q_mu[0]
        q_pi_per = q_pi[1:]  # length L
        q_mu_per = q_mu[1:]

        pav_rewards: List[float] = []
        prev_mu = q_mu_empty
        for qi_pi, qi_mu in zip(q_pi_per, q_mu_per):
            A_mu = qi_mu - prev_mu
            pav = qi_pi + alpha * A_mu
            pav_rewards.append(pav)
            prev_mu = qi_mu
        return pav_rewards
    
    def mc_labelling(self, *, ds, base_type: str = "qval", ds_task_tag: Optional[str] = None):
        t_ds0 = time.perf_counter()
        for si, rec in tqdm(enumerate(ds)):
            t0 = time.perf_counter()
            parsed = self.parse_math_shepherd_record(rec)
            question      = parsed["question"]
            steps         = parsed["steps"]
            gold_answer   = parsed["gold_answer"]
            correct_mask  = parsed["correct_mask"]
            task          = parsed.get("task") or ds_task_tag or "mathshepherd"

            if not question or not steps or not gold_answer:
                try:
                    self.prof.log(tag="skip_empty", dataset=task, sample_idx=si)
                except Exception:
                    pass
                continue

            if base_type == "qval":
                base_reward = self.compute_qval_step_rewards(question, steps, gold_answer, tokenizer=self.tokenizer)
            elif base_type == "pav":
                base_reward = self.compute_pav_step_rewards(question, steps, gold_answer, tokenizer=self.tokenizer)
            else:
                raise ValueError("No valid baseline reward type: expected 'qval' or 'pav'.")

            entry = {
                "question": question,
                "completion": steps,              # 원본 step들
                "gold_answer": gold_answer,   # gold (정규화됨)
                "base_reward": base_reward,       # 길이 = len(steps)
                "correct_mask": correct_mask,     # +/− 라벨
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

