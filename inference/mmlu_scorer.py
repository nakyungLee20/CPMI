import re, math
from fractions import Fraction
from decimal import Decimal, InvalidOperation
import string
from typing import List, Optional, Tuple, Dict, Union


class MmluScorer:
    LABELS = "ABCD"

    # --- patterns (last match wins) ---
    _BOX_ANY      = re.compile(r"\\boxed\s*\{\s*([^}]*)\s*\}", re.I)  # <- payload 전체
    _ANS_LINE     = re.compile(r"(?:^|\n)\s*(?:the\s+)?answer\s*(?:is|=|:)\s*\(?([A-Da-d0-3])\)?", re.I)
    _CORR_LINE    = re.compile(r"(?:^|\n)\s*(?:the\s+)?correct\s+(?:answer|choice)\s*(?:is|=|:)\s*\(?([A-Da-d0-3])\)?", re.I)
    _PAREN_LETTER = re.compile(r"\(([A-Da-d])\)")
    _LOOSE_LETTER = re.compile(r"\b([A-Da-d])\b")

    # inner text wrappers: \text{…}, \mathrm{…}, \textbf{…}, …
    _TEXT_WRAP = re.compile(
        r"\\(?:text|mathrm|mathbf|mathsf|textrm|textit|textbf)\s*\{\s*(.+?)\s*\}\s*$", re.I
    )
    _DIGIT_0_3   = re.compile(r"^[0-3]$")
    _PAREN_TOKEN = re.compile(r"^\(?\s*([A-Da-d0-3])\s*\)?$")

    # ---------------- label/index utils ----------------
    def idx_to_label(self, idx: Union[int, str]) -> str:
        try:
            i = int(idx)
        except Exception:
            return ""
        return self.LABELS[i] if 0 <= i < len(self.LABELS) else ""

    def label_to_idx(self, label: str) -> Optional[int]:
        if not label:
            return None
        pos = self.LABELS.find(label.upper())
        return pos if pos != -1 else None

    # --------------- unwrap helpers ----------------
    def _unwrap_token(self, t: str) -> str:
        """strip wrappers like \text{C}, (C) → C, keep '0-3' as is"""
        if t is None:
            return ""
        t = t.strip()
        # unnest text wrappers repeatedly
        prev = None
        while prev != t:
            prev = t
            m = self._TEXT_WRAP.fullmatch(t)
            if m:
                t = m.group(1).strip()
        # (A) / (2) → A / 2
        m = self._PAREN_TOKEN.fullmatch(t)
        if m:
            t = m.group(1).strip()
        return t
    
    # ---------------- gold ----------------
    def extract_gold(self, gold: Union[int, str], choices: List[str]) -> str:
        # 1) index 0..3 우선
        lab = self.idx_to_label(gold)
        if lab:
            return lab
        # 2) 문자 변형들
        s = str(gold).strip()
        s = self._unwrap_token(s)
        if self._DIGIT_0_3.fullmatch(s):
            return self.idx_to_label(s)
        if re.fullmatch(r"[A-Da-d]", s):
            return s.upper()
        # 3) \boxed{…} 안에 들어온 gold도 처리
        m = self._BOX_ANY.search(str(gold))
        if m:
            payload = self._unwrap_token(m.group(1))
            if self._DIGIT_0_3.fullmatch(payload):
                return self.idx_to_label(payload)
            if re.fullmatch(r"[A-Da-d]", payload):
                return payload.upper()
        return ""

    # ---------------- pred ----------------
    def extract_pred(self, text: str, choices: List[str]) -> str:
        if not text:
            return ""
        s = str(text)
        # 1) \boxed{ ... } → unwrap 후 해석 (마지막 박스 우선)
        boxes = list(self._BOX_ANY.finditer(s))
        if boxes:
            payload = self._unwrap_token(boxes[-1].group(1))
            if self._DIGIT_0_3.fullmatch(payload):
                lab = self.idx_to_label(payload)
                if lab: return lab
            if re.fullmatch(r"[A-Da-d]", payload):
                lab = payload.upper()
                if self.label_to_idx(lab) is not None:
                    return lab
        # 2) “Answer: …” / “Correct answer/choice is …”
        for rx in (self._ANS_LINE, self._CORR_LINE):
            ms = list(rx.finditer(s))
            if ms:
                token = self._unwrap_token(ms[-1].group(1))
                if self._DIGIT_0_3.fullmatch(token):
                    lab = self.idx_to_label(token)
                    if lab: return lab
                if re.fullmatch(r"[A-Da-d]", token):
                    lab = token.upper()
                    if self.label_to_idx(lab) is not None:
                        return lab
        # 3) anywhere: last '(A)'
        m = list(self._PAREN_LETTER.finditer(s))
        if m:
            lab = m[-1].group(1).upper()
            if self.label_to_idx(lab) is not None:
                return lab
        # 4) fallback: 마지막 두 줄에서 홀로 있는 A-D
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        for ln in reversed(lines[-2:]):
            ms = list(self._LOOSE_LETTER.finditer(ln))
            if ms:
                lab = ms[-1].group(1).upper()
                if self.label_to_idx(lab) is not None:
                    return lab
        return ""
    
    # ----------------------- grading -----------------------
    def grade(self, pred: Union[str, int], gold: Union[str, int], choices: List[str]) -> bool:
        p = self.extract_pred(pred if isinstance(pred, str) else str(pred), choices)
        g = self.extract_gold(gold, choices)
        return bool(p) and bool(g) and (p == g)


MMLUScorer = MmluScorer()

# # Test # 
# if __name__ == "__main__":
#     from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
#     import torch
#     from datasets import load_dataset
#     import random
#     from typing import List, Optional

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

#     # prompt generation
#     def _get_mmlu_fewshot_demos() -> list[tuple[str, str]]:
#         demos: list[tuple[str, str]] = []
#         # 1) Algebra
#         u1 = ("Simplify and write the result with a rational denominator:\n"
#             "$$\\sqrt{\\sqrt[3]{\\sqrt{\\frac{1}{729}}}}$$\n"
#             "Answer Choices: (A) \\frac{3\\sqrt{3}}{3} (B) \\frac{1}{3} (C) \\sqrt{3} (D) \\frac{\\sqrt{3}}{3}")
#         a1 = ("Step 1: Factoring $729=3^6$, we have $\\sqrt{\\frac{1}{729}}=\\frac{1}{27}$.\n\n"
#             "Step 2: $\\sqrt[3]{\\frac{1}{27}}=\\frac{1}{3}$. \n\n"
#             "Step 3: Finally $\\sqrt{\\frac{1}{3}}=\\frac{\\sqrt{3}}{3}$.\n"
#             "Answer: \\boxed{(D)}")
#         demos.append((u1, a1))
#         # 2) Biology
#         u2 = ("In animal cells, which of the following represents the most likely pathway that a secretory protein takes as it is synthesized in a cell?\n"
#             "Answer Choices: (A) Plasma membrane–Golgi apparatus–ribosome–secretory vesicle–rough ER "
#             "(B) Ribosome–Golgi apparatus–rough ER–secretory vesicle–plasma membrane "
#             "(C) Plasma membrane–Golgi apparatus–ribosome–secretory vesicle–rough ER "
#             "(D) Ribosome–rough ER–Golgi apparatus–secretory vesicle–plasma membrane" )
#         a2 = ("Step 1: Protein synthesis starts at the ribosome, often on the rough ER.\n"
#             "Step 2: Proteins then move to the Golgi apparatus, are modified and packaged into a vesicle, and the vesicle fuses with the plasma membrane to secrete the protein.\n"
#             "Answer: \\boxed{(D)}")
#         demos.append((u2, a2))
#         # 3) Physics
#         u3 = ( "A microwave oven is connected to an outlet, 120 V, and draws a current of 2 amps. At what rate is energy being used by the microwave oven?\n"
#             "Answer Choices: (A) 10 W (B) 30 W (C) 60 W (D) 240 W")
#         a3 = ( "Step 1: Rate of energy usage is power; in a resistive circuit, power is $P=VI$.\n"
#             "Step 2: So $120\\,\\text{V}\\times 2\\,\\text{A}=240\\,\\text{W}$.\n"
#             "Answer: \\boxed{(D)}")
#         demos.append((u3, a3))
#         # 4) Chemistry
#         u4 = ( "Which of the following is considered an acid anhydride?\n"
#             "Answer Choices: (A) HCl (B) H2SO3 (C) SO2 (D) Al(NO3)3")
#         a4 = ("Step 1: An acid anhydride is a compound that forms an acid upon reaction with water.\n"
#             "Step 2: SO2, sulfur dioxide, when combined with H2O, forms H2SO4 (sulfuric acid).\n"
#             "Answer: \\boxed{(C)}")
#         demos.append((u4, a4))
#         # 5) CS
#         u5 = ( 'What is the output of "abc"[::-1] in Python 3?\n'
#             "Answer Choices: (A) Error (B) abc (C) cba (D) c" )
#         a5 = ("Step 1: The slicing operator [::-1] reverses the string.\n"
#             'Step 2: So \"abc\" becomes \"cba\".\n'
#             "Answer: \\boxed{(C)}")
#         demos.append((u5, a5))
#         return demos
    
#     def _format_mmlu_test_block(question: str, choices: List[str]) -> str:
#         LABELS = list("ABCDEFGHIJ")
#         ac_line = "Answer Choices: " + " ".join(f"({LABELS[i]}) {c}" for i, c in enumerate(choices))
#         return f"{question.strip()}\n{ac_line}\n\n"

#     def _to_mmlu_policy_prompt(tokenizer, question: str, choices: List[str]) -> str:
#         TRAIN_SYS_PROMPT = (
#             "You are Qwen-Math, a meticulous math tutor. "
#             "Solve the given math problem step by step. "
#             "Use the EXACT format:\n"
#             "Step 1: <reasoning>\n\n"
#             "Step 2: <reasoning>\n\n"
#             "...\n\n"
#             "Answer: \\boxed{<final answer>}"
#         )
#         messages = [{"role": "system", "content": TRAIN_SYS_PROMPT}]
#         for u_text, a_text in _get_mmlu_fewshot_demos():
#             messages.append({"role": "user", "content": u_text})
#             messages.append({"role": "assistant", "content": a_text})
#         test_block = _format_mmlu_test_block(question, choices)
#         final_user = f"{test_block}\n\n"  # Please reason briefly and end with a single line: The answer is (X).
#         messages.append({"role": "user", "content": final_user})
#         return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

#     def format_test_block_with_letters(question: str, choices: List[str]) -> str:
#         LABELS = list("ABCDEFGHIJ")
#         ac_line = "Answer Choices: " + " ".join(
#             f"({LABELS[i]}) {c}" for i, c in enumerate(choices)
#         )
#         tail = ("""\nProvide your reasoning and conclude with with exactly one final line: "The answer is (A)." """)
#         return f"{tail}\n\n{question.strip()}\n{ac_line}\n"

#     def _get_fields(sample: dict) -> Tuple[str, List[str], str, str]:
#         # question
#         q = sample.get("question") or sample.get("prompt") or sample.get("input") or ""
#         # choices
#         choices = sample.get("choices") or sample.get("options") or []
#         # gold
#         gold = sample.get("answer")
#         # subject
#         subj = sample.get("subject") or sample.get("category") or sample.get("task") or "unknown"
#         return str(q), list(choices), str(gold), str(subj)

#     # ---------------- Dataset ----------------
#     ds = load_dataset("TIGER-Lab/MMLU-STEM", split="test")
#     ds = ds.select(range(20, 40))  # 20개만 예시

#     batch_questions, batch_choices, batch_gold, batch_subjects = [], [], [], []
#     for ex in ds:
#         q, choices, gold, subj = _get_fields(ex)
#         if not choices or not q:
#             continue
#         batch_questions.append(q)
#         batch_choices.append(choices)
#         batch_gold.append(gold)
#         batch_subjects.append(subj)

#     # ------------- Build prompts -------------
#     prompts = [
#         _to_mmlu_policy_prompt(tokenizer, q, ch)
#         for q, ch in zip(batch_questions, batch_choices)
#     ]
#     # ---------------- Generation ----------------
#     enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
#     enc = {k: v.to(model.device) for k, v in enc.items()}
#     prompt_lens = enc["attention_mask"].sum(dim=1).tolist()

#     with torch.no_grad():
#         gen_ids = model.generate(
#             **enc,
#             max_new_tokens=2048,
#             temperature=0.7,
#             top_p=0.8,
#             do_sample=True,  # temperature/top_p 사용 시 권장
#             eos_token_id=tokenizer.eos_token_id,
#             pad_token_id=tokenizer.pad_token_id,
#         )

#     gen_texts = []
#     for ids, plen in zip(gen_ids, prompt_lens):
#         gen_only = ids[plen:]  # 프롬프트 길이만큼 자르기
#         gen_texts.append(tokenizer.decode(gen_only, skip_special_tokens=True).strip())

#     # ---------------- Scoring ----------------
#     scorer = MmluScorer()
#     results = []
#     n_correct = 0
#     for txt, choices, gold, subj in zip(gen_texts, batch_choices, batch_gold, batch_subjects):
#         correct = scorer.grade(txt, gold, choices)
#         gold_label = scorer.extract_gold(gold, choices)
#         pred_label = scorer.extract_pred(txt, choices)
#         n_correct += int(bool(correct))
#         # txt = txt.split("assistant")[1]
#         results.append({
#             "subject": subj,
#             "gold": gold,
#             "gold_label": gold_label,
#             "pred_label": pred_label,
#             "pred_text": txt.strip(),
#             "correct": bool(correct),
#         })

#     # ------------- Report -------------
#     total = len(results)
#     acc = (n_correct / total) if total else 0.0
#     print(f"\nEvaluated {total} examples | Accuracy = {acc:.3f}\n")
#     for idx, r in enumerate(results):
#         gl = r['gold_label'] if r['gold_label'] is not None else "-"
#         pl = r['pred_label'] if r['pred_label'] is not None else "-"
#         print(f"[{idx}] subj={r['subject']:<20} gold={r['gold']} gold_label={gl} pred_label={pl:<2} correct={r['correct']}")
#         print("GEN:\n", r['pred_text'], "\n")

