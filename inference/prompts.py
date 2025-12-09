from typing import Optional, List, Dict, Any, Tuple, Type, Union

# Generate Solution Helpers (Policy Sampling Trajectory) #
TRAIN_SYS_PROMPT = (
    "You are a careful math solver. "
    "Solve the given math problem step by step. Keep each step concise. "
    "Use the EXACT format:\n"
    "Step 1: <reasoning>\n\n"
    "Step 2: <reasoning>\n\n"
    "...\n\n"
    "The answer is: \\boxed{<final answer>}"
)

##############################################################################
def build_prompt(question_or_items: Union[str, List[Dict[str, Any]]], tokenizer, dataset_name: str) -> Union[str, List[str]]:
    if dataset_name == "mmlu":
        items: List[Dict[str, Any]] = question_or_items  # type: ignore[assignment]
        return [_to_mmlu_policy_prompt(tokenizer, ex["question"], list(ex.get("choices", [])))
                for ex in items]

    # Non-MMLU: use your training-time math prompt (single or batch)
    def _one(q: str) -> str:
        messages = [
                {"role": "system", "content": TRAIN_SYS_PROMPT},
                {"role": "user", "content": f"Problem: {q.strip()}\nSolution: Let's think step by step.\n"},
            ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if isinstance(question_or_items, str):
        return _one(question_or_items)
    else:
        return [_one(ex["question"]) for ex in question_or_items]

def build_prm_prompt(tokenizer, question: str, steps: List[str], rw_token: str = "<RW>", ds_name: str = "", model_type: str = "", choices: Optional[List[str]] = None) -> str:
    if ds_name == "mmlu":
        return _to_mmlu_prm_prompt(tokenizer, question, choices or [], steps, rw_token, model_type=model_type)
    
    assistant_text = join_steps_with_rw(steps, rw_token=rw_token, step_sep="\n\n")
    if model_type.startswith("prompt_"):
        template = "You are a careful math solver. Follow the steps methodically. Keep each step concise.\nAt the end, output exactly one line in the format:\nThe answer is: <final answer>\n\n"
        messages = [
            {"role": "user", "content": template + f"Problem: {question}\nSolution: Let's think step by step.\n"},
            {"role": "assistant", "content": assistant_text},
        ]
    elif model_type.startswith("contr_"):
        messages = [
            {"role": "user", "content": f"Problem: {question}\nSolution: Let's think step by step.\n"},
            {"role": "assistant", "content": assistant_text},
        ]
    else:
        template = "You are a careful math solver. Follow the steps methodically. Keep each step concise.\nAt the end, output exactly one line in the format:\nThe answer is: <final answer>\n\n"
        messages = [
            {"role": "user", "content": f"Problem: {question}\nSolution: Let's think step by step.\n"},
            {"role": "assistant", "content": assistant_text},
        ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

##############################################################################
def join_steps_with_rw(steps: List[str], rw_token: str = "<RW>", step_sep: str = "\n\n") -> str:
    # "step1 + <RW> + step_sep + step2 + <RW> + ... + <RW>" 형태
    cleaned = [str(s).strip() for s in steps if s is not None and str(s).strip()]
    if not cleaned:
        return rw_token  # 최소 1개 <RW>는 남기도록
    return (f"{step_sep}{rw_token}{step_sep}").join(cleaned) + f"{step_sep}{rw_token}"

# MMLU-specific prompt builders (few-shot prefix + per-example block)
def _format_mmlu_test_block(question: str, choices: List[str]) -> str:
    LABELS = list("ABCDEFGHIJ")
    ac_line = "Answer Choices: " + " ".join(f"({LABELS[i]}) {c}" for i, c in enumerate(choices))
    return f"{question.strip()}\n{ac_line}\n"

def _to_mmlu_policy_prompt(tokenizer, question: str, choices: List[str]) -> str:
    test_block = _format_mmlu_test_block(question, choices)
    final_user = f"{test_block}\nSolution: Let's think step by step.\n\n"
    messages = [
            {"role": "system", "content": TRAIN_SYS_PROMPT},
            {"role": "user", "content": final_user}
        ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def _to_mmlu_prm_prompt_messages(question: str, choices: List[str], model_type: str) -> List[Dict[str, str]]:
    if model_type.startswith("prompt_"): 
        template = "You are a careful math solver. Follow the steps methodically. Keep each step concise.\nAt the end, output exactly one line in the format:\nThe answer is: <final answer>\n"
        test_block = _format_mmlu_test_block(question, choices)
        final_user = f"{template}\n{test_block}\nSolution: Let's think step by step.\n"
        messages = [{"role": "user", "content": final_user}]
    elif model_type.startswith("contr_"):
        test_block = _format_mmlu_test_block(question, choices)
        final_user = f"{test_block}\nSolution: Let's think step by step.\n"
        messages = [{"role": "user", "content": final_user}]
    else: # Default: TRAIN_SYS_PROMPT
        template = "You are a careful math solver. Follow the steps methodically. Keep each step concise.\nAt the end, output exactly one line in the format:\nThe answer is: <final answer>\n"
        test_block = _format_mmlu_test_block(question, choices)
        final_user = f"{template}\n{test_block}\nSolution: Let's think step by step.\n"
        messages= [{"role": "user", "content": final_user}]
    return messages

def _to_mmlu_prm_prompt(tokenizer, question: str, choices: List[str], steps: List[str], rw_token: str, model_type: str) -> str:
    messages = _to_mmlu_prm_prompt_messages(question, choices, model_type=model_type)
    messages.append({"role": "assistant", "content": join_steps_with_rw(steps, rw_token=rw_token, step_sep="\n\n")})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

def _get_mmlu_fewshot_demos() -> list[tuple[str, str]]:
    demos: list[tuple[str, str]] = []
    # 1) Algebra
    u1 = ("Simplify and write the result with a rational denominator:\n"
        "$$\\sqrt{\\sqrt[3]{\\sqrt{\\frac{1}{729}}}}$$\n"
        "Answer Choices: (A) \\frac{3\\sqrt{3}}{3} (B) \\frac{1}{3} (C) \\sqrt{3} (D) \\frac{\\sqrt{3}}{3}")
    a1 = ("Step 1: Factoring $729=3^6$, we have $\\sqrt{\\frac{1}{729}}=\\frac{1}{27}$.\n\n"
        "Step 2: $\\sqrt[3]{\\frac{1}{27}}=\\frac{1}{3}$. \n\n"
        "Step 3: Finally $\\sqrt{\\frac{1}{3}}=\\frac{\\sqrt{3}}{3}$.\n"
        "Answer: \\boxed{(D)}")
    demos.append((u1, a1))
    # 2) Biology
    u2 = ("In animal cells, which of the following represents the most likely pathway that a secretory protein takes as it is synthesized in a cell?\n"
        "Answer Choices: (A) Plasma membrane–Golgi apparatus–ribosome–secretory vesicle–rough ER "
        "(B) Ribosome–Golgi apparatus–rough ER–secretory vesicle–plasma membrane "
        "(C) Plasma membrane–Golgi apparatus–ribosome–secretory vesicle–rough ER "
        "(D) Ribosome–rough ER–Golgi apparatus–secretory vesicle–plasma membrane" )
    a2 = ("Step 1: Protein synthesis starts at the ribosome, often on the rough ER.\n"
        "Step 2: Proteins then move to the Golgi apparatus, are modified and packaged into a vesicle, and the vesicle fuses with the plasma membrane to secrete the protein.\n"
        "Answer: \\boxed{(D)}")
    demos.append((u2, a2))
    # 3) Physics
    u3 = ( "A microwave oven is connected to an outlet, 120 V, and draws a current of 2 amps. At what rate is energy being used by the microwave oven?\n"
        "Answer Choices: (A) 10 W (B) 30 W (C) 60 W (D) 240 W")
    a3 = ( "Step 1: Rate of energy usage is power; in a resistive circuit, power is $P=VI$.\n"
        "Step 2: So $120\\,\\text{V}\\times 2\\,\\text{A}=240\\,\\text{W}$.\n"
        "Answer: \\boxed{(D)}")
    demos.append((u3, a3))
    # 4) Chemistry
    u4 = ( "Which of the following is considered an acid anhydride?\n"
        "Answer Choices: (A) HCl (B) H2SO3 (C) SO2 (D) Al(NO3)3")
    a4 = ("Step 1: An acid anhydride is a compound that forms an acid upon reaction with water.\n"
        "Step 2: SO2, sulfur dioxide, when combined with H2O, forms H2SO4 (sulfuric acid).\n"
        "Answer: \\boxed{(C)}")
    demos.append((u4, a4))
    # 5) CS
    u5 = ( 'What is the output of "abc"[::-1] in Python 3?\n'
        "Answer Choices: (A) Error (B) abc (C) cba (D) c" )
    a5 = ("Step 1: The slicing operator [::-1] reverses the string.\n"
        'Step 2: So \"abc\" becomes \"cba\".\n'
        "Answer: \\boxed{(C)}")
    demos.append((u5, a5))
    return demos
