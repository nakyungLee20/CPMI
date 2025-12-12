from base_hard_reward import BaseHardReward
from datasets import load_from_disk

def parse_transform(rec):
    parsed = rewarder.parse_math_shepherd_record(rec)
    return {
        "parsed_question": parsed["question"],
        "parsed_steps": parsed["steps"],
        "parsed_gold": parsed["gold_answer"],
        "parsed_correct_mask": parsed["correct_mask"],
    }

# 최초 1회만:
class Config:
        max_new_tokens: int = 2048
        num_rollouts: int = 8
        tensor_parallel_size: int = 1
        max_model_len: int = 4000
        gpu_mem_util: float = 0.4

cfg = Config()
model_name = "Qwen/Qwen3-4B-base"
rewarder = BaseHardReward(config=cfg, model_name=model_name, base_type="qval")

data_dir = "/home/leena/rs_prm/datasets/hard/ms_with_gold_80k_balanced_mixed"
ds = load_from_disk(data_dir)
ds_parsed = ds.map(parse_transform, num_proc=8)  # vLLM 미사용 단계이므로 멀티프로세스 OK
ds_parsed.save_to_disk("/home/leena/rs_prm/datasets/hard/ms_gold_80k_balanced_mixed_parsed")
