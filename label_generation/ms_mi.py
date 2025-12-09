import json, os, math, argparse, time
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from mi_hard_reward_batch import MIHardRewardBatch

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def jsonl_to_json(jsonl_path, json_path):
    data = read_jsonl(jsonl_path)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Converted {jsonl_path} to {json_path}")

def slice_dataset_contiguous(ds, num_shards: int, shard_index: int):
    assert 0 <= shard_index < num_shards, f"shard_index must be in [0, {num_shards-1}], got {shard_index}"
    N = len(ds)
    per = math.ceil(N / num_shards)
    start = shard_index * per
    end = min(start + per, N)
    if start >= end:  # in case of tiny remainder
        return ds.select([])  # empty
    # Contiguous slice via indices (keeps order)
    return ds.select(range(start, end)), (start, end, N)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_shards", type=int, default=4)         # <<< default 3-way split
    ap.add_argument("--shard_index", type=int, default=0)        # 0,1,2
    ap.add_argument("--batch_size", type=int, default=16)        # used inside MI class batch calls if needed
    return ap.parse_args()

def main():
    args = parse_args()
    model_name = "Qwen/Qwen3-4B-base"  # "Qwen/Qwen2.5-Math-7B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    mi = MIHardRewardBatch(model=model, tokenizer=tokenizer)
    print(f"[INFO] Loaded model={model_name}")

    data_dir = "/home/leena/rs_prm/datasets/hard_80k/ms_gold_80k_balanced_mixed_parsed"
    ds = load_from_disk(data_dir)
    # idx = [20000,40000]
    # ds = ds.select(idx)
    print(f"Dataset loaded: {len(ds)} rows")

    # --- shard (contiguous) ---
    ds_shard, (start, end, N) = slice_dataset_contiguous(ds, args.num_shards, args.shard_index)
    print(f"[INFO] Using shard {args.shard_index}/{args.num_shards} -> indices [{start}:{end}) size={len(ds_shard)}/{N}")

    # --- outputs per shard ---
    out_dir = "/home/leena/rs_prm/datasets/hard_80k/mi_mean"
    os.makedirs(out_dir, exist_ok=True)
    shard_tag = f"s{args.shard_index}of{args.num_shards}"
    output_file = os.path.join(out_dir, f"ms_mi_qw3_4b_{shard_tag}.jsonl")
    profile_txt = os.path.join(out_dir, f"ms_mi_qw3_4b_{shard_tag}_profile.txt")

    # --- run ---
    t0 = time.perf_counter()
    with open(output_file, "w", encoding="utf-8") as f:
        for i, entry in enumerate(mi.mi_labelling(ds=ds_shard, ds_task_tag="task")):
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()

    print(f"[DONE] Data saved to {output_file} (elapsed: {time.perf_counter()-t0:.1f}s)")
    header = f"MathShepherd/mi_mean :: model={model_name} :: shard={shard_tag} :: rows={len(ds_shard)}"
    mi.prof.dump_summary_text(profile_txt, header=header)
    print(f"[DONE] Profile (text) -> {profile_txt}")


if __name__ == "__main__":
    main() 
