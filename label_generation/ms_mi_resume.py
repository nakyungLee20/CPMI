import json, os, math, argparse, time, re, glob, signal
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
# from mi_hard_reward_batch import MIHardRewardBatch
from cpmi_hard_reward_batch import CPMIHardRewardBatch

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
    if start >= end:
        return ds.select([]), (start, end, N)
    return ds.select(range(start, end)), (start, end, N)

def count_valid_jsonl_lines(path: str) -> int:
    n = 0
    try:
        with open(path, "r", encoding="utf-8") as fin:
            for ln in fin:
                s = ln.strip()
                if not s:
                    continue
                try:
                    json.loads(s)
                    n += 1
                except json.JSONDecodeError:
                    break  # 마지막 라인이 깨졌으면 거기까지만 유효
    except FileNotFoundError:
        return 0
    return n

def safe_flush(f, retries: int = 3, base: float = 0.05):
    for k in range(retries):
        try:
            f.flush()
            return
        except OSError:
            if k + 1 >= retries:
                raise
            time.sleep(base * (2 ** k))

FROM_RE = re.compile(r"\.from(\d+)\.")  # ...from{OFFSET}.TIMESTAMP.jsonl
def compute_resume_offset(out_dir: str, shard_tag: str, shard_len: int, filename_prefix: str) -> tuple[int, list[tuple[str,int,int]]]:
    paths = glob.glob(os.path.join(out_dir, f"{filename_prefix}*.jsonl"))
    max_covered = 0
    details = []
    for p in paths:
        fname = os.path.basename(p)
        m = FROM_RE.search(fname)
        start = int(m.group(1)) if m else 0
        n_valid = count_valid_jsonl_lines(p)
        covered = min(shard_len, start + n_valid)
        details.append((p, start, n_valid))
        if covered > max_covered:
            max_covered = covered
    return max_covered, details

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_shards", type=int, default=4)         # <<< default 3-way split
    ap.add_argument("--shard_index", type=int, default=0)        # 0,1,2
    return ap.parse_args()

def main():
    args = parse_args()
    model_name = "Qwen/Qwen3-4B-base"  # "Qwen/Qwen2.5-Math-7B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    mi = CPMIHardRewardBatch(model=model, tokenizer=tokenizer)
    print(f"[INFO] Loaded model={model_name}")

    data_dir = "/home/leena/rs_prm/datasets/hard_80k/ms_gold_80k_balanced_mixed_parsed"
    ds = load_from_disk(data_dir)
    print(f"Dataset loaded: {len(ds)} rows")
    ds_shard, (start, end, N) = slice_dataset_contiguous(ds, args.num_shards, args.shard_index)
    shard_len = len(ds_shard)
    print(f"[INFO] Using shard {args.shard_index}/{args.num_shards} -> indices [{start}:{end}) size={shard_len}/{N}")

    # --- outputs per shard ---
    out_dir = "/home/leena/rs_prm/datasets/hard_80k/cpmi"
    os.makedirs(out_dir, exist_ok=True)
    shard_tag = f"s{args.shard_index}of{args.num_shards}"
    prefix = f"ms_cpmi_qw3_4b_{shard_tag}"

    # --- 기존 산출물 스캔해서 resume offset 결정 ---
    already, scanned = compute_resume_offset(out_dir, shard_tag, shard_len, filename_prefix=prefix)
    print("[RESUME] scanned outputs for this shard:")
    for p, s, n in sorted(scanned):
        covered = min(shard_len, s + n)
        print(f"  - {os.path.basename(p)} | start={s} | valid_lines={n} | covered≈{covered}")
    if already >= shard_len:
        print(f"[RESUME] nothing left to do: already={already} >= shard_len={shard_len}")
        return
    print(f"[RESUME] will start from index {already} (of {shard_len})")

    # --- shard 내에서 offset 이후만 선택 ---
    ds_resume = ds_shard.select(range(already, shard_len))
    print(f"[RESUME] will process indices [{already}:{shard_len}) within this shard -> {len(ds_resume)} rows")

    # new segment file (final path; per-line flush as requested)
    ts = time.strftime("%Y%m%d-%H%M%S")
    output_file = os.path.join(out_dir, f"{prefix}.from{already}.{ts}.jsonl")
    profile_txt = output_file + "_profile.txt"
    print(f"[INFO] writing to: {output_file}")

    # graceful flush on signals
    fout_holder = {"fp": None}
    def _graceful(_sig, _frm):
        try:
            fp = fout_holder.get("fp")
            if fp and not fp.closed:
                fp.flush()
            print(f"[SIGNAL] flushed on {_sig}", flush=True)
        except Exception as e:
            print(f"[SIGNAL] flush failed: {e}", flush=True)
    try:
        signal.signal(signal.SIGTERM, _graceful)
        signal.signal(signal.SIGINT, _graceful)
    except Exception:
        pass

    # --- run ---
    t0 = time.perf_counter()
    with open(output_file, "w", encoding="utf-8", buffering=1) as f:
        fout_holder["fp"] = f
        try:
            f.reconfigure(line_buffering=True, write_through=True)
        except Exception:
            pass

        for i, entry in enumerate(mi.mi_labelling(ds=ds_resume, ds_task_tag="task"), start=already):
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            try:
                f.flush()
            except Exception:
                print(f"[INFO] Flush Excepted at {i}-th data.")
                pass

    print(f"[DONE] Data saved to {output_file} (elapsed: {time.perf_counter()-t0:.1f}s)")
    header = f"MathShepherd/cpmi :: model={model_name} :: shard={shard_tag} :: rows={len(ds_resume)} :: from={already}"
    mi.prof.dump_summary_text(profile_txt, header=header)
    print(f"[DONE] Profile (text) -> {profile_txt}")


if __name__ == "__main__":
    main() 
