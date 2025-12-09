import os, sys, json, time, argparse, signal, faulthandler
from datetime import datetime
import torch
from datasets import load_from_disk
from base_hard_reward import BaseHardReward
import multiprocessing as mp

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
                    break
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

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_shards", type=int, default=5)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--contiguous", action="store_true", help="use contiguous slice instead of modulo sharding")
    p.add_argument("--base_type", type=str, default="pav")  # "qval" | "pav"
    p.add_argument("--heartbeat_sec", type=int, default=600)
    p.add_argument("--dump_sec",      type=int, default=2100)
    return p.parse_args()

def main():
    if os.environ.get("PROCESS_GRADER", "0") == "1":
        import multiprocessing as mp
        try:
            mp.set_start_method("forkserver")  # 리눅스 권장
        except RuntimeError:
            pass
    
    args = parse_args()
    class Config:
        max_new_tokens: int = 2048
        num_rollouts: int = 8
        tensor_parallel_size: int = 1
        max_model_len: int = 8000
        gpu_mem_util: float = 0.60
    cfg = Config()

    model_name = "Qwen/Qwen3-4B-base"
    rewarder = BaseHardReward(config=cfg, model_name=model_name, base_type=args.base_type)
    print(f"vLLM model loaded: {model_name}")
    print(f"Finish loading model and tokenizer with {args.base_type}!")

    # faulthandler: 주기적으로 전체 스택 덤프 (멈춤 구분용)
    faulthandler.enable(file=sys.stdout)
    faulthandler.dump_traceback_later(args.dump_sec, repeat=True, file=sys.stdout)

    data_dir = "/home/leena/rs_prm/datasets/hard_80k/ms_gold_80k_balanced_mixed_parsed"
    ds = load_from_disk(data_dir)
    N = len(ds)
    if args.num_shards > 1:
        if args.contiguous:
            per = (N + args.num_shards - 1) // args.num_shards
            lo = args.shard_index * per
            hi = min(N, lo + per)
            ds = ds.select(range(lo, hi))
            print(f"[SHARD] contiguous {args.shard_index+1}/{args.num_shards} -> rows={len(ds)} (range {lo}:{hi})")
        else:
            ds = ds.shard(num_shards=args.num_shards, index=args.shard_index)  # modulo
            print(f"[SHARD] modulo {args.shard_index+1}/{args.num_shards} -> rows={len(ds)}")

    out_dir = f"/home/leena/rs_prm/datasets/hard_80k/{args.base_type}"
    os.makedirs(out_dir, exist_ok=True)
    shard_tag = f"s{args.shard_index:02d}of{args.num_shards:02d}"
    output_file = os.path.join(out_dir, f"ms_{args.base_type}_qw3_4b_{shard_tag}.jsonl")
    profile_txt = output_file + "_profile.txt"

    # --- resume ---
    already = count_valid_jsonl_lines(output_file)
    if already > 0:
        print(f"[RESUME] valid lines in {output_file}: {already}")
    else:
        print(f"[RESUME] start fresh at {output_file}")

    t0 = time.time()
    last_hb = time.time()
    written = 0
    fout_holder = {"fp": None} 

    # 시그널 시 마지막까지 flush만 보장 (fsync는 안 함)
    def _graceful_flush(signum, frame):
        try:
            fp = fout_holder.get("fp")
            if fp and not fp.closed:
                safe_flush(fp)
            print(f"[SIGNAL] {signum} -> flushed", flush=True)
        except Exception as e:
            print(f"[SIGNAL] flush failed: {e}", flush=True)

    signal.signal(signal.SIGTERM, _graceful_flush)
    signal.signal(signal.SIGINT, _graceful_flush)

    try:
        # 최종 파일에 '바로' 이어쓰기
        fout = open(output_file, "a", encoding="utf-8", newline="\n", buffering=1)  # <= buffering=1(라인버퍼), newline 고정
        fout_holder["fp"] = fout

        # 줄바꿈마다 자동 flush 시도 (가능하면 사용)
        try:
            fout.reconfigure(line_buffering=True, write_through=True)  # <= write_through 추가
            line_buffering_active = True
        except Exception:
            line_buffering_active = False
        print(f"[INFO] line_buffering_active={line_buffering_active}")

        for i, entry in enumerate(rewarder.mc_labelling(ds=ds, base_type=args.base_type)):
            # resume skip
            if i < already:
                # 가끔 현황 찍어주기
                if (time.time() - last_hb) >= args.heartbeat_sec:
                    print(f"[HEARTBEAT] skipping i={i} (resume) | elapsed={time.time()-t0:.1f}s")
                    last_hb = time.time()
                continue

            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1

            if not line_buffering_active:
                safe_flush(fout)
            if (time.time() - last_hb) >= args.heartbeat_sec:
                print(f"[HEARTBEAT] shard={shard_tag} i={i} new_lines={written} | elapsed={time.time()-t0:.1f}s")
                last_hb = time.time()

        # 종료 직전 한 번 더 flush
        safe_flush(fout)
        fout.close()
        fout_holder["fp"] = None

        total = already + written
        print(f"[DONE] Saved to {output_file} | new_rows={written} | total≈{total} | {time.time()-t0:.1f}s")
        header = (f"MathShepherd/{args.base_type} :: model={model_name}, rollouts={cfg.num_rollouts}, max_new_tokens={cfg.max_new_tokens}")
        rewarder.prof.dump_summary_text(profile_txt, header=header)
        
    finally:
        # vLLM 종료 보장
        try:
            if getattr(rewarder, "llm", None):
                rewarder.llm.shutdown()
        except Exception as e:
            print(f"[WARN] llm.shutdown() failed: {e}")
        try:
            if getattr(rewarder, "_prover", None):
                rewarder._prover.shutdown()
        except Exception as e:
            print(f"[WARN] prover.shutdown() failed: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
