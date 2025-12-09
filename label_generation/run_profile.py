import json

# ======================= Computation Cost Class =======================
class RunProfiler:
    def __init__(self):
        self.records: list[dict] = []

    def log(self, *, tag: str, num_prompts: int= 0, n: int= 1, wall_s: float, gen_tokens: int | None = None, peak_mem_gb: float | None = None, **meta):
        rec = {
            "tag": tag, "num_prompts": num_prompts, "n": n,
            "wall_s": wall_s, "gen_tokens": gen_tokens, "peak_mem_gb": peak_mem_gb
        }
        rec.update(meta)  # sample_idx, dataset, base_type 등
        self.records.append(rec)

    def summary(self, header: str = "") -> str:
        # 태그별 합계
        by_tag = {}
        for r in self.records:
            t = r["tag"]
            d = by_tag.setdefault(t, {"calls":0,"prompts":0,"samples":0,"wall":0.0,"tokens":0,"mem":0.0})
            d["calls"] += 1
            d["prompts"] += r["num_prompts"]
            d["samples"] += r["num_prompts"] * r["n"]
            d["wall"] += r["wall_s"]
            d["tokens"] += (r.get("gen_tokens") or 0)
            d["mem"] = max(d["mem"], r.get("peak_mem_gb") or 0.0)
        lines = [f"[PROFILE]{' ' + header if header else ''}"]
        for tag, d in by_tag.items():
            tps = (d["tokens"]/d["wall"]) if d["wall"]>0 else 0.0
            lines.append(
                f" - {tag:14s} | calls={d['calls']:3d} | prompts={d['prompts']:5d} | "
                f"samples≈{d['samples']:6d} | wall={d['wall']:.2f}s | "
                f"gen_tokens≈{d['tokens']:,} | tok/s≈{tps:.1f} | peak_mem≈{d['mem']:.2f}GB"
            )
        return "\n".join(lines)

    def summary_filtered(self, header: str = "", *, tag_prefix: str | None = None, tag_contains: str | None = None, **filters) -> str:
        """Summarize only records whose fields match all provided filters.
        - filters: key=value pairs to match in each record (e.g., dataset="gsm8k").
        - tag_prefix: only include tags that start with this prefix (optional).
        - tag_contains: only include tags that contain this substring (optional).
        """
        by_tag = {}
        for r in self.records:
            ok = True
            for k, v in filters.items():
                if r.get(k) != v:
                    ok = False
                    break
            if not ok:
                continue
            t = r["tag"]
            if tag_prefix and not str(t).startswith(tag_prefix):
                continue
            if tag_contains and (tag_contains not in str(t)):
                continue
            d = by_tag.setdefault(t, {"calls":0,"prompts":0,"samples":0,"wall":0.0,"tokens":0,"mem":0.0})
            d["calls"] += 1
            d["prompts"] += r.get("num_prompts", 0)
            d["samples"] += r.get("num_prompts", 0) * r.get("n", 0)
            d["wall"] += r.get("wall_s", 0.0)
            d["tokens"] += (r.get("gen_tokens") or 0)
            d["mem"] = max(d["mem"], r.get("peak_mem_gb") or 0.0)
        lines = [f"[PROFILE]{' ' + header if header else ''}"]
        for tag, d in by_tag.items():
            tps = (d["tokens"]/d["wall"]) if d["wall"]>0 else 0.0
            lines.append(
                f" - {tag:14s} | calls={d['calls']:3d} | prompts={d['prompts']:5d} | "
                f"samples≈{d['samples']:6d} | wall={d['wall']:.2f}s | "
                f"gen_tokens≈{d['tokens']:,} | tok/s≈{tps:.1f} | peak_mem≈{d['mem']:.2f}GB"
            )
        return "\n".join(lines)
    
    def aggregate(self) -> dict:
        by_tag = {}
        for r in self.records:
            t = r["tag"]
            d = by_tag.setdefault(t, {"calls":0,"prompts":0,"samples":0,"wall":0.0,"tokens":0,"peak_mem_gb":0.0})
            d["calls"] += 1
            d["prompts"] += r["num_prompts"]
            d["samples"] += r["num_prompts"] * r["n"]
            d["wall"] += r["wall_s"]
            d["tokens"] += (r.get("gen_tokens") or 0)
            d["peak_mem_gb"] = max(d["peak_mem_gb"], r.get("peak_mem_gb") or 0.0)
        return {"by_tag": by_tag, "total_calls": len(self.records)}
    
    def dump_summary_text(self, filepath: str, header: str = ""):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.summary(header) + "\n")

    def dump_summary_json(self, filepath: str, header: str = ""):
        payload = {"header": header, "aggregation": self.aggregate(), "raw_records": self.records}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

