"""Analytic inventory against observed kernels: is every kernel the engine ran explained?

    python3 -m bench.census.diff [--observed results/census/observed_ops.csv]
                                 [--inventory results/census/analytic_ops.json]

GEMM kernels are matched to inventory GEMMs by their K and N from the recorded input
shapes, with the block label breaking the o_proj / out_proj tie. Every other kernel is
matched by name and block through the rule table below. Engine work outside the model
(input preparation, sampling, state alignment for prefix caching) is explained as such.
Whatever matches nothing is listed so the inventory can be amended, not silently kept.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

GEMM_OPS = {
    "vllm::dynamic_flashinfer_deepgemm_blockscale_gemm": "nk",
    "_C::cutlass_scaled_mm": "nk",
    "aten::mm": "kn",
    "aten::matmul": "kn",
    "aten::linear": "nk",
}
ENGINE = "engine work outside the model"
RULES: tuple[tuple[str, Optional[str], tuple[str, ...]], ...] = (
    (r"causal_conv1d", None, ("causal_conv1d",)),
    (r"fused_post_conv", None, ("post_conv_prep",)),
    (r"flashinfergdn|delta_rule|chunk_|fused_recurrent|solve_tril|fwd_h|fwd_o|cumsum|wy_", "gdn", ("gated_delta_rule",)),
    (r"layer_norm_fwd|rms_norm_gated", "gdn", ("gated_rmsnorm",)),
    (r"index_elementwise|index_put|gather|scatter", "gdn", ("state_gather",)),
    (r"elementwise|copy_|reduce_kernel|fill", "gdn", ("post_conv_prep",)),
    (r"mamba_align|mamba_fused", None, ("state_checkpoint",)),
    (r"flash::|fa3|flash_fwd|enable_sm90_or_later|prepare_varlen_num_blocks", None, ("attention",)),
    (r"prepare_rope_positions|prepare_", None, (ENGINE,)),
    (r"fused_qk_rmsnorm_rope_gate", None, ("q_norm", "k_norm", "rotary")),
    (r"reshape_and_cache|concat_and_cache|apply_write", None, ("kv_cache_write",)),
    (r"sigmoid|elementwise|copy_", "attention", ("output_gate",)),
    (r"act_and_mul|silu", None, ("silu_and_mul",)),
    (r"per_token_group_quant|quant", None, ("fp8_activation_quant",)),
    (r"copy_", "mlp", ("fp8_activation_quant",)),
    (r".*", "norm", ("input_layernorm", "post_attention_layernorm", "residual_add", "final_norm")),
    (r".*", "embedding", ("embed_tokens",)),
    (r".*", "rotary", ("rotary",)),
    (r".*", "kv_cache", ("kv_cache_write",)),
    (r".*", "logits", ("lm_head",)),
    (r".*", "sampler", (ENGINE,)),
    (r"prepare_|_post_update|num_accepted|fill|arange|index|copy_|pos_seq|block_table|slot_mapping", "other", (ENGINE,)),
)
RULES_COMPILED = [(re.compile(pattern, re.IGNORECASE), block, targets) for pattern, block, targets in RULES]


def gemm_dims(op: str, input_dims: str) -> Optional[tuple[int, int]]:
    order = GEMM_OPS.get(op)
    if not order or not input_dims:
        return None
    dims = json.loads(input_dims)
    if len(dims) < 2 or len(dims[0]) != 2 or len(dims[1]) != 2:
        return None
    a, b = dims[0], dims[1]
    if order == "nk":
        return b[1], b[0]
    return b[0], b[1]


def gemm_index(inventory: list[dict]) -> dict[tuple[int, int], list[dict]]:
    index: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for op in inventory:
        if op["kind"] == "gemm":
            match = re.match(r"\[M, (\d+)\] x \[(\d+), (\d+)\]", op["shape"])
            if match:
                index[(int(match.group(1)), int(match.group(3)))].append(op)
    return index


QUANT_KERNEL = re.compile(r"per_token_group_quant|scale_1x128|quant", re.IGNORECASE)


def explain(row: dict, gemms: dict[tuple[int, int], list[dict]]) -> tuple[str, ...]:
    if QUANT_KERNEL.search(row["kernel"]):
        return ("fp8_activation_quant",)
    dims = gemm_dims(row["op"], row["input_dims"])
    if dims:
        candidates = gemms.get(dims, [])
        if len(candidates) > 1:
            same_block = [c for c in candidates if c["block"] == row["block"]]
            candidates = same_block or candidates
        return (candidates[0]["name"],) if candidates else ()
    for pattern, block, targets in RULES_COMPILED:
        if block is not None and block != row["block"]:
            continue
        if pattern.search(row["kernel"]) or pattern.search(row["op"] or ""):
            return targets
    return ()


def diff(observed: list[dict], inventory: list[dict]) -> dict:
    gemms = gemm_index(inventory)
    explained: dict[str, dict] = defaultdict(lambda: {"kernels": set(), "per_step": defaultdict(int), "total_us": defaultdict(float)})
    unexplained = []
    steps = defaultdict(lambda: defaultdict(float))
    for row in observed:
        targets = explain(row, gemms)
        instances = max(1, int(row.get("step_instances") or 1))
        count, total = int(row["count"]) / instances, float(row["total_us"]) / instances
        if not targets:
            unexplained.append({k: row[k] for k in ("mode", "step", "block", "kernel", "op", "input_dims", "count", "total_us")})
            steps[(row["mode"], row["step"])]["unexplained"] += total
            continue
        steps[(row["mode"], row["step"])]["explained"] += total
        for target in targets:
            entry = explained[target]
            entry["kernels"].add(row["kernel"].split("(")[0][:70])
            entry["per_step"][(row["mode"], row["step"])] += count / len(targets)
            entry["total_us"][(row["mode"], row["step"])] += total / len(targets)
    inventory_names = {op["name"]: op for op in inventory}
    ops = []
    for name in list(inventory_names) + [n for n in explained if n not in inventory_names]:
        entry = explained.get(name)
        ops.append({
            "op": name,
            "block": inventory_names.get(name, {}).get("block", "engine" if name == ENGINE else "added"),
            "expected_per_step": inventory_names.get(name, {}).get("per_step"),
            "in_inventory": name in inventory_names,
            "observed": entry is not None,
            "kernels": sorted(entry["kernels"]) if entry else [],
            "observed_per_step": {f"{m}: {s}": round(c, 1) for (m, s), c in sorted(entry["per_step"].items())} if entry else {},
            "total_us_per_step": {f"{m}: {s}": round(t, 1) for (m, s), t in sorted(entry["total_us"].items())} if entry else {},
        })
    coverage = {f"{m}: {s}": {"explained_us": round(v["explained"], 1), "unexplained_us": round(v["unexplained"], 1),
                              "explained_share": round(v["explained"] / (v["explained"] + v["unexplained"]), 4) if (v["explained"] + v["unexplained"]) else 1.0}
                for (m, s), v in sorted(steps.items())}
    return {"ops": ops, "unexplained": sorted(unexplained, key=lambda r: -float(r["total_us"])), "coverage": coverage}


def markdown(result: dict) -> str:
    lines = ["| op | block | in inventory | expected per step | observed per step | kernels |", "|---|---|---|---:|---|---|"]
    for op in result["ops"]:
        observed = "; ".join(f"{k}: {v:g}" for k, v in op["observed_per_step"].items()) or "not observed"
        kernels = "<br>".join(op["kernels"][:6]) + ("<br>..." if len(op["kernels"]) > 6 else "")
        lines.append(f"| {op['op']} | {op['block']} | {'yes' if op['in_inventory'] else 'no'} | {op['expected_per_step'] if op['expected_per_step'] is not None else ''} | {observed} | {kernels} |")
    lines += ["", "| step | explained us | unexplained us | explained share |", "|---|---:|---:|---:|"]
    for step, cov in result["coverage"].items():
        lines.append(f"| {step} | {cov['explained_us']:.0f} | {cov['unexplained_us']:.0f} | {100 * cov['explained_share']:.1f}% |")
    if result["unexplained"]:
        lines += ["", "Unexplained kernels:", ""]
        for row in result["unexplained"][:40]:
            lines.append(f"- {row['mode']} {row['step']} {row['block']}: `{row['kernel'][:80]}` op `{row['op']}` dims `{row['input_dims'][:60]}` x{row['count']} {float(row['total_us']):.0f} us")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Diff the analytic inventory against observed kernels")
    parser.add_argument("--observed", default="results/census/observed_ops.csv")
    parser.add_argument("--inventory", default="results/census/analytic_ops.json")
    parser.add_argument("--out", default="results/census/census_diff.json")
    parser.add_argument("--table", default="results/census/census_diff.md")
    args = parser.parse_args(argv)

    with Path(args.observed).open(newline="", encoding="utf-8") as handle:
        observed = list(csv.DictReader(handle))
    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))["ops"]
    result = diff(observed, inventory)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    Path(args.table).write_text(markdown(result) + "\n", encoding="utf-8", newline="\n")
    missing = [op["op"] for op in result["ops"] if op["in_inventory"] and op["expected_per_step"] and not op["observed"]]
    print(f"{len(result['unexplained'])} unexplained kernel rows; inventory ops not observed: {missing or 'none'}")
    for step, cov in result["coverage"].items():
        print(f"  {step}: {100 * cov['explained_share']:.1f}% of kernel time explained")
    return 0 if not result["unexplained"] else 1


if __name__ == "__main__":
    sys.exit(main())
