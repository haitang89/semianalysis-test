import csv
import gzip
import json
from pathlib import Path

from bench.census import trace_parser as tp


def event(cat, name, ts, dur, pid=1, tid=1, **args):
    return {"ph": "X", "cat": cat, "name": name, "pid": pid, "tid": tid, "ts": ts, "dur": dur, "args": args}


def kernel(name, ts, dur, ext, graph=0, grid=(1, 1, 1), block=(128, 1, 1)):
    return event("kernel", name, ts, dur, pid=0, tid=7, **{"External id": ext, "correlation": ext, "grid": list(grid),
                                                          "block": list(block), "graph id": graph})


def op(name, ts, dur, ext, dims, types, tid=1):
    return event("cpu_op", name, ts, dur, tid=tid, **{"External id": ext, "Input Dims": dims, "Input type": types})


def frame(name, ts, dur, tid=1):
    return event("python_function", name, ts, dur, tid=tid)


def trace():
    events = [
        event("user_annotation", "execute_context_1(784)_generation_0(0)", 0, 1000),
        event("gpu_user_annotation", "execute_context_1(784)_generation_0(0)", 5, 1000, pid=0, tid=7),
        event("gpu_user_annotation", "execute_context_0(0)_generation_2(2)", 2000, 500, pid=0, tid=7),
        frame("vllm/model_executor/models/qwen3_next.py(546): forward", 0, 900),
        frame("vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py(828): forward", 10, 300),
        frame("vllm/model_executor/layers/linear.py(593): forward", 20, 50),
        op("vllm::dynamic_flashinfer_deepgemm_blockscale_gemm", 30, 20, 11, [[784, 5120], [16384, 5120]], ["c10::Float8_e4m3fn", "c10::Float8_e4m3fn"]),
        op("vllm::qwen_gdn_attention_core_fused_norm", 100, 100, 12, [[784, 16384], [784, 96]], ["c10::BFloat16", "c10::BFloat16"]),
        frame("vllm/model_executor/models/qwen3_next.py(446): forward", 400, 200),
        frame("vllm/model_executor/layers/linear.py(593): forward", 410, 50),
        op("vllm::dynamic_flashinfer_deepgemm_blockscale_gemm", 420, 20, 13, [[784, 5120], [14336, 5120]], ["c10::Float8_e4m3fn", "c10::Float8_e4m3fn"]),
        frame("vllm/v1/sample/sampler.py(50): forward", 1200, 100, tid=2),
        op("aten::softmax", 1210, 10, 14, [[1, 248320]], ["float"], tid=2),
        kernel("deep_gemm::sm90_fp8_gemm", 40, 120.0, 11),
        kernel("_causal_conv1d_fwd_kernel", 200, 18.0, 12),
        kernel("delta_rule_chunk", 220, 30.0, 12),
        kernel("delta_rule_chunk", 250, 30.5, 12),
        kernel("deep_gemm::sm90_fp8_gemm", 500, 33.0, 13),
        kernel("_fused_qk_rmsnorm_rope_gate_kernel", 540, 8.0, None),
        kernel("softmax_warp_forward", 1250, 4.0, 14),
        kernel("graph_replayed_gemm", 2100, 40.0, 99, graph=3),
    ]
    return {"traceEvents": events}


def test_steps_are_read_from_the_gpu_annotations():
    steps = tp.step_windows(trace()["traceEvents"])
    assert [s.name for s in steps][0].startswith("execute_context_1(784)")
    assert (steps[0].prefill_sequences, steps[0].prefill_tokens, steps[0].tokens) == (1, 784, 784)
    assert (steps[1].decode_sequences, steps[1].decode_tokens) == (2, 2)
    assert tp.step_for(steps, 2100) is steps[1] and tp.step_for(steps, 1500) is None


def test_kernels_join_ops_frames_and_steps():
    all_records = tp.kernel_records(trace()["traceEvents"], "eager")
    records = {(r.kernel, r.op): r for r in all_records if "14336" not in r.input_dims}
    gdn_gemm = records[("deep_gemm::sm90_fp8_gemm", "vllm::dynamic_flashinfer_deepgemm_blockscale_gemm")]
    assert gdn_gemm.block == "gdn" and gdn_gemm.input_dims == "[[784,5120],[16384,5120]]"
    assert gdn_gemm.frame == "model_executor/layers/linear.py:593 forward"
    assert gdn_gemm.step.startswith("execute_context_1(784)") and gdn_gemm.step_tokens == 784
    conv = records[("_causal_conv1d_fwd_kernel", "vllm::qwen_gdn_attention_core_fused_norm")]
    assert conv.block == "gdn" and conv.grid == "1x1x1" and conv.block_dims == "128x1x1"
    rope = records[("_fused_qk_rmsnorm_rope_gate_kernel", "")]
    assert rope.block == "attention" and rope.input_dims == ""
    sampler = records[("softmax_warp_forward", "aten::softmax")]
    assert sampler.block == "sampler" and sampler.step == "outside_step"
    replayed = records[("graph_replayed_gemm", "")]
    assert replayed.graph is True and replayed.step.startswith("execute_context_0(0)_generation_2")


def test_attention_projection_is_labelled_by_the_decoder_layer_frame():
    records = tp.kernel_records(trace()["traceEvents"], "eager")
    attention_gemm = [r for r in records if r.kernel == "deep_gemm::sm90_fp8_gemm" and "14336" in r.input_dims]
    assert len(attention_gemm) == 1 and attention_gemm[0].block == "attention"


def test_frames_are_tracked_per_thread():
    frames = tp.op_frames(trace()["traceEvents"])
    assert frames[14] == ["vllm/v1/sample/sampler.py(50): forward"]
    assert frames[13][-1] == "vllm/model_executor/layers/linear.py(593): forward"
    assert "qwen_gdn_linear_attn.py(828)" not in " ".join(frames[13])


def test_aggregate_counts_identical_launches():
    rows = tp.aggregate(tp.kernel_records(trace()["traceEvents"], "eager"))
    chunk = next(r for r in rows if r["kernel"] == "delta_rule_chunk")
    assert chunk["count"] == 2 and chunk["total_us"] == 60.5 and chunk["min_us"] == 30.0
    assert rows[0]["step"] <= rows[-1]["step"]


def test_cli_writes_csv_and_summary_and_appends_modes(tmp_path):
    path = tmp_path / "t.pt.trace.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(trace(), handle)
    out, summary = tmp_path / "observed.csv", tmp_path / "summary.md"
    assert tp.main(["--trace", str(path), "--mode", "eager", "--out", str(out), "--summary", str(summary)]) == 0
    assert tp.main(["--trace", str(path), "--mode", "compiled", "--out", str(out), "--summary", str(summary), "--append"]) == 0
    with out.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {r["mode"] for r in rows} == {"eager", "compiled"}
    assert len([r for r in rows if r["mode"] == "eager"]) == len([r for r in rows if r["mode"] == "compiled"])
    text = summary.read_text()
    assert text.startswith("| mode | trace | step |") and "| compiled | t | execute_context_1(784)_generation_0(0) | 1 |" in text


def test_cli_replaces_rows_of_a_trace_parsed_again_and_keeps_the_others(tmp_path):
    first, second = tmp_path / "a.pt.trace.json.gz", tmp_path / "b.pt.trace.json.gz"
    for path in (first, second):
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(trace(), handle)
    out, summary = tmp_path / "observed.csv", tmp_path / "summary.md"
    tp.main(["--trace", str(first), "--trace", str(second), "--label", "cold_1k", "--label", "decode_64", "--mode", "compiled",
             "--out", str(out), "--summary", str(summary)])
    tp.main(["--trace", str(second), "--label", "decode_64", "--mode", "compiled", "--out", str(out), "--summary", str(summary), "--append"])
    rows = tp.read_rows(out)
    assert {r["trace"] for r in rows} == {"cold_1k", "decode_64"}
    assert len([r for r in rows if r["trace"] == "decode_64"]) == len([r for r in rows if r["trace"] == "cold_1k"])
