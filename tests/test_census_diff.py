import csv
import json
from pathlib import Path

from bench.census import diff as census_diff
from bench.census.analytic import census

INVENTORY = census(json.loads(Path("configs/qwen3.8-27b-fp8.config.json").read_text(encoding="utf-8")))["ops"]
GEMMS = census_diff.gemm_index(INVENTORY)


def observed(kernel, op="", dims="", block="gdn", step="execute_context_1(784)_generation_0(0)", count=48, total_us=100.0, instances=1):
    return {"mode": "eager", "step": step, "step_instances": instances, "block": block, "kernel": kernel, "op": op,
            "input_dims": dims, "count": count, "total_us": total_us}


def test_gemm_dims_follow_the_operator_layout():
    assert census_diff.gemm_dims("vllm::dynamic_flashinfer_deepgemm_blockscale_gemm", "[[784,5120],[16384,5120],[128,40]]") == (5120, 16384)
    assert census_diff.gemm_dims("aten::mm", "[[1,5120],[5120,248320]]") == (5120, 248320)
    assert census_diff.gemm_dims("aten::add", "[[1,5120]]") is None


def test_graph_replayed_gemms_are_read_from_the_kernel_template_and_split_when_ambiguous():
    assert census_diff.gemm_dims("", "", "void deep_gemm::fp8_gemm_kernel_swapAB<34816u, 5120u, 128u, 16u>") == (5120, 34816)
    assert census_diff.gemm_dims("", "", "void deep_gemm::sm90_fp8_gemm_1d2d_impl<(cute::UMMA::Major)0, 0u, 5120u, 17408u, 1u>") == (17408, 5120)
    assert census_diff.explain(observed("void deep_gemm::fp8_gemm_kernel_swapAB<34816u, 5120u, 128u>", block="gemm"), GEMMS) == ("gate_up_proj",)
    shared = census_diff.explain(observed("void deep_gemm::fp8_gemm_kernel_swapAB<5120u, 6144u, 128u>", block="gemm"), GEMMS)
    assert sorted(set(shared)) == ["o_proj", "out_proj"] and shared.count("out_proj") == 3 * shared.count("o_proj")
    hinted = dict(observed("void deep_gemm::fp8_gemm_kernel_swapAB<5120u, 6144u, 128u>", block="gemm"), frame="model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:828 forward")
    assert census_diff.explain(hinted, GEMMS) == ("out_proj",)
    assert census_diff.explain(observed("nvjet_sm90_tst_32x64_64x16_1x2_h_bz_splitK_TNN", block="gemm"), GEMMS) == ("in_proj_ba",)
    assert census_diff.explain(observed("triton_red_fused__to_copy_add_fused_add_rms_norm_2", "triton_red_fused__to_copy_add_fused_add_rms_norm_2", block="norm"), GEMMS) == census_diff.NORMS
    assert census_diff.explain(observed("triton_poi_fused_mul_silu_slice_1", block="mlp"), GEMMS) == ("silu_and_mul",)


def test_gemms_match_by_shape_and_the_block_breaks_the_tie():
    row = observed("deep_gemm::sm90_fp8_gemm", "vllm::dynamic_flashinfer_deepgemm_blockscale_gemm", "[[784,6144],[5120,6144]]", block="attention")
    assert census_diff.explain(row, GEMMS) == ("o_proj",)
    row["block"] = "gdn"
    assert census_diff.explain(row, GEMMS) == ("out_proj",)
    unknown = observed("some_gemm", "aten::mm", "[[784,999],[999,7]]")
    assert census_diff.explain(unknown, GEMMS) == ()


def test_named_kernels_follow_the_rule_table():
    assert census_diff.explain(observed("_causal_conv1d_fwd_kernel"), GEMMS) == ("causal_conv1d",)
    assert census_diff.explain(observed("_fused_qk_rmsnorm_rope_gate_kernel", block="attention"), GEMMS) == ("q_norm", "k_norm", "rotary")
    assert census_diff.explain(observed("tensorrt_llm::scale_1x128_kernel", "vllm::dynamic_flashinfer_deepgemm_blockscale_gemm", "[[1,5120],[34816,5120]]", block="mlp"), GEMMS) == ("fp8_activation_quant",)
    assert census_diff.explain(observed("void at::native::reduce_kernel", "aten::mean", "[[784,5120]]", block="norm"), GEMMS)[0] == "input_layernorm"
    assert census_diff.explain(observed("_prepare_rope_positions_kernel", block="attention"), GEMMS) == (census_diff.ENGINE,)
    assert census_diff.explain(observed("mystery_kernel", block="other"), GEMMS) == ()


def test_diff_reports_coverage_per_step_and_normalizes_by_step_instances():
    rows = [
        observed("_causal_conv1d_update_kernel", step="decode", count=336, total_us=700.0, instances=7),
        observed("mystery_kernel", block="other", step="decode", count=7, total_us=70.0, instances=7),
        observed("deep_gemm::sm90_fp8_gemm", "vllm::dynamic_flashinfer_deepgemm_blockscale_gemm", "[[784,5120],[16384,5120]]", count=48, total_us=5850.0),
    ]
    result = census_diff.diff(rows, INVENTORY)
    conv = next(op for op in result["ops"] if op["op"] == "causal_conv1d")
    assert conv["observed_per_step"] == {"eager: decode": 48.0} and conv["expected_per_step"] == 48
    assert [r["kernel"] for r in result["unexplained"]] == ["mystery_kernel"]
    assert result["coverage"]["eager: decode"]["explained_share"] == round(100.0 / 110.0, 4)
    assert result["coverage"]["eager: execute_context_1(784)_generation_0(0)"]["explained_share"] == 1.0
    table = census_diff.markdown(result)
    assert "| in_proj_qkvz | gdn | yes | 48 | eager: execute_context_1(784)_generation_0(0): 48 |" in table
    assert "Unexplained kernels" in table


def test_cli_exit_code_reflects_unexplained_rows(tmp_path):
    inventory = tmp_path / "inv.json"
    inventory.write_text(json.dumps({"ops": INVENTORY}))
    observed_path = tmp_path / "obs.csv"
    with observed_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(observed("k").keys()))
        writer.writeheader()
        writer.writerow(observed("_causal_conv1d_fwd_kernel"))
    out, table = tmp_path / "d.json", tmp_path / "d.md"
    assert census_diff.main(["--observed", str(observed_path), "--inventory", str(inventory), "--out", str(out), "--table", str(table)]) == 0
    with observed_path.open("a", newline="") as handle:
        csv.DictWriter(handle, fieldnames=list(observed("k").keys())).writerow(observed("mystery_kernel", block="other"))
    assert census_diff.main(["--observed", str(observed_path), "--inventory", str(inventory), "--out", str(out), "--table", str(table)]) == 1
