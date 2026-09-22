import json
from pathlib import Path

import pytest

from bench.census import analytic
from bench.census.analytic import census, gdn_state_bytes, geometry_from_config, inventory, kv_bytes_per_token

CONFIG = json.loads(Path("configs/qwen3.8-27b-fp8.config.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def geometry():
    return geometry_from_config(CONFIG)


@pytest.fixture(scope="module")
def ops(geometry):
    return {op.name: op for op in inventory(geometry)}


def test_layer_layout(geometry):
    assert geometry.layers == 64
    assert geometry.gdn_layers == 48
    assert len(geometry.attention_layers) == 16
    assert geometry.attention_layers[:3] == (3, 7, 11) and geometry.attention_layers[-1] == 63
    assert geometry.rotary_dims == 64 and geometry.conv_dim == 10240


def test_gemm_shapes_follow_the_fused_vllm_layout(ops):
    def shape(name):
        return ops[name].shape

    assert shape("in_proj_qkvz") == "[M, 5120] x [5120, 16384] -> [M, 16384]"
    assert shape("in_proj_ba") == "[M, 5120] x [5120, 96] -> [M, 96]"
    assert shape("out_proj") == "[M, 6144] x [6144, 5120] -> [M, 5120]"
    assert shape("qkv_proj") == "[M, 5120] x [5120, 14336] -> [M, 14336]"
    assert shape("o_proj") == "[M, 6144] x [6144, 5120] -> [M, 5120]"
    assert shape("gate_up_proj") == "[M, 5120] x [5120, 34816] -> [M, 34816]"
    assert shape("down_proj") == "[M, 17408] x [17408, 5120] -> [M, 5120]"
    assert shape("lm_head") == "[M, 5120] x [5120, 248320] -> [M, 248320]"


def test_per_step_counts(ops):
    assert ops["in_proj_qkvz"].per_step == 48 and ops["gated_delta_rule"].per_step == 48
    assert ops["qkv_proj"].per_step == 16 and ops["attention"].per_step == 16
    assert ops["gate_up_proj"].per_step == 64 and ops["input_layernorm"].per_step == 64
    assert ops["lm_head"].per_step == 1
    assert ops["mtp_fc"].per_step == 0


def test_dtypes_follow_the_fp8_checkpoint(ops):
    assert ops["gate_up_proj"].dtype == "fp8" and ops["qkv_proj"].dtype == "fp8" and ops["in_proj_qkvz"].dtype == "fp8"
    assert ops["in_proj_ba"].dtype == "bf16" and ops["lm_head"].dtype == "bf16"
    assert ops["fp8_activation_quant"].per_step == 2 * 48 + 2 * 16 + 2 * 64


CHECKPOINT_BYTES = 30.88e9
VISION_TOWER_BYTES = 0.86e9
MTP_HEAD_BYTES = 0.5e9


def test_weight_bytes_match_the_checkpoint_language_model():
    predicted = census(CONFIG)["language_model_weight_bytes"]
    expected = CHECKPOINT_BYTES - VISION_TOWER_BYTES - MTP_HEAD_BYTES
    assert predicted == pytest.approx(expected, rel=0.02)


def test_state_and_kv_sizes(geometry):
    state = gdn_state_bytes(geometry)
    assert state["recurrent_bytes"] == 48 * 128 * 128 * 4
    assert state["conv_bytes"] == 10240 * 3 * 2
    kv = kv_bytes_per_token(geometry)
    assert kv["per_layer_bytes"] == 2 * 4 * 256 * 2 and kv["layers"] == 16


def test_cli_writes_json_and_table(tmp_path):
    out, table = tmp_path / "ops.json", tmp_path / "ops.md"
    assert analytic.main(["--out", str(out), "--table", str(table)]) == 0
    record = json.loads(out.read_text())
    assert record["attention_layers"] == list(range(3, 64, 4))
    assert table.read_text().startswith("| block | op |")
