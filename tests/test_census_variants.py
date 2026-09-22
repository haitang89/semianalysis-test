import json
from pathlib import Path

from bench.census import variants as varmod
from bench.census.analytic import geometry_from_config
from bench.census.variants import manager_block, variants

CONFIG = json.loads(Path("configs/qwen3.8-27b-fp8.config.json").read_text(encoding="utf-8"))
GEOMETRY = geometry_from_config(CONFIG)


def by_key(rows):
    return {(row.setting, row.value): row for row in rows}


def test_manager_block_matches_the_engine_log_for_the_benchmarked_configuration():
    assert manager_block(GEOMETRY, tp=1, kv_dtype="bf16", speculative_tokens=0) == 784


def test_fp8_kv_and_speculation_change_the_block_as_the_engine_does():
    assert manager_block(GEOMETRY, 1, "fp8", 0) == 1568
    assert manager_block(GEOMETRY, 1, "bf16", 1) == 800


def test_tensor_parallel_two_splits_projections_heads_and_state():
    row = by_key(variants(GEOMETRY))[("tensor_parallel", "2")]
    assert row.gdn_in_proj_qkvz == "[M, 5120] x [5120, 8192]"
    assert row.mlp_gate_up_proj == "[M, 5120] x [5120, 17408]"
    assert (row.q_heads_per_rank, row.kv_heads_per_rank, row.gdn_v_heads_per_rank) == (12, 2, 24)
    assert row.gdn_state_per_rank.startswith("(24, 128, 128)")
    assert row.measured is False


def test_tensor_parallel_eight_replicates_kv_heads():
    row = by_key(variants(GEOMETRY))[("tensor_parallel", "8")]
    assert row.kv_heads_per_rank == 1 and row.q_heads_per_rank == 3
    assert "replicated" in row.extra_ops


def test_only_the_baseline_is_marked_measured():
    rows = variants(GEOMETRY)
    assert [row.measured for row in rows].count(True) == 1
    assert rows[0].setting == "baseline" and rows[0].measured


def test_prefix_caching_off_removes_chunk_alignment():
    rows = by_key(variants(GEOMETRY))
    assert rows[("baseline", "TP 1, bf16 KV, no speculation, prefix caching on")].prefill_chunk_alignment == 784
    assert rows[("prefix_caching", "off")].prefill_chunk_alignment is None


def test_cli_writes_table_and_json(tmp_path):
    out, table = tmp_path / "v.json", tmp_path / "v.md"
    assert varmod.main(["--out", str(out), "--table", str(table)]) == 0
    rows = json.loads(out.read_text())
    assert len(rows) == 7 and table.read_text().count("analytic only") == 6
