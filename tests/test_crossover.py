import json
from pathlib import Path

from analysis import crossover as cx
from analysis.data import load_csv, parse
from bench.core.results import write_csv


def attention_row(new_tokens, batch, cached, us):
    return {"benchmark": "attention_warm", "backend": "flash_attn_3", "kind": "warm", "new_tokens": new_tokens,
            "batch": batch, "cached": cached, "fraction": None, "kernel_us": us}


def gdn_row(new_tokens, batch, us, backend="flashinfer", history=1024):
    return {"benchmark": "gdn_warm", "backend": backend, "kind": "warm", "new_tokens": new_tokens,
            "batch": batch, "history": history, "fraction": None, "kernel_us": us}


ATTENTION = [attention_row(512, 1, c, us) for c, us in [(0, 40.0), (8000, 100.0), (16000, 200.0), (32000, 400.0)]]
GDN = [gdn_row(512, 1, 150.0), gdn_row(512, 1, 999.0, backend="triton"), gdn_row(512, 1, 151.0, history=0)]


def test_parse_keeps_ints_floats_bools_and_blanks_apart():
    assert parse("") is None and parse("True") is True and parse("784") == 784
    assert parse("1.5") == 1.5 and parse("flash_attn_3") == "flash_attn_3"
    assert isinstance(parse("784"), int) and isinstance(parse("784.0"), float)


def test_crossing_interpolates_between_grid_points():
    curve = [(0, 40.0), (8000, 100.0), (16000, 200.0)]
    assert cx.crossing(curve, 150.0) == 12000.0
    assert cx.crossing(curve, 10.0) == 0.0
    assert cx.crossing(curve, 500.0) is None


def test_crossovers_use_the_requested_backend_and_history():
    rows = cx.crossovers(ATTENTION, GDN)
    assert len(rows) == 1
    row = rows[0]
    assert row.gdn_layer_us == 150.0 and row.layer_crossover_cached == 12000.0
    assert row.step_crossover_cached == cx.crossing([(c, 16 * us) for c, us in [(0, 40.0), (8000, 100.0), (16000, 200.0), (32000, 400.0)]], 48 * 150.0)
    assert row.step_share_attention_at_max == 16 * 400.0 / (16 * 400.0 + 48 * 150.0)
    assert cx.crossovers(ATTENTION, GDN, gdn_backend="triton")[0].layer_crossover_cached is None


def test_fraction_view_rows_are_ignored():
    extra = ATTENTION + [dict(attention_row(512, 1, 7056, 5.0), fraction=0.9)]
    assert cx.crossovers(extra, GDN)[0].attention_layer_us_at_zero == 40.0


def test_cli_round_trips_through_csv(tmp_path):
    attention, gdn = tmp_path / "a.csv", tmp_path / "g.csv"
    write_csv(ATTENTION, attention)
    write_csv(GDN, gdn)
    out, table = tmp_path / "c.json", tmp_path / "c.md"
    assert cx.main(["--attention", str(attention), "--gdn", str(gdn), "--out", str(out), "--table", str(table)]) == 0
    assert load_csv(attention)[0]["cached"] == 0
    data = json.loads(out.read_text())
    assert data[0]["layer_crossover_cached"] == 12000.0
    assert "| 512 | 1 | 150 | 40 | 400 | 12000 |" in table.read_text()
