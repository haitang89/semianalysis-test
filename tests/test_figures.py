from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from analysis import figures
from bench.core.results import write_csv
from bench.core.roofline import Ceilings

CEILINGS = Ceilings(4000.0, 700.0, 1400.0)


def row(benchmark, backend, **fields):
    base = {"point_id": "p", "benchmark": benchmark, "backend": backend, "kernel_us": 100.0, "flops": 1e9, "bytes_moved": 1e8,
            "kernel_gbps": 1000.0, "kernel_tflops": 10.0}
    base.update(fields)
    return base


def sample_data():
    attention_cold = [row("attention_cold", "flash_attn_3", kind="decode", kv=kv, batch=b, kernel_us=10.0 * b) for kv in (1024, 4096) for b in (1, 64)]
    attention_cold += [row("attention_cold", "flash_attn_3", kind="prefill", tokens=t, batch=1) for t in (128, 4096)]
    attention_warm = [row("attention_warm", "flash_attn_3", kind="warm", new_tokens=512, batch=8, cached=c, fraction=None,
                          kernel_us=50.0 + c / 100) for c in (0, 7840, 62720)]
    gdn_warm = [row("gdn_warm", "flashinfer", kind="warm", new_tokens=512, batch=8, history=1024, fraction=None, kernel_us=260.0)]
    gdn_decode = [row("gdn_decode", b, kind="decode", batch=n, history=None) for b in ("packed", "sigmoid_gating") for n in (1, 64)]
    gdn_cold = [row("gdn_cold", b, kind="prefill", tokens=t, batch=1) for b in ("flashinfer", "triton") for t in (128, 4096)]
    ragged = [row("gdn_ragged", "flashinfer", kind="ragged", distribution=d, total_tokens=16384, sequences=32, kernel_us=us, kernel_tflops=100.0 / us)
              for d, us in (("uniform", 100.0), ("jitter", 102.0), ("lognormal", 104.0), ("bimodal", 110.0), ("mixed", 101.0))]
    pages = [row("attention_pages", "flash_attn_3", kind="warm", page_size=p, new_tokens=512, cached=1568, batch=8) for p in (16, 784)]
    pages += [row("attention_pages", "flash_attn_3", kind="decode", page_size=p, kv=16384, batch=64) for p in (16, 784)]
    return {"attention_cold": attention_cold, "attention_warm": attention_warm, "gdn_warm": gdn_warm, "gdn_decode": gdn_decode,
            "gdn_cold": gdn_cold, "gdn_ragged": ragged, "attention_pages": pages}


def test_every_figure_renders_from_complete_data(tmp_path):
    made = figures.render(sample_data(), CEILINGS, tmp_path, log=lambda m: None)
    assert made == [name for name, _ in figures.FIGURES]
    for name in made:
        assert (tmp_path / f"{name}.png").stat().st_size > 0 and (tmp_path / f"{name}.svg").exists()


def test_missing_series_are_skipped_with_a_note(tmp_path):
    notes = []
    made = figures.render({"gdn_decode": sample_data()["gdn_decode"]}, CEILINGS, tmp_path, log=notes.append)
    assert made == ["gdn_decode_bandwidth", "roofline_attainment"]
    assert any(note.startswith("skipped crossover") for note in notes)


def test_ragged_efficiency_is_relative_to_the_uniform_batch():
    values = figures.ragged_efficiencies(sample_data()["gdn_ragged"], 16384, 32)
    assert values["flashinfer"]["uniform"] == 1.0 and values["flashinfer"]["bimodal"] == pytest.approx(100 / 110)
    assert figures.ragged_efficiencies(sample_data()["gdn_ragged"], 4096, 8) == {}


def test_cli_reads_processed_csvs(tmp_path):
    processed = tmp_path / "processed"
    for name, rows in sample_data().items():
        write_csv(rows, processed / f"{name}.csv")
    ceilings = tmp_path / "ceilings.json"
    ceilings.write_text('{"hbm_bw_gbps_measured": 4000, "bf16_tflops_measured": 700, "fp8_tflops_measured": 1400}')
    out = tmp_path / "figs"
    assert figures.main(["--processed", str(processed), "--ceilings", str(ceilings), "--out", str(out)]) == 0
    assert len(list(out.glob("*.png"))) == len(figures.FIGURES)
