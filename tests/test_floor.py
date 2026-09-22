from bench.core.floor import measure
from bench.core.mock import ModelClock


def test_measure_reports_both_floors():
    result = measure(lambda: None, ModelClock(latency_us=2.0, launch_us=20.0, noise=0.0), repeats=30)
    assert result["repeats"] == 30
    assert result["graph_replay_floor_us"] < result["eager_floor_us"]
    assert result["eager_floor_us"] > 15.0
    assert set(result) == {"eager_floor_us", "graph_replay_floor_us", "eager_p10_us", "eager_p90_us", "repeats"}
