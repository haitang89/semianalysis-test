import pytest

from bench.core.regimes import (
    RAGGED_DISTRIBUTIONS,
    ragged_batch,
    ragged_decode_batch,
    warm_contexts,
    warm_point,
)


def test_ninety_percent_warm_is_exact_when_the_context_is_whole_blocks():
    point = warm_point(15680, 0.9)
    assert (point.cached, point.new) == (14112, 1568)
    assert point.actual_fraction == pytest.approx(0.900)
    assert point.cached_blocks == 18


def test_a_power_of_two_context_rounds_the_cache_down_to_whole_blocks():
    point = warm_point(16384, 0.9)
    assert (point.cached, point.new) == (14112, 2272)
    assert point.actual_fraction == pytest.approx(0.861, abs=0.001)


def test_zero_fraction_is_cold_and_full_fraction_is_rejected():
    cold = warm_point(7840, 0.0)
    assert cold.cached == 0 and cold.new == 7840
    with pytest.raises(ValueError):
        warm_point(7840, 1.0)


def test_warm_contexts_are_multiples_of_ten_blocks():
    contexts = warm_contexts()
    assert contexts == [7840, 15680, 31360, 62720, 125440]
    assert all(context % 7840 == 0 for context in contexts)


@pytest.mark.parametrize("distribution", RAGGED_DISTRIBUTIONS)
@pytest.mark.parametrize("total,sequences", [(4096, 8), (16384, 32), (16384, 128)])
def test_every_ragged_batch_sums_exactly_to_its_total(distribution, total, sequences):
    batch = ragged_batch(distribution, total, sequences, seed=7)
    assert sum(batch.lengths) == total
    assert len(batch.lengths) == sequences
    assert min(batch.lengths) >= 1


def test_same_seed_gives_same_lengths_and_different_seed_differs():
    first = ragged_batch("lognormal", 16384, 32, seed=1)
    again = ragged_batch("lognormal", 16384, 32, seed=1)
    other = ragged_batch("lognormal", 16384, 32, seed=2)
    assert first.lengths == again.lengths
    assert first.lengths != other.lengths


def test_uniform_has_no_spread_and_bimodal_has_one_giant():
    uniform = ragged_batch("uniform", 4096, 8)
    assert uniform.length_cv == 0.0 and set(uniform.lengths) == {512}
    bimodal = ragged_batch("bimodal", 4096, 8)
    assert max(bimodal.lengths) == pytest.approx(2048, abs=8)
    assert bimodal.max_over_mean == pytest.approx(4.0, abs=0.02)


def test_jitter_stays_within_eighty_to_hundred_percent_before_rescaling():
    batch = ragged_batch("jitter", 16384, 32, seed=3)
    longest, shortest = max(batch.lengths), min(batch.lengths)
    assert shortest / longest >= 0.79


def test_mixed_batch_marks_half_the_sequences_as_single_token_decodes():
    batch = ragged_batch("mixed", 4096, 8)
    assert batch.decode_mask == (True,) * 4 + (False,) * 4
    assert batch.lengths[:4] == (1, 1, 1, 1)
    assert sum(batch.lengths) == 4096


def test_ragged_decode_keeps_total_kv_and_one_token_per_sequence():
    batch = ragged_decode_batch("lognormal", 64, 16384, seed=5)
    assert batch.sequences == 64 and sum(batch.lengths) == 64 * 16384
    assert all(batch.decode_mask)
    assert batch.max_over_mean > 1.5


def test_unknown_distribution_is_rejected():
    with pytest.raises(ValueError):
        ragged_batch("zipf", 4096, 8)
