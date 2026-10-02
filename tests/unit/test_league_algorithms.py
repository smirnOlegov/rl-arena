"""Чистые алгоритмы лиги: статистика, PFSP, расписание эксплоитеров."""

import random
from collections import Counter

import pytest

from rl_arena.domain import MatchResult
from rl_arena.league.exploiters import exploiter_name, is_resample_epoch
from rl_arena.league.pfsp import PfspWeighting, pfsp_weight, sample_opponent
from rl_arena.league.stats import Record, wilson_interval


def test_record_score_counts_draw_as_half() -> None:
    record = Record().add(MatchResult.win, 2).add(MatchResult.draw).add(MatchResult.loss)
    assert (record.wins, record.draws, record.losses, record.games) == (2, 1, 1, 4)
    assert record.score == pytest.approx(0.625)


def test_empty_record_has_no_score() -> None:
    assert Record().score is None


def test_record_flip_and_sum() -> None:
    record = Record(wins=3, draws=1, losses=2)
    assert record.flipped() == Record(wins=2, draws=1, losses=3)
    assert record + record.flipped() == Record(wins=5, draws=2, losses=5)


def test_wilson_interval_known_value() -> None:
    low, high = wilson_interval(0.5, 100)
    assert low == pytest.approx(0.4038, abs=1e-4)
    assert high == pytest.approx(0.5962, abs=1e-4)


@pytest.mark.parametrize(("score", "games"), [(0.0, 5), (1.0, 5), (1.0, 1), (0.3, 1000)])
def test_wilson_interval_stays_in_unit_range_and_contains_score(score: float, games: int) -> None:
    low, high = wilson_interval(score, games)
    assert 0.0 <= low <= score <= high <= 1.0


def test_wilson_interval_without_games_is_uninformative() -> None:
    assert wilson_interval(0.0, 0) == (0.0, 1.0)


def test_wilson_interval_narrows_with_more_games() -> None:
    small = wilson_interval(0.8, 10)
    large = wilson_interval(0.8, 1000)
    assert large[1] - large[0] < small[1] - small[0]


@pytest.mark.parametrize(
    ("weighting", "winrate", "expected"),
    [
        (PfspWeighting.hard, 0.0, 1.0),
        (PfspWeighting.hard, 1.0, 0.0),
        (PfspWeighting.hard, 0.5, 0.25),
        (PfspWeighting.variance, 0.5, 0.25),
        (PfspWeighting.variance, 1.0, 0.0),
        (PfspWeighting.uniform, 0.9, 1.0),
    ],
)
def test_pfsp_weight(weighting: PfspWeighting, winrate: float, expected: float) -> None:
    assert pfsp_weight(winrate, weighting) == pytest.approx(expected)


def test_sample_opponent_requires_candidates() -> None:
    with pytest.raises(ValueError, match="empty"):
        sample_opponent([], {}, PfspWeighting.hard, random.Random(0))


def test_sample_opponent_falls_back_to_uniform_when_all_weights_zero() -> None:
    candidates = ["a", "b", "c"]
    winrates = dict.fromkeys(candidates, 1.0)
    picks = Counter(
        sample_opponent(candidates, winrates, PfspWeighting.hard, random.Random(seed))
        for seed in range(300)
    )
    assert set(picks) == set(candidates)


def test_sample_opponent_hard_never_picks_fully_beaten_opponent() -> None:
    winrates = {"beaten": 1.0, "rival": 0.4}
    picks = {
        sample_opponent(["beaten", "rival"], winrates, PfspWeighting.hard, random.Random(seed))
        for seed in range(100)
    }
    assert picks == {"rival"}


def test_sample_opponent_unknown_candidate_gets_prior_weight() -> None:
    picks = Counter(
        sample_opponent(["new", "known"], {"known": 0.5}, PfspWeighting.hard, random.Random(s))
        for s in range(400)
    )
    assert 150 < picks["new"] < 250


@pytest.mark.parametrize(
    ("epoch", "every", "expected"),
    [(0, 5, False), (3, 5, False), (5, 5, True), (10, 5, True), (1, 1, True)],
)
def test_is_resample_epoch(epoch: int, every: int, expected: bool) -> None:
    assert is_resample_epoch(epoch, every) is expected


def test_exploiter_name_is_sortable_by_epoch() -> None:
    assert exploiter_name(5, 1) == "exploiter-e0005-1"
    assert exploiter_name(10, 0) > exploiter_name(9, 0)
