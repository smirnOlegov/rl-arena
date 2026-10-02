"""Prioritized Fictitious Self-Play (PFSP) — выбор оппонента, как в AlphaStar.

Оппонент сэмплируется с весом f(p), где p — винрейт обучаемого агента против
кандидата:

- `hard`: f(p) = (1 - p)^2 — чаще играем с теми, кого не можем обыграть;
- `variance`: f(p) = p(1 - p) — с равными по силе (максимум сигнала);
- `uniform`: f(p) = 1 — классический fictitious self-play.

Против неизвестного кандидата (игр не было) берётся p = 0.5.
"""

import random
from collections.abc import Hashable, Mapping, Sequence
from enum import StrEnum

UNKNOWN_WINRATE = 0.5


class PfspWeighting(StrEnum):
    """Функция приоритета PFSP."""

    hard = "hard"
    variance = "variance"
    uniform = "uniform"


def pfsp_weight(winrate: float, weighting: PfspWeighting) -> float:
    """Вернуть вес кандидата при данном винрейте против него."""
    match weighting:
        case PfspWeighting.hard:
            return (1.0 - winrate) ** 2
        case PfspWeighting.variance:
            return winrate * (1.0 - winrate)
        case PfspWeighting.uniform:
            return 1.0


def sample_opponent[T: Hashable](
    candidates: Sequence[T],
    winrates: Mapping[T, float],
    weighting: PfspWeighting,
    rng: random.Random,
) -> T:
    """Выбрать оппонента из `candidates` по весам PFSP.

    Если все веса нулевые (агент обыгрывает всех с p = 1 при `hard`), падаем
    в равномерный выбор — иначе обучение встанет.
    """
    if not candidates:
        raise ValueError("candidates must not be empty")
    weights = [pfsp_weight(winrates.get(c, UNKNOWN_WINRATE), weighting) for c in candidates]
    if sum(weights) == 0:
        return rng.choice(candidates)
    return rng.choices(candidates, weights=weights, k=1)[0]
