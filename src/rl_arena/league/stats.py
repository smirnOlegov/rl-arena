"""Статистика матчей: счёт и доверительный интервал винрейта."""

import math
from dataclasses import dataclass

from rl_arena.domain import MatchResult


@dataclass(frozen=True, slots=True)
class Record:
    """Сводка игр одного агента (против одного оппонента или против пула)."""

    wins: int = 0
    draws: int = 0
    losses: int = 0

    @property
    def games(self) -> int:
        """Всего сыграно."""
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float | None:
        """Доля очков: победа = 1, ничья = 0.5. `None`, если игр не было."""
        if self.games == 0:
            return None
        return (self.wins + 0.5 * self.draws) / self.games

    def add(self, result: MatchResult, count: int = 1) -> "Record":
        """Вернуть новую сводку с учётом `count` исходов `result`."""
        match result:
            case MatchResult.win:
                return Record(self.wins + count, self.draws, self.losses)
            case MatchResult.draw:
                return Record(self.wins, self.draws + count, self.losses)
            case MatchResult.loss:
                return Record(self.wins, self.draws, self.losses + count)

    def flipped(self) -> "Record":
        """Та же сводка с точки зрения оппонента."""
        return Record(wins=self.losses, draws=self.draws, losses=self.wins)

    def __add__(self, other: "Record") -> "Record":
        """Сложить две сводки."""
        return Record(self.wins + other.wins, self.draws + other.draws, self.losses + other.losses)


def wilson_interval(score: float, games: int, z: float = 1.96) -> tuple[float, float]:
    """Вернуть доверительный интервал Уилсона для доли побед.

    В отличие от нормального приближения, не вылезает за [0, 1] и адекватен на
    малых выборках — а у свежего агента в лиге выборка всегда маленькая.
    Без игр возвращается максимально неинформативный интервал (0, 1).
    """
    if games == 0:
        return 0.0, 1.0
    z2 = z * z
    denominator = 1 + z2 / games
    center = (score + z2 / (2 * games)) / denominator
    margin = z * math.sqrt(score * (1 - score) / games + z2 / (4 * games * games)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)
