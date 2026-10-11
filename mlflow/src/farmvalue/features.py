"""Признаки снимка партии из публичной части observation.

Используется и при сборке датасета, и в inference-сервисе, поэтому признаки
считаются только по тому, что видит любой из игроков: обе фермы, рынок, город.
Приватная часть observation (сарай, семена, инвентарь рабочих) не участвует.
"""

from collections.abc import Mapping
from statistics import fmean
from typing import Any

CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = {"GOOSE": 300, "COW": 400, "SHEEP": 500}

FARM_FEATURES = (
    "money",
    "assets",
    "hands",
    "hires_today",
    "quadrants",
    "plants",
    *(f"plants_{crop.lower()}" for crop in CROPS),
    "plants_ready",
    "plants_unwatered",
    "plants_fertilized",
    "animals",
    *(f"animals_{animal.lower()}" for animal in ANIMALS),
    "animals_unfed",
    "animal_yield",
    "structures_empty",
)
GLOBAL_FEATURES = ("day", "shops", "price_mean")
FEATURES = (
    *GLOBAL_FEATURES,
    *(f"p0_{name}" for name in FARM_FEATURES),
    *(f"p1_{name}" for name in FARM_FEATURES),
    *(f"diff_{name}" for name in FARM_FEATURES),
)

type Observation = Mapping[str, Any]


def farm_features(farm: Mapping[str, Any], day: int) -> dict[str, float]:
    """Признаки одной фермы."""
    row: dict[str, float] = dict.fromkeys(FARM_FEATURES, 0.0)
    row["money"] = float(farm["money"])
    row["hands"] = len(farm.get("hands") or [])
    row["hires_today"] = float(farm.get("hires_today") or 0)
    row["quadrants"] = len(farm.get("unlocked_quadrants") or [])

    for tile in (t for line in farm["tiles"] for t in line):
        if not isinstance(tile, Mapping):
            continue
        if tile["kind"] == "PLANT":
            row["plants"] += 1
            crop = str(tile.get("crop", "")).lower()
            if f"plants_{crop}" in row:
                row[f"plants_{crop}"] += 1
            row["plants_ready"] += tile.get("yield_units", 0) > 0
            row["plants_unwatered"] += tile.get("consecutive_unwatered", 0) > 0
            row["plants_fertilized"] += tile.get("fertilized_until_day", -1) >= day
        elif animal := tile.get("animal"):
            row["animals"] += 1
            row[f"animals_{animal.lower()}"] += 1
            row["animals_unfed"] += tile.get("consecutive_unfed", 0) > 0
            row["animal_yield"] += tile.get("yield_units", 0)
            row["assets"] += ANIMALS.get(animal, 0)
        else:
            row["structures_empty"] += 1

    row["assets"] += row["money"]
    return row


def snapshot_features(obs: Observation) -> dict[str, float]:
    """Признаки снимка: общие, по каждой ферме и разности seat 0 минус seat 1."""
    day = int(obs["day"])
    farms = obs["farms"]
    if len(farms) != 2:
        raise ValueError(f"expected 2 farms, got {len(farms)}")
    p0, p1 = (farm_features(farm, day) for farm in farms)
    prices = (obs.get("market") or {}).get("prices") or {}

    row: dict[str, float] = {
        "day": day + int(obs.get("hour", 0)) / 24,
        "shops": len((obs.get("town") or {}).get("unlocked_shops") or []),
        "price_mean": fmean(prices.values()) if prices else 0.0,
    }
    row.update({f"p0_{k}": v for k, v in p0.items()})
    row.update({f"p1_{k}": v for k, v in p1.items()})
    row.update({f"diff_{k}": p0[k] - p1[k] for k in FARM_FEATURES})
    return row
