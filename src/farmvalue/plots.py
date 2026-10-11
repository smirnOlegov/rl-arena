"""Единый стиль графиков-артефактов (PNG для MLflow UI)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.figure import Figure

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
NEUTRAL = "#8a8a86"
SURFACE = "#fcfcfb"
INK = "#2b2b29"
MUTED = "#6b6b67"
GRID = "#e6e6e3"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": cycler(color=SERIES),
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "font.size": 10,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
    }
)


def figure(width: float = 7, height: float = 4) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(width, height))
    return fig, ax


def close(fig: Figure) -> None:
    plt.close(fig)
