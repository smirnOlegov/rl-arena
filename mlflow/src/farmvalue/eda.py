"""EDA датасета снимков: графики и сводка в артефакты отдельного run'а.

Запуск: `uv run python -m farmvalue.eda v1`
"""

import argparse
import logging
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from farmvalue import plots
from farmvalue.config import PipelineSettings, get_settings
from farmvalue.dataset import TARGET, VERSIONS, as_mlflow_dataset, load
from farmvalue.features import FARM_FEATURES
from farmvalue.tracking import git_commit, use_experiment

log = logging.getLogger(__name__)

PHASES = {"early": (1, 10), "mid": (11, 20), "late": (21, 29)}


def leader_accuracy(df: pd.DataFrame, column: str) -> pd.Series:
    """Доля снимков, где лидер по признаку в итоге выиграл (равенство не считается)."""
    lead = df[df[column] != 0]
    hit = (lead[column] > 0).astype(int) == lead[TARGET]
    return hit.groupby(lead["day"]).mean()


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    cols = [f"diff_{name}" for name in FARM_FEATURES]
    out = {}
    for phase, (lo, hi) in PHASES.items():
        part = df[df["day"].between(lo, hi)]
        varying = [c for c in cols if part[c].nunique() > 1]
        out[phase] = part[varying].corrwith(part[TARGET]).reindex(cols, fill_value=0.0)
    return pd.DataFrame(out)


def plot_leader(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plots.figure()
    for column, label in (("diff_money", "лидер по деньгам"), ("diff_assets", "лидер по активам")):
        acc = leader_accuracy(df, column)
        ax.plot(acc.index, acc.to_numpy(), marker="o", markersize=4, label=label)
    ax.axhline(0.5, color=plots.NEUTRAL, linewidth=1, linestyle="--")
    ax.set(xlabel="день", ylabel="доля побед лидера", ylim=(0.4, 1.0))
    ax.set_title("Насколько рано определяется исход партии")
    ax.legend(loc="lower right")
    fig.savefig(out / "leader_wins_by_day.png")
    plots.close(fig)


def plot_money_gap(df: pd.DataFrame, out: Path) -> None:
    days = [5, 10, 15, 20, 25]
    fig, ax = plots.figure()
    data, positions, colors = [], [], []
    for i, day in enumerate(days):
        part = df[df["day"] == day]
        for j, won in enumerate((1, 0)):
            data.append(part.loc[part[TARGET] == won, "diff_money"].to_numpy())
            positions.append(i * 3 + j)
            colors.append(plots.SERIES[j])
    box = ax.boxplot(data, positions=positions, widths=0.8, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], colors, strict=True):
        patch.set(facecolor=color, edgecolor=color, alpha=0.75)
    for median in box["medians"]:
        median.set(color=plots.SURFACE, linewidth=1.5)
    ax.set_xticks([i * 3 + 0.5 for i in range(len(days))], [f"день {d}" for d in days])
    ax.set_ylabel("деньги seat 0 минус seat 1")
    ax.set_title("Разрыв в деньгах по исходу партии")
    ax.legend(
        [box["boxes"][0], box["boxes"][1]], ["seat 0 выиграл", "seat 0 проиграл"], loc="upper left"
    )
    fig.savefig(out / "money_gap_by_outcome.png")
    plots.close(fig)


def plot_correlations(corr: pd.DataFrame, out: Path) -> None:
    top = corr.reindex(corr["mid"].abs().sort_values().index).tail(12)
    fig, ax = plots.figure(7, 5)
    y = np.arange(len(top))
    for i, phase in enumerate(PHASES):
        ax.barh(y + (i - 1) * 0.27, top[phase], height=0.25, label=phase)
    ax.set_yticks(y, [name.removeprefix("diff_") for name in top.index])
    ax.axvline(0, color=plots.NEUTRAL, linewidth=1)
    ax.set_xlabel("корреляция разности признака с победой seat 0")
    ax.set_title("Какие разности связаны с исходом, по фазам игры")
    ax.legend(loc="lower right")
    fig.savefig(out / "feature_correlations.png")
    plots.close(fig)


def plot_final_bank(episodes: pd.DataFrame, out: Path) -> None:
    fig, ax = plots.figure()
    bank = pd.concat([episodes["reward0"], episodes["reward1"]])
    margin = (episodes["reward0"] - episodes["reward1"]).abs()
    ax.hist(bank, bins=40, color=plots.SERIES[0], alpha=0.85, label="финальный банк")
    ax.hist(margin, bins=40, color=plots.SERIES[1], alpha=0.85, label="|разница банков|")
    ax.set(xlabel="деньги", ylabel="партий")
    ax.set_title("Финальный банк и отрыв победителя")
    ax.legend()
    fig.savefig(out / "final_bank.png")
    plots.close(fig)


def summary(df: pd.DataFrame, episodes: pd.DataFrame, corr: pd.DataFrame) -> str:
    acc = leader_accuracy(df, "diff_money")
    mirror = (episodes["diff_money_end"] == 0).mean()
    top = corr["mid"].abs().sort_values(ascending=False).head(5)
    lines = [
        "# EDA",
        "",
        f"- партий: {len(episodes)}, снимков: {len(df)}, снимков на партию: "
        f"{df.groupby('episode_id').size().median():.0f}",
        f"- доля побед seat 0: {episodes[TARGET].mean():.3f}, классы почти сбалансированы",
        f"- лидер по деньгам выигрывает: день 5 {acc.get(5.0, float('nan')):.2f}, "
        f"день 15 {acc.get(15.0, float('nan')):.2f}, день 25 {acc.get(25.0, float('nan')):.2f}",
        f"- партий с равными деньгами на 29 день (зеркальные стратегии): {mirror:.1%}",
        "- снимки одной партии сильно зависимы, поэтому сплит только по episode_id",
        "- самые связанные с исходом разности в середине игры: "
        + ", ".join(f"{k.removeprefix('diff_')} ({corr.loc[k, 'mid']:+.2f})" for k in top.index),
        "",
        "Метрика: ROC-AUC по снимкам тестовых партий, дополнительно log loss и Brier,",
        "потому что модель используется как вероятность победы, и AUC по фазам игры.",
    ]
    return "\n".join(lines) + "\n"


def run_eda(version: str, settings: PipelineSettings) -> str:
    df, source, data_run_id = load(version, settings)
    episodes = df.groupby("episode_id").agg(
        reward0=("reward0", "first"),
        reward1=("reward1", "first"),
        **{TARGET: (TARGET, "first")},
        diff_money_end=("diff_money", "last"),
    )
    corr = correlations(df)

    use_experiment(settings, settings.data_experiment)
    with mlflow.start_run(run_name=f"eda-{version}") as run:
        mlflow.set_tags({"stage": "eda", "dataset_version": version, "data_run_id": data_run_id})
        mlflow.log_param("code_commit", git_commit())
        mlflow.log_input(
            as_mlflow_dataset(df, source, f"kaggriculture-snapshots-{version}"), context="eda"
        )
        acc = leader_accuracy(df, "diff_money")
        mlflow.log_metrics(
            {
                "seat0_win_rate": float(episodes[TARGET].mean()),
                "leader_acc_day5": float(acc.get(5.0, np.nan)),
                "leader_acc_day15": float(acc.get(15.0, np.nan)),
                "leader_acc_day25": float(acc.get(25.0, np.nan)),
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            plot_leader(df, out)
            plot_money_gap(df, out)
            plot_correlations(corr, out)
            plot_final_bank(episodes, out)
            (out / "summary.md").write_text(summary(df, episodes, corr), encoding="utf-8")
            corr.round(4).to_csv(out / "correlations.csv")
            mlflow.log_artifacts(str(out), artifact_path="eda")
        mlflow.log_table(df.describe().T.reset_index(), artifact_file="eda/describe.json")
        return str(run.info.run_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", choices=VERSIONS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(run_eda(args.version, get_settings()))


if __name__ == "__main__":
    main()
