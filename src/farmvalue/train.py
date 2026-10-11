"""Эксперимент kaggriculture-winprob: варианты модели вероятности победы seat 0.

Все варианты принимают на вход одинаковую таблицу из `FEATURES`, а нужные
столбцы выбирают внутри пайплайна. Поэтому любую версию модели можно поставить
под alias champion, и сервис не заметит разницы во входе.

Запуск: `uv run python -m farmvalue.train --dataset v1 [--variant hgb_tuned]`
"""

import argparse
import json
import logging
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from farmvalue import plots
from farmvalue.config import PipelineSettings, get_settings
from farmvalue.dataset import TARGET, VERSIONS, as_mlflow_dataset, load
from farmvalue.eda import PHASES
from farmvalue.features import FEATURES
from farmvalue.tracking import git_commit, use_experiment

log = logging.getLogger(__name__)

PRIMARY_METRIC = "test_roc_auc"
# skops грузит только явно разрешённые типы; дерево бустинга надо разрешить
TRUSTED_TYPES = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]


@dataclass(frozen=True)
class Variant:
    description: str
    columns: tuple[str, ...]
    estimator: Callable[[], Any]
    scale: bool = False
    params: dict[str, Any] = field(default_factory=dict)

    def pipeline(self) -> Pipeline:
        select = StandardScaler() if self.scale else "passthrough"
        return Pipeline(
            [
                ("select", ColumnTransformer([("x", select, list(self.columns))])),
                ("model", self.estimator().set_params(**self.params)),
            ]
        )


VARIANTS = {
    "baseline_logreg": Variant(
        "логистическая регрессия на дне и разнице денег",
        ("day", "diff_money"),
        lambda: LogisticRegression(max_iter=1000),
        scale=True,
    ),
    "logreg_all": Variant(
        "логистическая регрессия на всех признаках",
        FEATURES,
        lambda: LogisticRegression(max_iter=5000),
        scale=True,
        params={"C": 0.3},
    ),
    "hgb_default": Variant(
        "градиентный бустинг, параметры по умолчанию",
        FEATURES,
        HistGradientBoostingClassifier,
        params={"max_iter": 200, "learning_rate": 0.1, "max_leaf_nodes": 31, "random_state": 0},
    ),
    "hgb_tuned": Variant(
        "градиентный бустинг: меньше листьев, сильнее регуляризация",
        FEATURES,
        HistGradientBoostingClassifier,
        params={
            "max_iter": 400,
            "learning_rate": 0.05,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 80,
            "l2_regularization": 1.0,
            "random_state": 0,
        },
    ),
}


def xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[list(FEATURES)].astype("float64"), df[TARGET]


def scores(y: pd.Series, proba: np.ndarray, prefix: str) -> dict[str, float]:
    if y.nunique() < 2:
        return {}
    return {
        f"{prefix}_roc_auc": float(roc_auc_score(y, proba)),
        f"{prefix}_log_loss": float(log_loss(y, proba, labels=[0, 1])),
        f"{prefix}_brier": float(brier_score_loss(y, proba)),
        f"{prefix}_accuracy": float(accuracy_score(y, proba >= 0.5)),
    }


def phase_scores(test: pd.DataFrame, proba: np.ndarray) -> dict[str, float]:
    out = {}
    for phase, (lo, hi) in PHASES.items():
        mask = test["day"].between(lo, hi).to_numpy()
        out[f"test_roc_auc_{phase}"] = float(roc_auc_score(test.loc[mask, TARGET], proba[mask]))
    return out


def plot_diagnostics(
    test: pd.DataFrame, proba: np.ndarray, pipe: Pipeline, variant: Variant, out: Path
) -> None:
    y = test[TARGET].to_numpy()

    fig, ax = plots.figure(5, 5)
    fpr, tpr, _ = roc_curve(y, proba)
    ax.plot(fpr, tpr, label=f"AUC {roc_auc_score(y, proba):.3f}")
    ax.plot([0, 1], [0, 1], color=plots.NEUTRAL, linewidth=1, linestyle="--")
    ax.set(xlabel="FPR", ylabel="TPR")
    ax.set_title("ROC на тестовых партиях")
    ax.legend(loc="lower right")
    fig.savefig(out / "roc_curve.png")
    plots.close(fig)

    fig, ax = plots.figure(5, 5)
    frac, mean = calibration_curve(y, proba, n_bins=10, strategy="quantile")
    ax.plot(mean, frac, marker="o", markersize=5)
    ax.plot([0, 1], [0, 1], color=plots.NEUTRAL, linewidth=1, linestyle="--")
    ax.set(xlabel="предсказанная вероятность", ylabel="доля побед")
    ax.set_title("Калибровка")
    fig.savefig(out / "calibration.png")
    plots.close(fig)

    fig, ax = plots.figure(4, 4)
    cm = confusion_matrix(y, proba >= 0.5)
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center", color=plots.INK)
    ax.set_xticks([0, 1], ["проигрыш", "победа"])
    ax.set_yticks([0, 1], ["проигрыш", "победа"])
    ax.set(xlabel="предсказано", ylabel="факт")
    ax.set_title("Матрица ошибок, порог 0.5")
    fig.savefig(out / "confusion_matrix.png")
    plots.close(fig)

    days = test["day"].to_numpy()
    by_day = pd.Series({d: roc_auc_score(y[days == d], proba[days == d]) for d in np.unique(days)})
    fig, ax = plots.figure()
    ax.plot(by_day.index, by_day.to_numpy(), marker="o", markersize=4)
    ax.axhline(0.5, color=plots.NEUTRAL, linewidth=1, linestyle="--")
    ax.set(xlabel="день", ylabel="ROC-AUC", ylim=(0.4, 1.0))
    ax.set_title("Качество по дню снимка")
    fig.savefig(out / "auc_by_day.png")
    plots.close(fig)

    x, _ = xy(test)
    imp = permutation_importance(
        pipe, x, y, scoring="roc_auc", n_repeats=5, random_state=0, n_jobs=-1
    )
    top = (
        pd.Series(imp.importances_mean, index=x.columns)
        .loc[list(variant.columns)]
        .sort_values()
        .tail(15)
    )
    fig, ax = plots.figure(6, 0.35 * len(top) + 1.2)
    ax.barh(top.index, top.to_numpy(), height=0.6)
    ax.set_xlabel("падение ROC-AUC при перемешивании")
    ax.set_title("Permutation importance")
    fig.savefig(out / "permutation_importance.png")
    plots.close(fig)


def league_test(settings: PipelineSettings) -> pd.DataFrame | None:
    """Тестовые партии лиги агентов из датасета v2, если он уже собран."""
    try:
        df, _, _ = load("v2", settings)
    except SystemExit:
        return None
    part = df[(df["split"] == "test") & (df["source"] == "league")]
    return part if len(part) else None


def train(name: str, version: str, settings: PipelineSettings) -> str:
    variant = VARIANTS[name]
    df, source, data_run_id = load(version, settings)
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]
    x_train, y_train = xy(train_df)
    x_test, y_test = xy(test_df)
    league = league_test(settings)

    use_experiment(settings, settings.winprob_experiment)
    with mlflow.start_run(run_name=f"{name}-{version}") as run:
        mlflow.set_tags(
            {
                "variant": name,
                "dataset_version": version,
                "data_run_id": data_run_id,
                "mlflow.note.content": variant.description,
            }
        )
        dataset_name = f"kaggriculture-snapshots-{version}"
        train_ds = as_mlflow_dataset(train_df, source, f"{dataset_name}-train")
        test_ds = as_mlflow_dataset(test_df, source, f"{dataset_name}-test")
        mlflow.log_input(train_ds, context="training")
        mlflow.log_input(test_ds, context="evaluation")
        mlflow.log_params(
            {
                "variant": name,
                "estimator": type(variant.estimator()).__name__,
                "features": len(variant.columns),
                "scale": variant.scale,
                "split": "sha1(episode_id) % 5",
                "code_commit": git_commit(),
                **{f"model__{k}": v for k, v in variant.params.items()},
            }
        )

        pipe = variant.pipeline()
        cv = cross_val_score(
            variant.pipeline(),
            x_train,
            y_train,
            groups=train_df["episode_id"],
            cv=GroupKFold(n_splits=5),
            scoring="roc_auc",
            n_jobs=-1,
        )
        pipe.fit(x_train, y_train)
        proba = pipe.predict_proba(x_test)[:, 1]

        model_info = mlflow.sklearn.log_model(
            pipe,
            name="model",
            signature=infer_signature(x_test.head(20), pipe.predict_proba(x_test.head(20))),
            input_example=x_test.head(3),
            pyfunc_predict_fn="predict_proba",
            skops_trusted_types=TRUSTED_TYPES,
        )
        metrics = {
            "cv_roc_auc_mean": float(cv.mean()),
            "cv_roc_auc_std": float(cv.std()),
            **scores(y_test, proba, "test"),
            **phase_scores(test_df, proba),
        }
        mlflow.log_metrics(metrics, model_id=model_info.model_id, dataset=test_ds)
        if league is not None:
            x_league, y_league = xy(league)
            league_metrics = scores(y_league, pipe.predict_proba(x_league)[:, 1], "league")
            mlflow.log_metrics(league_metrics, model_id=model_info.model_id)
            metrics |= league_metrics

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            plot_diagnostics(test_df, proba, pipe, variant, out)
            (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            mlflow.log_artifacts(str(out), artifact_path="diagnostics")
        log.info("%s-%s: %s", name, version, {k: round(v, 4) for k, v in metrics.items()})
        return str(run.info.run_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=VERSIONS, default="v1")
    parser.add_argument("--variant", choices=list(VARIANTS), action="append")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    for name in args.variant or list(VARIANTS):
        print(train(name, args.dataset, settings))


if __name__ == "__main__":
    main()
