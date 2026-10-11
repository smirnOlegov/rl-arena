"""Model Registry для kaggriculture-winprob: регистрация версий и alias.

`register` берёт baseline и лучшие runs по test_roc_auc, регистрирует их
версиями (по возрастанию качества) и ставит champion на лучшую, challenger на
следующую. Повторный запуск не создаёт дубликатов для уже зарегистрированных runs.

    uv run python -m farmvalue.registry register --top 2
    uv run python -m farmvalue.registry promote 1
    uv run python -m farmvalue.registry show
"""

import argparse
import logging

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Run
from mlflow.entities.model_registry import ModelVersion

from farmvalue.config import PipelineSettings, get_settings
from farmvalue.train import PRIMARY_METRIC

log = logging.getLogger(__name__)

CHAMPION = "champion"
CHALLENGER = "challenger"


def client(settings: PipelineSettings) -> MlflowClient:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return MlflowClient()


def candidates(c: MlflowClient, settings: PipelineSettings, top: int) -> list[Run]:
    """Baseline плюс `top` лучших runs, отсортированные от худшего к лучшему."""
    exp = c.get_experiment_by_name(settings.winprob_experiment)
    if exp is None:
        raise SystemExit("no training runs yet, run `make train` first")
    runs = c.search_runs(
        [exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{PRIMARY_METRIC} DESC"],
    )
    best = runs[:top]
    baseline = next((r for r in runs if r.data.tags.get("variant") == "baseline_logreg"), None)
    chosen = [*best, baseline] if baseline and baseline not in best else best
    return sorted(chosen, key=lambda r: r.data.metrics[PRIMARY_METRIC])


def registered_version(c: MlflowClient, name: str, run_id: str) -> ModelVersion | None:
    versions = c.search_model_versions(f"name = '{name}' and run_id = '{run_id}'")
    return versions[0] if versions else None


def register(settings: PipelineSettings, top: int) -> list[ModelVersion]:
    c = client(settings)
    name = settings.winprob_model
    if not c.search_registered_models(f"name = '{name}'"):
        c.create_registered_model(
            name,
            description="Вероятность победы seat 0 по снимку партии Kaggriculture. "
            "Вход: признаки farmvalue.features.FEATURES, выход: predict_proba.",
        )

    versions = []
    for run in candidates(c, settings, top):
        version = registered_version(c, name, run.info.run_id)
        if version is None:
            model = c.search_logged_models(
                experiment_ids=[run.info.experiment_id],
                filter_string=f"source_run_id = '{run.info.run_id}'",
            )[0]
            version = mlflow.register_model(f"models:/{model.model_id}", name)
            tags = run.data.tags
            for key, value in {
                "variant": tags["variant"],
                "dataset_version": tags["dataset_version"],
                PRIMARY_METRIC: f"{run.data.metrics[PRIMARY_METRIC]:.4f}",
            }.items():
                c.set_model_version_tag(name, version.version, key, value)
            c.update_model_version(
                name, version.version, description=tags.get("mlflow.note.content", "")
            )
        log.info("%s v%s <- run %s", name, version.version, run.info.run_id)
        versions.append(version)

    if len(versions) >= 1:
        c.set_registered_model_alias(name, CHAMPION, versions[-1].version)
    if len(versions) >= 2:
        c.set_registered_model_alias(name, CHALLENGER, versions[-2].version)
    return versions


def promote(settings: PipelineSettings, version: str, alias: str) -> None:
    c = client(settings)
    c.set_registered_model_alias(settings.winprob_model, alias, version)
    log.info("%s@%s -> v%s", settings.winprob_model, alias, version)


def show(settings: PipelineSettings) -> None:
    c = client(settings)
    name = settings.winprob_model
    aliases = c.get_registered_model(name).aliases
    by_version: dict[str, list[str]] = {}
    for alias, version in aliases.items():
        by_version.setdefault(str(version), []).append(alias)
    for v in sorted(c.search_model_versions(f"name = '{name}'"), key=lambda v: int(v.version)):
        print(
            f"v{v.version:<3} {v.tags.get('variant', ''):<16} "
            f"{v.tags.get('dataset_version', ''):<3} {PRIMARY_METRIC}={v.tags.get(PRIMARY_METRIC)} "
            f"run={v.run_id} {','.join(by_version.get(v.version, []))}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    reg = sub.add_parser("register")
    reg.add_argument("--top", type=int, default=2)
    prom = sub.add_parser("promote")
    prom.add_argument("version")
    prom.add_argument("--alias", default=CHAMPION)
    sub.add_parser("show")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    match args.command:
        case "register":
            register(settings, args.top)
            show(settings)
        case "promote":
            promote(settings, args.version, args.alias)
            show(settings)
        case "show":
            show(settings)


if __name__ == "__main__":
    main()
