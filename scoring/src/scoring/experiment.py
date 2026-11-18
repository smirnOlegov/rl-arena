"""Эксперимент: конвейер с ограничениями и без них, второй воркер подключается по ходу прогона.

Запуск:
    uv run --env-file .env python -m scoring.experiment run bounded
    uv run --env-file .env python -m scoring.experiment run unbounded
    uv run --env-file .env python -m scoring.experiment plot
"""

import argparse
import asyncio
import csv
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import matplotlib.pyplot as plt
import pandas as pd
import psutil
from sqlalchemy import delete, select

from scoring.config import Settings, get_settings
from scoring.db import TaskRow, create_schema, make_engine, make_sessionmaker

MANAGEMENT = "http://localhost:15672/api"
RUNS = Path("runs")
MODES: dict[str, dict[str, str]] = {
    "bounded": {"QUEUE_NAME": "exp-bounded"},
    "unbounded": {"QUEUE_NAME": "exp-unbounded", "QUEUE_MAX_LENGTH": "0", "PREFETCH_COUNT": "0"},
}


def spawn(args: list[str], env: dict[str, str], log: Path) -> subprocess.Popen[bytes]:
    with log.open("wb") as fh:
        return subprocess.Popen([sys.executable, *args], env=env, stdout=fh, stderr=fh)  # noqa: S603


def start_worker(port: int, env: dict[str, str], out: Path) -> subprocess.Popen[bytes]:
    return spawn(
        ["-m", "uvicorn", "--factory", "scoring.worker:create_app", "--port", str(port)],
        env,
        out / f"worker-{port}.log",
    )


def wait_healthy(port: int) -> None:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/healthz").status_code == 204:
                return
        except httpx.TransportError:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"worker on port {port} did not start")


def rss_mb(proc: subprocess.Popen[bytes]) -> float:
    if proc.poll() is not None:
        return 0.0
    try:
        return float(psutil.Process(proc.pid).memory_info().rss) / 2**20
    except psutil.NoSuchProcess:
        return 0.0


def stop(procs: list[subprocess.Popen[bytes]]) -> None:
    for proc in procs:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
    for proc in procs:
        try:
            proc.wait(timeout=45)
        except subprocess.TimeoutExpired:
            proc.kill()


async def clear_results(settings: Settings) -> None:
    engine = make_engine(settings.database_url)
    await create_schema(engine)
    async with make_sessionmaker(engine)() as session:
        await session.execute(delete(TaskRow))
        await session.commit()
    await engine.dispose()


async def latencies(settings: Settings, started: float) -> list[tuple[float, float]]:
    engine = make_engine(settings.database_url)
    async with make_sessionmaker(engine)() as session:
        rows = await session.execute(select(TaskRow.enqueued_at, TaskRow.finished_at))
    await engine.dispose()
    return [
        (finished.timestamp() - started, (finished - enqueued).total_seconds())
        for enqueued, finished in rows
        if finished is not None
    ]


def run(mode: str, duration: int, second_worker_at: int) -> None:
    os.environ.update(MODES[mode])
    env = dict(os.environ)
    settings = get_settings()
    out = RUNS / mode
    out.mkdir(parents=True, exist_ok=True)
    client = httpx.Client(auth=("guest", "guest"))
    for name in (settings.queue_name, f"{settings.queue_name}.dlq"):
        client.delete(f"{MANAGEMENT}/queues/%2F/{name}")
    asyncio.run(clear_results(settings))

    workers = [start_worker(8001, env, out)]
    wait_healthy(8001)
    source = spawn(["-m", "scoring.source"], env, out / "source.log")
    started = time.time()
    samples: list[dict[str, Any]] = []
    last = {"publish": 0, "ack": 0}
    last_t, last_depth = started, 0
    try:
        while (elapsed := time.time() - started) < duration:
            if elapsed >= second_worker_at and len(workers) == 1:
                workers.append(start_worker(8002, env, out))
            queue = client.get(f"{MANAGEMENT}/queues/%2F/{settings.queue_name}").json()
            stats = queue.get("message_stats", {})
            counts = {key: stats.get(key, 0) for key in last}
            now, depth = time.time(), queue.get("messages", 0)
            ack_rate = (counts["ack"] - last["ack"]) / (now - last_t)
            samples.append(
                {
                    "t": round(now - started, 1),
                    "depth": depth,
                    "ready": queue.get("messages_ready", 0),
                    "unacked": queue.get("messages_unacknowledged", 0),
                    "attempt_rate": (counts["publish"] - last["publish"]) / (now - last_t),
                    "source_rate": ack_rate + (depth - last_depth) / (now - last_t),
                    "ack_rate": ack_rate,
                    "consumers": queue.get("consumers", 0),
                    "source_mb": rss_mb(source),
                    "worker1_mb": rss_mb(workers[0]),
                    "worker2_mb": rss_mb(workers[1]) if len(workers) > 1 else 0.0,
                    "rabbit_mb": client.get(f"{MANAGEMENT}/nodes").json()[0]["mem_used"] / 2**20,
                }
            )
            last, last_t, last_depth = counts, now, depth
            time.sleep(max(0.0, 1 - (time.time() - started - elapsed)))
    finally:
        stop([source, *workers])

    with (out / "metrics.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    pd.DataFrame(asyncio.run(latencies(settings, started)), columns=["t", "latency_s"]).sort_values(
        "t"
    ).to_csv(out / "latency.csv", index=False)


def summary(metrics: pd.DataFrame, latency: pd.DataFrame, split: float) -> dict[str, float]:
    one = metrics[(metrics.t > 20) & (metrics.t < split)]
    two = metrics[metrics.t > split + 20]
    return {
        "max_depth": metrics.depth.max(),
        "source_rate_1w": one.source_rate.mean(),
        "ack_rate_1w": one.ack_rate.mean(),
        "source_rate_2w": two.source_rate.mean(),
        "ack_rate_2w": two.ack_rate.mean(),
        "max_source_mb": metrics.source_mb.max(),
        "max_worker_mb": metrics.worker1_mb.max(),
        "max_rabbit_mb": metrics.rabbit_mb.max(),
        "latency_p50_s": latency.latency_s.median(),
        "latency_p95_s": latency.latency_s.quantile(0.95),
        "latency_max_s": latency.latency_s.max(),
    }


def plot(second_worker_at: int, out: Path) -> None:
    panels = [
        ("Глубина очереди, сообщений", ["depth", "unacked"]),
        ("Скорость, сообщений/с", ["attempt_rate", "source_rate", "ack_rate"]),
        ("Память, МБ", ["source_mb", "worker1_mb", "worker2_mb", "rabbit_mb"]),
    ]
    fig, axes = plt.subplots(4, len(MODES), figsize=(13, 13), sharex=True)
    rows = []
    for col, mode in enumerate(MODES):
        metrics = pd.read_csv(RUNS / mode / "metrics.csv")
        rates = ["attempt_rate", "source_rate", "ack_rate"]
        # Management API отдаёт счётчики раз в 5 с, поэтому скорости сглаживаются окном 5 с
        metrics[rates] = metrics[rates].rolling(5, min_periods=1).mean()
        latency = pd.read_csv(RUNS / mode / "latency.csv")
        rows.append(summary(metrics, latency, second_worker_at) | {"mode": mode})
        axes[0][col].set_title(mode)
        for row, (title, columns) in enumerate(panels):
            ax = axes[row][col]
            for column in columns:
                ax.plot(metrics.t, metrics[column], label=column)
            ax.set_ylabel(title)
            ax.legend(fontsize=8)
        ax = axes[3][col]
        ax.scatter(latency.t, latency.latency_s, s=2, alpha=0.4)
        ax.set_ylabel("Время задачи, с (по моменту завершения)")
        ax.set_xlabel("секунды от старта")
        for ax in axes[:, col]:
            ax.axvline(second_worker_at, color="grey", linestyle="--", linewidth=1)
            ax.grid(alpha=0.3)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "experiment.png", dpi=110)
    print(pd.DataFrame(rows).set_index("mode").T.round(1).to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run_cmd = sub.add_parser("run")
    run_cmd.add_argument("mode", choices=list(MODES))
    run_cmd.add_argument("--duration", type=int, default=120)
    run_cmd.add_argument("--second-worker-at", type=int, default=60)
    plot_cmd = sub.add_parser("plot")
    plot_cmd.add_argument("--second-worker-at", type=int, default=60)
    plot_cmd.add_argument("--out", type=Path, default=Path("docs"))
    args = parser.parse_args()
    if args.command == "run":
        run(args.mode, args.duration, args.second_worker_at)
    else:
        plot(args.second_worker_at, args.out)


if __name__ == "__main__":
    main()
