"""Лига: PFSP-матчмейкинг, лидерборд, ресемплинг эксплоитеров."""

import uuid
from collections import Counter


async def test_opponent_pick_never_returns_evaluator(client, make_agent) -> None:
    learner = await make_agent()
    sparring = await make_agent()
    await make_agent(kind="baseline", pool="evaluation")
    for seed in range(30):
        pick = (
            await client.get(
                "/api/v1/league/opponent",
                params={"agent_id": learner["id"], "seed": seed, "weighting": "uniform"},
            )
        ).json()
        assert pick["opponent"]["id"] == sparring["id"]
        assert pick["opponent"]["pool"] == "training"


async def test_opponent_pick_skips_inactive_agents(client, make_agent) -> None:
    learner = await make_agent()
    await make_agent(kind="exploiter")
    response = await client.post(
        "/api/v1/league/exploiters/resample",
        json={"epoch": 5, "sft_agent_id": (await make_agent(kind="sft"))["id"]},
    )
    retired = response.json()["retired"]
    picks = {
        (
            await client.get(
                "/api/v1/league/opponent", params={"agent_id": learner["id"], "seed": seed}
            )
        ).json()["opponent"]["id"]
        for seed in range(30)
    }
    assert not picks & set(retired)


async def test_pfsp_hard_prefers_opponents_we_lose_to(client, make_agent, record_matches) -> None:
    learner = await make_agent(name="learner")
    strong = await make_agent(name="strong")
    weak = await make_agent(name="weak")
    await record_matches(learner, strong, "loss", "training", count=9)
    await record_matches(learner, strong, "win", "training", count=1)
    # матч, записанный от лица оппонента, тоже учитывается (в перевёрнутом виде)
    await record_matches(weak, learner, "loss", "training", count=10)

    picks = Counter()
    for seed in range(200):
        pick = (
            await client.get(
                "/api/v1/league/opponent",
                params={"agent_id": learner["id"], "seed": seed, "weighting": "hard"},
            )
        ).json()
        picks[pick["opponent"]["name"]] += 1
        if pick["opponent"]["name"] == "strong":
            assert pick["winrate_vs_opponent"] == 0.1
    assert picks["strong"] > 150


async def test_opponent_pick_is_reproducible_with_seed(client, make_agent) -> None:
    learner = await make_agent()
    for _ in range(5):
        await make_agent()
    params = {"agent_id": learner["id"], "seed": 7}
    first = (await client.get("/api/v1/league/opponent", params=params)).json()
    second = (await client.get("/api/v1/league/opponent", params=params)).json()
    assert first == second


async def test_opponent_pick_without_candidates_conflicts(client, make_agent) -> None:
    learner = await make_agent()
    response = await client.get("/api/v1/league/opponent", params={"agent_id": learner["id"]})
    assert response.status_code == 409


async def test_evaluator_does_not_get_training_opponents(client, make_agent) -> None:
    evaluator = await make_agent(kind="baseline", pool="evaluation")
    await make_agent()
    response = await client.get("/api/v1/league/opponent", params={"agent_id": evaluator["id"]})
    assert response.status_code == 409
    assert response.json()["error"] == "eval_leakage"


async def test_leaderboard_ranks_by_confidence_not_lucky_streak(
    client, make_agent, record_matches
) -> None:
    lucky = await make_agent(name="lucky")
    solid = await make_agent(name="solid")
    untested = await make_agent(name="untested")
    bot = await make_agent(kind="baseline", pool="evaluation")
    await record_matches(lucky, bot, "win", "evaluation", count=3)
    await record_matches(solid, bot, "win", "evaluation", count=40)
    await record_matches(solid, bot, "loss", "evaluation", count=10)

    board = (await client.get("/api/v1/league/leaderboard")).json()
    assert [row["agent"]["name"] for row in board] == ["solid", "lucky"]
    assert [row["rank"] for row in board] == [1, 2]
    assert untested["id"] not in {row["agent"]["id"] for row in board}

    top1 = (await client.get("/api/v1/league/leaderboard", params={"top": 1})).json()
    assert len(top1) == 1


async def test_resample_skipped_between_resample_epochs(client, make_agent) -> None:
    sft = await make_agent(kind="sft")
    body = (
        await client.post(
            "/api/v1/league/exploiters/resample", json={"epoch": 3, "sft_agent_id": sft["id"]}
        )
    ).json()
    assert body == {"epoch": 3, "resampled": False, "retired": [], "created": [], "jobs": []}


async def test_resample_rotates_exploiters_and_queues_jobs(app, client, make_agent) -> None:
    sft = await make_agent(kind="sft")

    first = (
        await client.post(
            "/api/v1/league/exploiters/resample", json={"epoch": 5, "sft_agent_id": sft["id"]}
        )
    ).json()
    assert first["resampled"] is True
    assert first["retired"] == []
    assert [e["name"] for e in first["created"]] == ["exploiter-e0005-0", "exploiter-e0005-1"]
    assert all(e["parent_id"] == sft["id"] for e in first["created"])
    assert all(e["checkpoint_uri"] == sft["checkpoint_uri"] for e in first["created"])

    second = (
        await client.post(
            "/api/v1/league/exploiters/resample", json={"epoch": 10, "sft_agent_id": sft["id"]}
        )
    ).json()
    assert sorted(second["retired"]) == sorted(e["id"] for e in first["created"])

    active = (
        await client.get("/api/v1/agents", params={"kind": "exploiter", "active": True})
    ).json()
    assert {e["id"] for e in active} == {e["id"] for e in second["created"]}

    assert await app.state.redis.xlen(app.state.settings.jobs_stream) == 4
    job = (await client.get(f"/api/v1/jobs/{second['jobs'][0]}")).json()
    assert job["kind"] == "exploiter_training"
    assert job["params"] == {"epoch": 10, "init_checkpoint": sft["checkpoint_uri"]}


async def test_resample_same_epoch_twice_conflicts(client, make_agent) -> None:
    sft = await make_agent(kind="sft")
    payload = {"epoch": 5, "sft_agent_id": sft["id"]}
    assert (await client.post("/api/v1/league/exploiters/resample", json=payload)).is_success
    repeat = await client.post("/api/v1/league/exploiters/resample", json=payload)
    assert repeat.status_code == 409
    active = (
        await client.get("/api/v1/agents", params={"kind": "exploiter", "active": True})
    ).json()
    assert len(active) == 2


async def test_resample_requires_training_sft_agent(client, make_agent) -> None:
    main = await make_agent(kind="main")
    response = await client.post(
        "/api/v1/league/exploiters/resample", json={"epoch": 5, "sft_agent_id": main["id"]}
    )
    assert response.status_code == 409
    missing = await client.post(
        "/api/v1/league/exploiters/resample", json={"epoch": 5, "sft_agent_id": str(uuid.uuid4())}
    )
    assert missing.status_code == 404
