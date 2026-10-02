"""Агенты, матчи и защита от утечки эвала."""

import uuid


async def test_register_agent(make_agent) -> None:
    agent = await make_agent(kind="sft", name="sft-v1")
    assert agent["name"] == "sft-v1"
    assert agent["pool"] == "training"
    assert agent["active"] is True


async def test_register_duplicate_name_conflicts(client, make_agent) -> None:
    await make_agent(name="dup")
    response = await client.post(
        "/api/v1/agents", json={"name": "dup", "kind": "main", "checkpoint_uri": "s3://x"}
    )
    assert response.status_code == 409
    assert response.json()["error"] == "conflict"


async def test_register_with_missing_parent_is_404(client) -> None:
    response = await client.post(
        "/api/v1/agents",
        json={
            "name": "orphan",
            "kind": "main",
            "checkpoint_uri": "s3://x",
            "parent_id": str(uuid.uuid4()),
        },
    )
    assert response.status_code == 404
    assert response.json()["error"] == "not_found"


async def test_register_rejects_bad_name(client) -> None:
    response = await client.post(
        "/api/v1/agents", json={"name": "bad name!", "kind": "main", "checkpoint_uri": "s3://x"}
    )
    assert response.status_code == 422


async def test_get_and_list_agents_with_filters(client, make_agent) -> None:
    sft = await make_agent(kind="sft")
    await make_agent(kind="main", parent_id=sft["id"])
    await make_agent(kind="baseline", pool="evaluation")

    assert (await client.get(f"/api/v1/agents/{sft['id']}")).json()["id"] == sft["id"]
    assert len((await client.get("/api/v1/agents")).json()) == 3
    evaluators = (await client.get("/api/v1/agents", params={"pool": "evaluation"})).json()
    assert [a["kind"] for a in evaluators] == ["baseline"]
    mains = (await client.get("/api/v1/agents", params={"kind": "main", "active": True})).json()
    assert mains[0]["parent_id"] == sft["id"]


async def test_get_unknown_agent_is_404(client) -> None:
    assert (await client.get(f"/api/v1/agents/{uuid.uuid4()}")).status_code == 404


async def test_training_match_against_evaluator_is_rejected(client, make_agent) -> None:
    """Главный инвариант: против held-out агентов не учимся."""
    learner = await make_agent()
    evaluator = await make_agent(kind="baseline", pool="evaluation")
    for agent, opponent in ((learner, evaluator), (evaluator, learner)):
        response = await client.post(
            "/api/v1/matches",
            json={
                "agent_id": agent["id"],
                "opponent_id": opponent["id"],
                "result": "win",
                "purpose": "training",
            },
        )
        assert response.status_code == 409
        assert response.json()["error"] == "eval_leakage"


async def test_evaluation_match_requires_held_out_opponent(client, make_agent) -> None:
    learner, other = await make_agent(), await make_agent()
    response = await client.post(
        "/api/v1/matches",
        json={
            "agent_id": learner["id"],
            "opponent_id": other["id"],
            "result": "win",
            "purpose": "evaluation",
        },
    )
    assert response.status_code == 409


async def test_evaluator_cannot_be_evaluated(client, make_agent) -> None:
    first = await make_agent(kind="baseline", pool="evaluation")
    second = await make_agent(kind="baseline", pool="evaluation")
    response = await client.post(
        "/api/v1/matches",
        json={
            "agent_id": first["id"],
            "opponent_id": second["id"],
            "result": "loss",
            "purpose": "evaluation",
        },
    )
    assert response.status_code == 409


async def test_self_match_is_invalid(client, make_agent) -> None:
    agent = await make_agent()
    response = await client.post(
        "/api/v1/matches",
        json={
            "agent_id": agent["id"],
            "opponent_id": agent["id"],
            "result": "draw",
            "purpose": "training",
        },
    )
    assert response.status_code == 422


async def test_match_with_unknown_agent_is_404(client, make_agent) -> None:
    agent = await make_agent()
    response = await client.post(
        "/api/v1/matches",
        json={
            "agent_id": agent["id"],
            "opponent_id": str(uuid.uuid4()),
            "result": "win",
            "purpose": "training",
        },
    )
    assert response.status_code == 404


async def test_uploaded_agent_winrate_against_held_out_pool(
    client, make_agent, record_matches
) -> None:
    """Сценарий «залил агента — посмотрел винрейт»."""
    uploaded = await make_agent(kind="uploaded", name="my-submission")
    bot_a = await make_agent(kind="baseline", pool="evaluation", name="bot-a")
    bot_b = await make_agent(kind="baseline", pool="evaluation", name="bot-b")
    await record_matches(uploaded, bot_a, "win", "evaluation", count=6)
    await record_matches(uploaded, bot_a, "loss", "evaluation", count=2)
    await record_matches(uploaded, bot_b, "draw", "evaluation", count=2)

    report = (await client.get(f"/api/v1/agents/{uploaded['id']}/evaluation")).json()

    overall = report["overall"]
    assert (overall["games"], overall["wins"], overall["draws"], overall["losses"]) == (10, 6, 2, 2)
    assert overall["winrate"] == 0.7
    assert 0 < overall["ci_low"] < 0.7 < overall["ci_high"] < 1
    assert [o["opponent_name"] for o in report["opponents"]] == ["bot-a", "bot-b"]
    assert report["opponents"][1]["winrate"] == 0.5


async def test_evaluation_ignores_training_matches(client, make_agent, record_matches) -> None:
    learner, sparring = await make_agent(), await make_agent()
    await record_matches(learner, sparring, "win", "training", count=5)
    report = (await client.get(f"/api/v1/agents/{learner['id']}/evaluation")).json()
    assert report["overall"]["games"] == 0
    assert report["overall"]["winrate"] is None
    assert report["opponents"] == []
