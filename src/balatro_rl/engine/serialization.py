from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

from balatro_rl.engine.models import (
    Action,
    Card,
    EnvironmentConfig,
    GameState,
    JokerInstance,
    JokerTemplate,
    RunSnapshot,
    ShopItem,
)

SNAPSHOT_VERSION = 1


def to_primitive(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        data: dict[str, Any] = {}
        for field in dataclasses.fields(value):
            data[field.name] = to_primitive(getattr(value, field.name))
        return data
    if isinstance(value, dict):
        return {str(key): to_primitive(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [to_primitive(item) for item in value]
    if isinstance(value, list):
        return [to_primitive(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(to_primitive(value), sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def snapshot_hash(
    config: EnvironmentConfig,
    state: GameState,
    rng_state: dict[str, Any],
    reward_state: dict[str, Any],
) -> str:
    payload = {
        "version": SNAPSHOT_VERSION,
        "config": to_primitive(config),
        "state": to_primitive(state),
        "rng_state": rng_state,
        "reward_state": reward_state,
    }
    return stable_hash(payload)


def build_snapshot(
    config: EnvironmentConfig,
    state: GameState,
    rng_state: dict[str, Any],
    reward_state: dict[str, Any],
) -> RunSnapshot:
    hash_value = snapshot_hash(config, state, rng_state, reward_state)
    return RunSnapshot(
        version=SNAPSHOT_VERSION,
        config=to_primitive(config),
        state=to_primitive(state),
        rng_state=to_primitive(rng_state),
        reward_state=to_primitive(reward_state),
        snapshot_hash=hash_value,
    )


def action_to_dict(action: Action) -> dict[str, Any]:
    return to_primitive(action)


def environment_config_from_dict(data: dict[str, Any]) -> EnvironmentConfig:
    return EnvironmentConfig(**data)


def game_state_from_dict(data: dict[str, Any]) -> GameState:
    return GameState(
        run_id=data["run_id"],
        phase=data["phase"],
        step_index=int(data["step_index"]),
        ante=int(data["ante"]),
        blind_index=int(data["blind_index"]),
        max_ante=int(data["max_ante"]),
        score=int(data["score"]),
        score_target=int(data["score_target"]),
        money=int(data["money"]),
        hands_remaining=int(data["hands_remaining"]),
        discards_remaining=int(data["discards_remaining"]),
        reroll_cost=int(data["reroll_cost"]),
        draw_pile=[Card(**card) for card in data["draw_pile"]],
        hand=[Card(**card) for card in data["hand"]],
        discard_pile=[Card(**card) for card in data["discard_pile"]],
        played_cards=[Card(**card) for card in data["played_cards"]],
        jokers=[JokerInstance(**joker) for joker in data["jokers"]],
        shop_inventory=[
            ShopItem(
                id=item["id"],
                name=item["name"],
                item_type=item["item_type"],
                price=int(item["price"]),
                payload=JokerTemplate(**item["payload"]),
            )
            for item in data["shop_inventory"]
        ],
        event_log=list(data["event_log"]),
        shop_generation=int(data["shop_generation"]),
        next_entity_id=int(data["next_entity_id"]),
        pending_blind_reward=int(data["pending_blind_reward"]),
        terminal_outcome=data["terminal_outcome"],
    )
