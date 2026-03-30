from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Sequence

from balatro_rl.engine.core import GameEngine
from balatro_rl.engine.models import Action, EnvironmentConfig, Observation
from balatro_rl.engine.serialization import to_primitive

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    gym = None
    spaces = None


class BalatroEnv:
    def __init__(self, config: EnvironmentConfig | None = None) -> None:
        self.engine = GameEngine(config=config)

    def reset(self, config_override: dict | None = None):
        return self.engine.reset(config_override=config_override)

    def step(self, action: Action):
        return self.engine.step(action)

    def legal_actions(self) -> list[Action]:
        return self.engine.legal_actions()

    def snapshot(self):
        return self.engine.snapshot()

    def restore(self, snapshot) -> None:
        self.engine.restore(snapshot)

    def replay(self, events: Iterable[Action]):
        return self.engine.replay(events)

    def observe(self) -> Observation:
        return self.engine.observe()


class GymnasiumBalatroEnv((gym.Env if gym else object)):  # type: ignore[misc]
    metadata = {"render_modes": []}

    def __init__(self, config: EnvironmentConfig | None = None) -> None:
        self.base_env = BalatroEnv(config=config)
        self.max_actions = 256
        self.action_space = spaces.Discrete(self.max_actions) if spaces else None
        self.observation_space = None
        self._current_legal_actions: list[Action] = []

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        override = dict(options or {})
        if seed is not None:
            override["seed"] = seed
        observation, info = self.base_env.reset(config_override=override or None)
        self._current_legal_actions = self.base_env.legal_actions()
        return self._convert_observation(observation), info

    def step(self, action_index: int):
        legal = self.base_env.legal_actions()
        self._current_legal_actions = legal
        if action_index < 0 or action_index >= len(legal):
            action = Action.create("noop")
        else:
            action = legal[action_index]
        result = self.base_env.step(action)
        self._current_legal_actions = self.base_env.legal_actions()
        return (
            self._convert_observation(result.observation),
            result.reward,
            result.terminated,
            result.truncated,
            result.info,
        )

    def legal_action_map(self) -> list[Action]:
        self._current_legal_actions = self.base_env.legal_actions()
        return list(self._current_legal_actions)

    def _convert_observation(self, observation: Observation):
        if self.base_env.engine.config.observation_mode == "tensor":
            return tensorize_observation(observation)
        return to_primitive(observation)


class EpisodeRunner:
    def __init__(self, env: BalatroEnv) -> None:
        self.env = env

    def run(
        self,
        actions: Sequence[Action] | None = None,
        policy: Callable[[Observation, list[Action]], Action] | None = None,
        trace_path: str | None = None,
        max_steps: int = 100,
    ) -> list[dict[str, object]]:
        if actions is None and policy is None:
            raise ValueError("Either actions or policy must be provided")

        trace: list[dict[str, object]] = []
        observation, info = self.env.reset()
        trace.append({"type": "reset", "observation": to_primitive(observation), "info": info})

        action_iter = iter(actions or [])
        for _ in range(max_steps):
            legal = self.env.legal_actions()
            if not legal:
                break
            if policy is not None:
                action = policy(observation, legal)
            else:
                try:
                    action = next(action_iter)
                except StopIteration:
                    break
            result = self.env.step(action)
            trace.append(
                {
                    "type": "step_result",
                    "action": to_primitive(action),
                    "reward": result.reward,
                    "transition_metrics": result.info["transition_metrics"],
                    "observation": to_primitive(result.observation),
                    "info": result.info,
                }
            )
            observation = result.observation
            if result.terminated or result.truncated:
                trace.append(
                    {
                        "type": "terminal_summary",
                        "snapshot_hash": result.info["snapshot_hash"],
                        "reward": result.reward,
                        "terminal": result.terminated,
                        "truncated": result.truncated,
                    }
                )
                break

        if trace_path:
            path = Path(trace_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                for record in trace:
                    handle.write(json.dumps(record, sort_keys=True))
                    handle.write("\n")
        return trace


def tensorize_observation(observation: Observation) -> dict[str, object]:
    suit_to_index = {"spades": 0, "hearts": 1, "clubs": 2, "diamonds": 3}
    hand_ranks = [card["rank"] for card in observation.zones["hand"]]
    hand_suits = [suit_to_index[card["suit"]] for card in observation.zones["hand"]]
    while len(hand_ranks) < 8:
        hand_ranks.append(0)
        hand_suits.append(0)
    return {
        "scalars": [
            observation.player_state["money"],
            observation.player_state["hands_remaining"],
            observation.player_state["discards_remaining"],
            observation.player_state["score"],
            observation.player_state["score_target"],
            observation.player_state["ante"],
            observation.player_state["blind_index"],
            len(observation.zones["jokers"]),
        ],
        "hand_ranks": hand_ranks,
        "hand_suits": hand_suits,
        "legal_mask": [1 if value else 0 for value in observation.action_mask],
    }
