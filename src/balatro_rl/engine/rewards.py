from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from balatro_rl.engine.models import EnvironmentConfig, TransitionMetrics


class RewardModel(Protocol):
    id: str

    def reset(self, run_config: EnvironmentConfig, initial_state: object) -> dict[str, object]:
        ...

    def on_step(
        self,
        prev_state: object,
        action: object,
        metrics: TransitionMetrics,
        next_state: object,
        reward_state: dict[str, object],
    ) -> tuple[float, dict[str, object], dict[str, object]]:
        ...

    def on_terminal(
        self,
        final_state: object,
        metrics: TransitionMetrics,
        reward_state: dict[str, object],
    ) -> tuple[float, dict[str, object], dict[str, object]]:
        ...


@dataclass(slots=True)
class BaseRewardModel:
    id: str
    params: dict[str, int | float | bool | str]

    def reset(self, run_config: EnvironmentConfig, initial_state: object) -> dict[str, object]:
        return {}

    def on_step(
        self,
        prev_state: object,
        action: object,
        metrics: TransitionMetrics,
        next_state: object,
        reward_state: dict[str, object],
    ) -> tuple[float, dict[str, object], dict[str, object]]:
        return 0.0, reward_state, {}

    def on_terminal(
        self,
        final_state: object,
        metrics: TransitionMetrics,
        reward_state: dict[str, object],
    ) -> tuple[float, dict[str, object], dict[str, object]]:
        return 0.0, reward_state, {}


class TerminalWinLossReward(BaseRewardModel):
    def on_terminal(
        self,
        final_state: object,
        metrics: TransitionMetrics,
        reward_state: dict[str, object],
    ) -> tuple[float, dict[str, object], dict[str, object]]:
        win_reward = float(self.params.get("win_reward", 1.0))
        loss_reward = float(self.params.get("loss_reward", 0.0))
        reward = win_reward if metrics.terminal_outcome == "win" else loss_reward
        return reward, reward_state, {"terminal_component": reward}


class TerminalScoreReward(BaseRewardModel):
    def on_terminal(
        self,
        final_state: object,
        metrics: TransitionMetrics,
        reward_state: dict[str, object],
    ) -> tuple[float, dict[str, object], dict[str, object]]:
        scale = float(self.params.get("scale", 0.001))
        score = getattr(final_state, "score", 0)
        reward = score * scale
        return reward, reward_state, {"terminal_score_component": reward}


class ScoreDeltaReward(BaseRewardModel):
    def on_step(
        self,
        prev_state: object,
        action: object,
        metrics: TransitionMetrics,
        next_state: object,
        reward_state: dict[str, object],
    ) -> tuple[float, dict[str, object], dict[str, object]]:
        scale = float(self.params.get("scale", 1.0))
        reward = float(metrics.score_delta) * scale
        return reward, reward_state, {"score_delta_component": reward}


class MoneyDeltaReward(BaseRewardModel):
    def on_step(
        self,
        prev_state: object,
        action: object,
        metrics: TransitionMetrics,
        next_state: object,
        reward_state: dict[str, object],
    ) -> tuple[float, dict[str, object], dict[str, object]]:
        scale = float(self.params.get("scale", 1.0))
        reward = float(metrics.money_delta) * scale
        return reward, reward_state, {"money_delta_component": reward}


class BlindProgressReward(BaseRewardModel):
    def on_step(
        self,
        prev_state: object,
        action: object,
        metrics: TransitionMetrics,
        next_state: object,
        reward_state: dict[str, object],
    ) -> tuple[float, dict[str, object], dict[str, object]]:
        blind_reward = float(self.params.get("blind_reward", 1.0))
        ante_reward = float(self.params.get("ante_reward", 2.0))
        reward = 0.0
        if metrics.blind_cleared:
            reward += blind_reward
        if metrics.ante_advanced:
            reward += ante_reward
        return reward, reward_state, {"blind_progress_component": reward}


class SurvivalReward(BaseRewardModel):
    def reset(self, run_config: EnvironmentConfig, initial_state: object) -> dict[str, object]:
        return {"steps": 0}

    def on_step(
        self,
        prev_state: object,
        action: object,
        metrics: TransitionMetrics,
        next_state: object,
        reward_state: dict[str, object],
    ) -> tuple[float, dict[str, object], dict[str, object]]:
        reward_state = dict(reward_state)
        reward_state["steps"] = int(reward_state.get("steps", 0)) + 1
        step_reward = float(self.params.get("step_reward", 0.01))
        blind_bonus = float(self.params.get("blind_bonus", 0.25)) if metrics.blind_cleared else 0.0
        reward = step_reward + blind_bonus
        return reward, reward_state, {"survival_component": reward, "survival_steps": reward_state["steps"]}

    def on_terminal(
        self,
        final_state: object,
        metrics: TransitionMetrics,
        reward_state: dict[str, object],
    ) -> tuple[float, dict[str, object], dict[str, object]]:
        penalty = float(self.params.get("loss_penalty", -1.0))
        reward = 0.0 if metrics.terminal_outcome == "win" else penalty
        return reward, reward_state, {"survival_terminal_component": reward}


class EfficiencyReward(BaseRewardModel):
    def on_step(
        self,
        prev_state: object,
        action: object,
        metrics: TransitionMetrics,
        next_state: object,
        reward_state: dict[str, object],
    ) -> tuple[float, dict[str, object], dict[str, object]]:
        reward = 0.0
        if metrics.blind_cleared:
            hands_left = getattr(next_state, "hands_remaining", 0)
            discards_left = getattr(next_state, "discards_remaining", 0)
            reward = hands_left * float(self.params.get("hand_weight", 0.25))
            reward += discards_left * float(self.params.get("discard_weight", 0.1))
            reward -= metrics.rerolls_spent * float(self.params.get("reroll_penalty", 0.5))
        return reward, reward_state, {"efficiency_component": reward}


class RewardRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, type[BaseRewardModel]] = {}

    def register(self, model_factory: type[BaseRewardModel]) -> None:
        self._factories[model_factory.id] = model_factory

    def create(
        self,
        reward_model: str,
        reward_params: dict[str, int | float | bool | str],
    ) -> RewardModel:
        if reward_model not in self._factories:
            raise KeyError(f"Unknown reward model: {reward_model}")
        return self._factories[reward_model](id=reward_model, params=reward_params)


def build_default_registry() -> RewardRegistry:
    registry = RewardRegistry()
    registry.register(type("TerminalWinLossFactory", (TerminalWinLossReward,), {"id": "terminal_win_loss"}))
    registry.register(type("TerminalScoreFactory", (TerminalScoreReward,), {"id": "terminal_score"}))
    registry.register(type("ScoreDeltaFactory", (ScoreDeltaReward,), {"id": "score_delta"}))
    registry.register(type("MoneyDeltaFactory", (MoneyDeltaReward,), {"id": "money_delta"}))
    registry.register(type("BlindProgressFactory", (BlindProgressReward,), {"id": "blind_progress"}))
    registry.register(type("SurvivalFactory", (SurvivalReward,), {"id": "survival"}))
    registry.register(type("EfficiencyFactory", (EfficiencyReward,), {"id": "efficiency"}))
    return registry
