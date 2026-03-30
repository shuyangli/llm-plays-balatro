from __future__ import annotations

import unittest

from balatro_rl import BalatroEnv
from balatro_rl.engine.env import GymnasiumBalatroEnv
from balatro_rl.engine.models import Action, EnvironmentConfig


def choose_simple_action(env: BalatroEnv) -> Action:
    observation = env.observe()
    legal = env.legal_actions()
    if observation.phase in {"blind_select", "shop"}:
        return Action.create("noop")

    play_actions = [action for action in legal if action.kind == "play_hand" and len(action.target_ids) == 5]
    if play_actions:
        return play_actions[0]

    play_actions = [action for action in legal if action.kind == "play_hand"]
    if play_actions:
        return play_actions[-1]

    return legal[0]


def run_steps(env: BalatroEnv, count: int) -> tuple[list[Action], list[float], list[str]]:
    actions: list[Action] = []
    rewards: list[float] = []
    hashes: list[str] = []
    for _ in range(count):
        legal = env.legal_actions()
        if not legal:
            break
        action = choose_simple_action(env)
        result = env.step(action)
        actions.append(action)
        rewards.append(result.reward)
        hashes.append(result.info["snapshot_hash"])
        if result.terminated or result.truncated:
            break
    return actions, rewards, hashes


class BalatroEnvTest(unittest.TestCase):
    def test_same_seed_same_actions_produce_identical_hashes(self) -> None:
        config = EnvironmentConfig(seed=7, reward_model="survival")
        env_one = BalatroEnv(config)
        env_two = BalatroEnv(config)
        env_one.reset()
        env_two.reset()

        rewards_one: list[float] = []
        rewards_two: list[float] = []
        hashes_one: list[str] = []
        hashes_two: list[str] = []

        for _ in range(25):
            action = choose_simple_action(env_one)
            result_one = env_one.step(action)
            result_two = env_two.step(action)
            rewards_one.append(result_one.reward)
            rewards_two.append(result_two.reward)
            hashes_one.append(result_one.info["snapshot_hash"])
            hashes_two.append(result_two.info["snapshot_hash"])
            if result_one.terminated or result_one.truncated:
                self.assertTrue(result_two.terminated or result_two.truncated)
                break

        self.assertEqual(rewards_one, rewards_two)
        self.assertEqual(hashes_one, hashes_two)
        self.assertEqual(env_one.engine.reward_state, env_two.engine.reward_state)

    def test_snapshot_restore_preserves_reward_state(self) -> None:
        env = BalatroEnv(EnvironmentConfig(seed=11, reward_model="survival"))
        env.reset()
        run_steps(env, 8)
        snapshot = env.snapshot()
        original_reward_state = dict(env.engine.reward_state)

        future_actions, future_rewards, future_hashes = run_steps(env, 5)

        restored_env = BalatroEnv()
        restored_env.restore(snapshot)
        replay_rewards: list[float] = []
        replay_hashes: list[str] = []
        for action in future_actions:
            result = restored_env.step(action)
            replay_rewards.append(result.reward)
            replay_hashes.append(result.info["snapshot_hash"])

        self.assertEqual(original_reward_state, snapshot.reward_state)
        self.assertEqual(future_rewards, replay_rewards)
        self.assertEqual(future_hashes, replay_hashes)
        self.assertEqual(env.engine.reward_state, restored_env.engine.reward_state)

    def test_replay_regenerates_original_hashes(self) -> None:
        env = BalatroEnv(EnvironmentConfig(seed=13, reward_model="survival"))
        env.reset()
        actions, rewards, hashes = run_steps(env, 12)

        replay = env.replay(actions)
        self.assertEqual(rewards, replay.rewards)
        self.assertEqual(hashes, replay.snapshot_hashes)

    def test_score_delta_reward_matches_transition_metrics(self) -> None:
        env = BalatroEnv(EnvironmentConfig(seed=3, reward_model="score_delta"))
        env.reset()
        env.step(Action.create("noop"))
        result = env.step(next(action for action in env.legal_actions() if action.kind == "play_hand" and len(action.target_ids) == 5))
        expected = float(result.info["transition_metrics"]["score_delta"])
        self.assertEqual(expected, result.reward)

    def test_invalid_action_mask_only_does_not_mutate_state(self) -> None:
        env = BalatroEnv(EnvironmentConfig(illegal_action_mode="mask_only"))
        env.reset()
        snapshot = env.snapshot()
        result = env.step(Action.create("play_hand"))
        self.assertEqual(0.0, result.reward)
        self.assertEqual(snapshot.snapshot_hash, result.info["snapshot_hash"])
        self.assertEqual(0, env.engine.state.step_index)
        self.assertIn("invalid_action", result.info)

    def test_shop_transition_and_buy_joker(self) -> None:
        env = BalatroEnv(EnvironmentConfig(seed=5, reward_model="money_delta"))
        env.reset()
        env.step(Action.create("noop"))
        env.engine.state.score_target = 10
        play_result = env.step(next(action for action in env.legal_actions() if action.kind == "play_hand" and len(action.target_ids) == 5))
        self.assertEqual("shop", play_result.observation.phase)
        buy_action = next(action for action in env.legal_actions() if action.kind == "buy_shop_item")
        buy_result = env.step(buy_action)
        self.assertEqual(1, len(env.engine.state.jokers))
        self.assertLess(buy_result.observation.player_state["money"], play_result.observation.player_state["money"])

    def test_observation_action_mask_matches_legal_actions(self) -> None:
        env = BalatroEnv()
        observation, _ = env.reset()
        self.assertEqual(len(observation.legal_actions), len(observation.action_mask))
        self.assertTrue(all(observation.action_mask))

    def test_gymnasium_wrapper_tensor_mode(self) -> None:
        wrapper = GymnasiumBalatroEnv(EnvironmentConfig(observation_mode="tensor"))
        observation, _ = wrapper.reset(seed=17)
        self.assertIn("scalars", observation)
        self.assertEqual(8, len(observation["hand_ranks"]))


if __name__ == "__main__":
    unittest.main()
