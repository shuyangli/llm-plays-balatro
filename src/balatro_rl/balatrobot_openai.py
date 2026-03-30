from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

STATE_TO_CALLS: dict[str, tuple[str, ...]] = {
    "MENU": ("start_run",),
    "BLIND_SELECT": ("skip_or_select_blind", "sell_joker", "sell_consumable", "use_consumable"),
    "SELECTING_HAND": (
        "play_hand_or_discard",
        "rearrange_hand",
        "sell_joker",
        "sell_consumable",
        "use_consumable",
    ),
    "ROUND_EVAL": ("cash_out", "sell_joker", "sell_consumable", "use_consumable"),
    "SHOP": ("shop", "sell_joker", "sell_consumable", "use_consumable"),
    "GAME_OVER": ("go_to_menu",),
}

STATE_VALUE_TO_NAME = {
    1: "SELECTING_HAND",
    4: "GAME_OVER",
    5: "SHOP",
    7: "BLIND_SELECT",
    8: "ROUND_EVAL",
    11: "MENU",
}

SHOP_ACTIONS_REQUIRING_INDEX = {"buy_card", "buy_and_use_card", "redeem_voucher"}
SHOP_ACTIONS = {"next_round", "buy_card", "buy_and_use_card", "reroll", "redeem_voucher"}
BLIND_ACTIONS = {"select", "skip"}
HAND_ACTIONS = {"play_hand", "discard"}


@dataclass(slots=True)
class BotConfig:
    model: str
    deck: str
    stake: int
    seed: str | None
    max_turns: int
    port: int
    log_dir: Path


@dataclass(slots=True)
class RunLogger:
    path: Path

    def log(self, event_type: str, payload: dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event_type,
            **payload,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def build_log_path(log_dir: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return log_dir / f"balatrobot-openai-{timestamp}.jsonl"


def normalize_state_name(game_state: dict[str, Any]) -> str:
    state_enum = game_state.get("state_enum")
    if isinstance(state_enum, str) and state_enum:
        return state_enum
    state_value = game_state.get("state")
    if isinstance(state_value, int) and state_value in STATE_VALUE_TO_NAME:
        return STATE_VALUE_TO_NAME[state_value]
    raise ValueError(f"Unsupported BalatroBot state payload: {game_state.get('state')!r}")


def allowed_calls_for_state(game_state: dict[str, Any]) -> tuple[str, ...]:
    state_name = normalize_state_name(game_state)
    try:
        return STATE_TO_CALLS[state_name]
    except KeyError as error:
        raise ValueError(f"No allowed calls configured for state {state_name!r}.") from error


def _find_first_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        raise ValueError("Model response did not contain a JSON object.")

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError("Model response contained an unterminated JSON object.")


def parse_model_command(text: str) -> dict[str, Any]:
    payload = json.loads(_find_first_json_object(text))
    if not isinstance(payload, dict):
        raise ValueError("Model response JSON must be an object.")
    return payload


def _require_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected non-empty string for {field_name}.")
    return value


def _require_int(value: Any, *, field_name: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"Expected integer for {field_name}.")
    return value


def _require_index_list(value: Any, *, field_name: str, min_size: int, max_size: int | None = None) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Expected non-empty list for {field_name}.")
    if not all(isinstance(item, int) for item in value):
        raise ValueError(f"Expected {field_name} to contain only integers.")
    if len(value) < min_size:
        raise ValueError(f"Expected at least {min_size} entries for {field_name}.")
    if max_size is not None and len(value) > max_size:
        raise ValueError(f"Expected at most {max_size} entries for {field_name}.")
    return value


def validate_command(command: dict[str, Any], game_state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    allowed_calls = allowed_calls_for_state(game_state)
    name = _require_string(command.get("name"), field_name="name")
    if name not in allowed_calls:
        raise ValueError(f"{name!r} is not valid in state {normalize_state_name(game_state)!r}.")

    arguments = command.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError("Expected 'arguments' to be an object.")

    if name == "start_run":
        deck = _require_string(arguments.get("deck"), field_name="arguments.deck")
        stake = _require_int(arguments.get("stake"), field_name="arguments.stake")
        validated: dict[str, Any] = {"deck": deck, "stake": stake}
        if "seed" in arguments and arguments["seed"] is not None:
            validated["seed"] = _require_string(arguments["seed"], field_name="arguments.seed")
        if "challenge" in arguments and arguments["challenge"] is not None:
            validated["challenge"] = _require_string(arguments["challenge"], field_name="arguments.challenge")
        if "log_path" in arguments and arguments["log_path"] is not None:
            validated["log_path"] = _require_string(arguments["log_path"], field_name="arguments.log_path")
        return name, validated

    if name == "skip_or_select_blind":
        action = _require_string(arguments.get("action"), field_name="arguments.action")
        if action not in BLIND_ACTIONS:
            raise ValueError(f"Invalid blind action {action!r}.")
        return name, {"action": action}

    if name == "play_hand_or_discard":
        action = _require_string(arguments.get("action"), field_name="arguments.action")
        if action not in HAND_ACTIONS:
            raise ValueError(f"Invalid hand action {action!r}.")
        cards = _require_index_list(arguments.get("cards"), field_name="arguments.cards", min_size=1, max_size=5)
        return name, {"action": action, "cards": cards}

    if name == "rearrange_hand":
        cards = _require_index_list(arguments.get("cards"), field_name="arguments.cards", min_size=1)
        return name, {"cards": cards}

    if name == "cash_out":
        return name, {}

    if name == "shop":
        action = _require_string(arguments.get("action"), field_name="arguments.action")
        if action not in SHOP_ACTIONS:
            raise ValueError(f"Invalid shop action {action!r}.")
        validated = {"action": action}
        if action in SHOP_ACTIONS_REQUIRING_INDEX:
            validated["index"] = _require_int(arguments.get("index"), field_name="arguments.index")
        return name, validated

    if name in {"sell_joker", "sell_consumable", "use_consumable"}:
        return name, {"index": _require_int(arguments.get("index"), field_name="arguments.index")}

    if name == "go_to_menu":
        return name, {}

    raise ValueError(f"Validation is not implemented for call {name!r}.")


def build_turn_prompt(game_state: dict[str, Any], config: BotConfig, previous_error: str | None) -> str:
    allowed_calls = allowed_calls_for_state(game_state)
    prompt = {
        "objective": "Choose the single best next BalatroBot API call for this turn.",
        "constraints": [
            "Return exactly one JSON object and no surrounding prose.",
            "Use one of the allowed function names for the current state.",
            "Card, consumable, joker, voucher, and shop indices are 0-based.",
            "Do not invent fields beyond name and arguments.",
            "Prefer legal, conservative actions over risky guesses.",
        ],
        "run_defaults": {
            "deck": config.deck,
            "stake": config.stake,
            "seed": config.seed,
        },
        "allowed_calls": allowed_calls,
        "response_shape": {
            "name": "BalatroBot function name",
            "arguments": {"field": "value"},
        },
        "examples": [
            {"name": "start_run", "arguments": {"deck": config.deck, "stake": config.stake, "seed": config.seed}},
            {"name": "skip_or_select_blind", "arguments": {"action": "select"}},
            {"name": "play_hand_or_discard", "arguments": {"action": "play_hand", "cards": [0, 1, 2, 3, 4]}},
            {"name": "shop", "arguments": {"action": "next_round"}},
        ],
        "previous_error": previous_error,
        "game_state": game_state,
    }
    return json.dumps(prompt, sort_keys=True)


def choose_command(game_state: dict[str, Any], config: BotConfig, previous_error: str | None) -> tuple[str, dict[str, Any], str]:
    try:
        from openai import OpenAI
    except ImportError as error:  # pragma: no cover
        raise SystemExit("Missing dependency: install the `openai` package before running this script.") from error

    client = OpenAI()
    response = client.responses.create(
        model=config.model,
        input=[
            {
                "role": "system",
                "content": (
                    "You are controlling a BalatroBot client. "
                    "Reply with exactly one JSON object containing `name` and `arguments`."
                ),
            },
            {"role": "user", "content": build_turn_prompt(game_state, config, previous_error)},
        ],
    )
    raw_output = response.output_text
    parsed = parse_model_command(raw_output)
    name, arguments = validate_command(parsed, game_state)
    return name, arguments, raw_output


def run_bot(config: BotConfig) -> None:
    try:
        from balatrobot.client import BalatroClient
    except ImportError as error:  # pragma: no cover
        raise SystemExit("Missing dependency: install the `balatrobot` package before running this script.") from error

    logger = RunLogger(build_log_path(config.log_dir))
    logger.log(
        "run_started",
        {
            "model": config.model,
            "deck": config.deck,
            "stake": config.stake,
            "seed": config.seed,
            "max_turns": config.max_turns,
            "port": config.port,
        },
    )
    print(f"[log] {logger.path}")
    with BalatroClient(port=config.port) as client:
        previous_error: str | None = None
        for turn_index in range(config.max_turns):
            game_state = client.send_message("get_game_state")
            state_name = normalize_state_name(game_state)
            logger.log(
                "game_state",
                {
                    "turn": turn_index + 1,
                    "state": state_name,
                    "payload": game_state,
                },
            )
            if state_name == "GAME_OVER":
                logger.log("run_finished", {"turn": turn_index + 1, "reason": "game_over"})
                print("Game over.")
                return

            for attempt in range(3):
                name, arguments, raw_output = choose_command(game_state, config, previous_error)
                logger.log(
                    "model_choice",
                    {
                        "turn": turn_index + 1,
                        "attempt": attempt + 1,
                        "state": state_name,
                        "name": name,
                        "arguments": arguments,
                        "raw_output": raw_output,
                        "previous_error": previous_error,
                    },
                )
                print(f"[turn {turn_index + 1} attempt {attempt + 1}] {name} {json.dumps(arguments, sort_keys=True)}")
                print(f"[model] {raw_output}")
                try:
                    result = client.send_message(name, arguments)
                    logger.log(
                        "api_result",
                        {
                            "turn": turn_index + 1,
                            "attempt": attempt + 1,
                            "name": name,
                            "arguments": arguments,
                            "result": result,
                        },
                    )
                    previous_error = None
                    break
                except Exception as error:  # pragma: no cover
                    previous_error = str(error)
                    logger.log(
                        "api_error",
                        {
                            "turn": turn_index + 1,
                            "attempt": attempt + 1,
                            "name": name,
                            "arguments": arguments,
                            "error": previous_error,
                        },
                    )
                    print(f"[api-error] {previous_error}")
            else:
                logger.log(
                    "run_finished",
                    {
                        "turn": turn_index + 1,
                        "reason": "command_retry_exhausted",
                        "state": state_name,
                    },
                )
                raise RuntimeError(f"Failed to produce a valid API call after 3 attempts in state {state_name}.")
        logger.log("run_finished", {"turn": config.max_turns, "reason": "max_turns_reached"})
        raise RuntimeError(f"Stopped after hitting max_turns={config.max_turns}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a BalatroBot client driven by the OpenAI Responses API.")
    parser.add_argument("--model", default="gpt-5-mini", help="OpenAI model name to use.")
    parser.add_argument("--deck", default="Red Deck", help="Deck name for start_run.")
    parser.add_argument("--stake", type=int, default=1, help="Stake level for start_run.")
    parser.add_argument("--seed", default=None, help="Optional Balatro seed for reproducible runs.")
    parser.add_argument("--port", type=int, default=12346, help="BalatroBot TCP port.")
    parser.add_argument("--max-turns", type=int, default=250, help="Maximum API turns before aborting.")
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory where JSONL run histories are written.",
    )
    args = parser.parse_args()

    config = BotConfig(
        model=args.model,
        deck=args.deck,
        stake=args.stake,
        seed=args.seed,
        max_turns=args.max_turns,
        port=args.port,
        log_dir=Path(args.log_dir),
    )
    run_bot(config)


if __name__ == "__main__":
    main()
