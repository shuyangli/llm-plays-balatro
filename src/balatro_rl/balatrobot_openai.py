from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

class PreprocessingError(Exception):
    """Raised when a candidate's preprocessing.py fails at load or call time."""


STATE_TO_CALLS: dict[str, tuple[str, ...]] = {
    "MENU": ("start",),
    "BLIND_SELECT": ("select", "skip"),
    "SELECTING_HAND": ("play", "discard"),
    "ROUND_EVAL": ("cash_out",),
    "SHOP": ("buy", "reroll", "next_round"),
    "SMODS_BOOSTER_OPENED": ("pack",),
    "GAME_OVER": ("menu",),
}

ALL_BALATROBOT_ACTIONS: tuple[str, ...] = (
    "add",
    "buy",
    "cash_out",
    "discard",
    "gamestate",
    "health",
    "load",
    "menu",
    "next_round",
    "pack",
    "play",
    "rearrange",
    "reroll",
    "save",
    "screenshot",
    "select",
    "sell",
    "set",
    "skip",
    "start",
    "use",
)

SUPPORTED_ACTION_SCHEMAS: dict[str, Any] = {
    "start": {
        "arguments": {
            "deck": "string",
            "stake": "string",
            "seed": "string | null",
            "challenge": "string | null",
            "log_path": "string | null",
        }
    },
    "select": {"arguments": {}},
    "skip": {"arguments": {}},
    "play": {"arguments": {"cards": "list[int] (1-5 cards)"}},
    "discard": {"arguments": {"cards": "list[int] (1-5 cards)"}},
    "cash_out": {"arguments": {}},
    "buy": {
        "arguments": {
            "card": "int | null",
            "voucher": "int | null",
            "pack": "int | null",
        },
        "rule": "Provide exactly one of card, voucher, or pack.",
    },
    "pack": {
        "arguments": {
            "card": "int | null",
        },
        "rule": "Provide the 0-based index of the chosen booster card.",
    },
    "reroll": {"arguments": {}},
    "next_round": {"arguments": {}},
    "menu": {"arguments": {}},
}

STATE_VALUE_TO_NAME = {
    1: "SELECTING_HAND",
    4: "GAME_OVER",
    5: "SHOP",
    7: "BLIND_SELECT",
    8: "ROUND_EVAL",
    11: "MENU",
}

VALID_STAKES = {
    "WHITE",
    "RED",
    "GREEN",
    "BLACK",
    "BLUE",
    "PURPLE",
    "ORANGE",
    "GOLD",
}

DECK_NAME_TO_CODE = {
    "red": "RED",
    "red deck": "RED",
    "blue": "BLUE",
    "blue deck": "BLUE",
    "yellow": "YELLOW",
    "yellow deck": "YELLOW",
    "green": "GREEN",
    "green deck": "GREEN",
    "black": "BLACK",
    "black deck": "BLACK",
    "magic": "MAGIC",
    "magic deck": "MAGIC",
    "nebula": "NEBULA",
    "nebula deck": "NEBULA",
    "ghost": "GHOST",
    "ghost deck": "GHOST",
    "abandoned": "ABANDONED",
    "abandoned deck": "ABANDONED",
    "checkered": "CHECKERED",
    "checkered deck": "CHECKERED",
    "zodiac": "ZODIAC",
    "zodiac deck": "ZODIAC",
    "painted": "PAINTED",
    "painted deck": "PAINTED",
    "anaglyph": "ANAGLYPH",
    "anaglyph deck": "ANAGLYPH",
    "plasma": "PLASMA",
    "plasma deck": "PLASMA",
    "erratic": "ERRATIC",
    "erratic deck": "ERRATIC",
}


@dataclass(slots=True)
class BotConfig:
    gateway_url: str
    function_name: str
    deck: str
    stake: str
    seed: str | None
    max_turns: int
    port: int
    log_dir: Path
    variant: str | None = None
    preprocessing_path: Path | None = None


def _load_preprocessing_fn(path: Path):
    """Load and return the preprocess(game_state) callable from a Python file."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("balatro_preprocessing", path)
    if spec is None or spec.loader is None:
        raise PreprocessingError(f"Cannot create module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception as exc:
        raise PreprocessingError(f"Error executing {path}: {exc}") from exc
    fn = getattr(module, "preprocess", None)
    if not callable(fn):
        raise PreprocessingError(
            f"{path} must define a callable 'preprocess(game_state: dict) -> dict'"
        )
    return fn


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
    if isinstance(state_value, str) and state_value:
        return state_value
    if isinstance(state_value, int) and state_value in STATE_VALUE_TO_NAME:
        return STATE_VALUE_TO_NAME[state_value]
    raise ValueError(
        f"Unsupported BalatroBot state payload: {game_state.get('state')!r}"
    )


def allowed_calls_for_state(game_state: dict[str, Any]) -> tuple[str, ...]:
    state_name = normalize_state_name(game_state)
    try:
        return STATE_TO_CALLS[state_name]
    except KeyError as error:
        raise ValueError(
            f"No allowed calls configured for state {state_name!r}."
        ) from error


def _require_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected non-empty string for {field_name}.")
    return value


def _require_int(value: Any, *, field_name: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"Expected integer for {field_name}.")
    return value


def _require_exactly_one_index(
    arguments: dict[str, Any], *, field_names: tuple[str, ...]
) -> dict[str, int]:
    present = {
        field_name: _require_int(arguments.get(field_name), field_name=f"arguments.{field_name}")
        for field_name in field_names
        if arguments.get(field_name) is not None
    }
    if len(present) != 1:
        joined = ", ".join(f"arguments.{field_name}" for field_name in field_names)
        raise ValueError(f"Expected exactly one of: {joined}.")
    return present


def normalize_stake(value: Any, *, field_name: str) -> str:
    if isinstance(value, str) and value:
        normalized = value.strip().upper()
        if normalized in VALID_STAKES:
            return normalized
    raise ValueError(f"Expected stake string for {field_name}.")


def normalize_deck(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected non-empty string for {field_name}.")
    normalized = value.strip().upper()
    if normalized in DECK_NAME_TO_CODE.values():
        return normalized
    try:
        return DECK_NAME_TO_CODE[value.strip().lower()]
    except KeyError as error:
        raise ValueError(
            f"Unsupported deck name for {field_name}: {value!r}."
        ) from error


def _require_index_list(
    value: Any, *, field_name: str, min_size: int, max_size: int | None = None
) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Expected non-empty list for {field_name}.")
    if not all(isinstance(item, int) for item in value):
        raise ValueError(f"Expected {field_name} to contain only integers.")
    if len(value) < min_size:
        raise ValueError(f"Expected at least {min_size} entries for {field_name}.")
    if max_size is not None and len(value) > max_size:
        raise ValueError(f"Expected at most {max_size} entries for {field_name}.")
    return value


def validate_command(
    command: dict[str, Any], game_state: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    allowed_calls = allowed_calls_for_state(game_state)
    name = _require_string(command.get("name"), field_name="name")
    if name not in allowed_calls:
        raise ValueError(
            f"{name!r} is not valid in state {normalize_state_name(game_state)!r}."
        )

    arguments = command.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError("Expected 'arguments' to be an object.")

    if name == "start":
        deck = normalize_deck(arguments.get("deck"), field_name="arguments.deck")
        stake = normalize_stake(arguments.get("stake"), field_name="arguments.stake")
        validated: dict[str, Any] = {"deck": deck, "stake": stake}
        if "seed" in arguments and arguments["seed"] is not None:
            validated["seed"] = _require_string(
                arguments["seed"], field_name="arguments.seed"
            )
        if "challenge" in arguments and arguments["challenge"] is not None:
            validated["challenge"] = _require_string(
                arguments["challenge"], field_name="arguments.challenge"
            )
        if "log_path" in arguments and arguments["log_path"] is not None:
            validated["log_path"] = _require_string(
                arguments["log_path"], field_name="arguments.log_path"
            )
        return name, validated

    if name in {"select", "skip", "cash_out", "reroll", "next_round", "menu"}:
        return name, {}

    if name in {"play", "discard"}:
        cards = _require_index_list(
            arguments.get("cards"), field_name="arguments.cards", min_size=1, max_size=5
        )
        return name, {"cards": cards}

    if name == "buy":
        return name, _require_exactly_one_index(
            arguments, field_names=("card", "voucher", "pack")
        )

    if name == "pack":
        return name, _require_exactly_one_index(arguments, field_names=("card",))

    raise ValueError(f"Validation is not implemented for call {name!r}.")


def build_inference_input(
    game_state: dict[str, Any],
    config: BotConfig,
    previous_error: str | None,
    *,
    preprocess_fn: Any = None,
) -> list[dict[str, Any]]:
    if preprocess_fn is not None:
        try:
            result = preprocess_fn(game_state)
            if isinstance(result, dict):
                game_state = result
            else:
                raise PreprocessingError(
                    f"preprocess() returned {type(result).__name__}, expected dict"
                )
        except PreprocessingError:
            raise
        except Exception as exc:
            raise PreprocessingError(str(exc)) from exc

    state_name = normalize_state_name(game_state)
    run_defaults = {
        "deck": normalize_deck(config.deck, field_name="config.deck"),
        "stake": normalize_stake(config.stake, field_name="config.stake"),
        "seed": config.seed,
    }
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "tensorzero::template",
                    "name": "turn_context",
                    "arguments": {
                        "state_name": state_name,
                        "all_actions_json": json.dumps(
                            ALL_BALATROBOT_ACTIONS, sort_keys=True
                        ),
                        "supported_action_schemas_json": json.dumps(
                            SUPPORTED_ACTION_SCHEMAS, sort_keys=True
                        ),
                        "allowed_calls_json": json.dumps(
                            allowed_calls_for_state(game_state), sort_keys=True
                        ),
                        "run_defaults_json": json.dumps(run_defaults, sort_keys=True),
                        "game_state_json": json.dumps(game_state, sort_keys=True),
                        "previous_error": previous_error,
                    },
                }
            ],
        }
    ]


def request_tensorzero_inference(
    gateway_url: str,
    function_name: str,
    inference_input: list[dict[str, Any]],
    episode_id: str | None,
    variant_name: str | None = None,
) -> tuple[str, dict[str, Any]]:
    from openai import OpenAI

    client = OpenAI(
        base_url=f"{gateway_url.rstrip('/')}/openai/v1",
        api_key="tensorzero",
    )
    extra_body: dict[str, Any] = {
        "tensorzero::include_raw_response": True,
        "tensorzero::include_raw_usage": True,
    }
    if episode_id is not None:
        extra_body["tensorzero::episode_id"] = episode_id
    if variant_name is not None:
        extra_body["tensorzero::variant_name"] = variant_name
    response = client.chat.completions.create(
        model=f"tensorzero::function_name::{function_name}",
        messages=inference_input,
        extra_body=extra_body,
    )
    raw_output = response.choices[0].message.content
    if not isinstance(raw_output, str):
        raise ValueError("TensorZero response did not include text output.")
    return raw_output, response.model_dump(mode="json")


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


def extract_tensorzero_output(
    raw_output: str, response_payload: dict[str, Any]
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    if not isinstance(response_payload, dict):
        raise ValueError("TensorZero response payload must be a JSON object.")
    return raw_output, parse_model_command(raw_output), response_payload


def choose_command(
    game_state: dict[str, Any],
    config: BotConfig,
    previous_error: str | None,
    episode_id: str | None,
    *,
    logger: RunLogger | None = None,
    turn: int | None = None,
    attempt: int | None = None,
    state_name: str | None = None,
    preprocess_fn: Any = None,
) -> tuple[str, dict[str, Any], str, str | None]:
    try:
        inference_input = build_inference_input(
            game_state, config, previous_error, preprocess_fn=preprocess_fn
        )
    except PreprocessingError as exc:
        if logger is not None:
            logger.log(
                "preprocessing_error",
                {"turn": turn, "attempt": attempt, "error": str(exc)},
            )
        # Fall back to unprocessed game state so the game can continue.
        inference_input = build_inference_input(game_state, config, previous_error)
    if logger is not None:
        logger.log(
            "model_input",
            {
                "turn": turn,
                "attempt": attempt,
                "state": state_name,
                "gateway_url": config.gateway_url,
                "function_name": config.function_name,
                "episode_id": episode_id,
                "input": inference_input,
                "previous_error": previous_error,
            },
        )

    raw_output, response_payload = request_tensorzero_inference(
        config.gateway_url,
        config.function_name,
        inference_input,
        episode_id,
        variant_name=config.variant,
    )
    raw_output, parsed_output, raw_response = extract_tensorzero_output(
        raw_output, response_payload
    )
    response_episode_id = raw_response.get("episode_id")
    if response_episode_id is not None and not isinstance(response_episode_id, str):
        raise ValueError("TensorZero response episode_id must be a string when present.")
    if logger is not None:
        logger.log(
            "model_output",
            {
                "turn": turn,
                "attempt": attempt,
                "state": state_name,
                "gateway_url": config.gateway_url,
                "function_name": config.function_name,
                "model": raw_response.get("model"),
                "id": raw_response.get("id"),
                "episode_id": response_episode_id,
                "usage": raw_response.get("usage"),
                "output_raw": raw_output,
                "output_parsed": parsed_output,
                "response": raw_response,
            },
        )
    name, arguments = validate_command(parsed_output, game_state)
    return name, arguments, raw_output, response_episode_id


def run_bot(config: BotConfig) -> None:
    from balatrobot.cli.client import BalatroClient

    preprocess_fn = None
    if config.preprocessing_path is not None:
        try:
            preprocess_fn = _load_preprocessing_fn(config.preprocessing_path)
            print(f"[preprocessing] Loaded from {config.preprocessing_path}")
        except PreprocessingError as exc:
            print(f"[preprocessing] Failed to load — running without it: {exc}")

    logger = RunLogger(build_log_path(config.log_dir))
    logger.log(
        "run_started",
        {
            "gateway_url": config.gateway_url,
            "function_name": config.function_name,
            "deck": config.deck,
            "stake": config.stake,
            "seed": config.seed,
            "max_turns": config.max_turns,
            "port": config.port,
        },
    )
    print(f"[log] {logger.path}")
    client = BalatroClient(port=config.port)
    previous_error: str | None = None
    episode_id: str | None = None
    for turn_index in range(config.max_turns):
        game_state = client.call("gamestate")
        state_name = normalize_state_name(game_state)
        logger.log(
            "game_state",
            {
                "turn": turn_index + 1,
                "state": state_name,
                "episode_id": episode_id,
                "payload": game_state,
            },
        )
        if state_name == "GAME_OVER":
            logger.log("run_finished", {"turn": turn_index + 1, "reason": "game_over"})
            print("Game over.")
            return

        refresh_state = False
        command_succeeded = False
        for attempt in range(3):
            try:
                name, arguments, raw_output, response_episode_id = choose_command(
                    game_state,
                    config,
                    previous_error,
                    episode_id,
                    logger=logger,
                    turn=turn_index + 1,
                    attempt=attempt + 1,
                    state_name=state_name,
                    preprocess_fn=preprocess_fn,
                )
            except ValueError as error:
                previous_error = str(error)
                logger.log(
                    "model_error",
                    {
                        "turn": turn_index + 1,
                        "attempt": attempt + 1,
                        "state": state_name,
                        "error": previous_error,
                    },
                )
                print(f"[model-error] {previous_error}")
                continue
            if episode_id is None:
                episode_id = response_episode_id
            logger.log(
                "model_choice",
                {
                    "turn": turn_index + 1,
                    "attempt": attempt + 1,
                    "state": state_name,
                    "episode_id": episode_id,
                    "name": name,
                    "arguments": arguments,
                    "raw_output": raw_output,
                    "previous_error": previous_error,
                },
            )
            print(
                f"[turn {turn_index + 1} attempt {attempt + 1}] {name} {json.dumps(arguments, sort_keys=True)}"
            )
            print(f"[model] {raw_output}")
            try:
                result = client.call(name, arguments)
                logger.log(
                    "api_result",
                    {
                        "turn": turn_index + 1,
                        "attempt": attempt + 1,
                        "episode_id": episode_id,
                        "name": name,
                        "arguments": arguments,
                        "result": result,
                    },
                )
                previous_error = None
                command_succeeded = True
                break
            except Exception as error:  # pragma: no cover
                previous_error = str(error)
                logger.log(
                    "api_error",
                    {
                        "turn": turn_index + 1,
                        "attempt": attempt + 1,
                        "episode_id": episode_id,
                        "name": name,
                        "arguments": arguments,
                        "error": previous_error,
                    },
                )
                print(f"[api-error] {previous_error}")
                if previous_error.startswith("INVALID_STATE:"):
                    refresh_state = True
                    previous_error = None
                    logger.log(
                        "state_refresh_requested",
                        {
                            "turn": turn_index + 1,
                            "attempt": attempt + 1,
                            "episode_id": episode_id,
                            "state": state_name,
                            "reason": "invalid_state",
                        },
                    )
                    break
        if refresh_state:
            continue
        if command_succeeded:
            continue
        else:
            logger.log(
                "run_finished",
                {
                    "turn": turn_index + 1,
                    "reason": "command_retry_exhausted",
                    "state": state_name,
                },
            )
            raise RuntimeError(
                f"Failed to produce a valid API call after 3 attempts in state {state_name}."
            )
    logger.log(
        "run_finished", {"turn": config.max_turns, "reason": "max_turns_reached"}
    )
    raise RuntimeError(f"Stopped after hitting max_turns={config.max_turns}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a BalatroBot client driven by the TensorZero inference API."
    )
    parser.add_argument(
        "--gateway-url",
        default="http://localhost:3000",
        help="TensorZero gateway base URL.",
    )
    parser.add_argument(
        "--function-name",
        default="balatro_next_command",
        help="TensorZero function name to call.",
    )
    parser.add_argument("--deck", default="Red Deck", help="Deck name for start_run.")
    parser.add_argument("--stake", default="WHITE", help="Stake name for start_run.")
    parser.add_argument(
        "--seed", default=None, help="Optional Balatro seed for reproducible runs."
    )
    parser.add_argument("--port", type=int, default=12346, help="BalatroBot TCP port.")
    parser.add_argument(
        "--max-turns", type=int, default=5000, help="Maximum API turns before aborting."
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory where JSONL run histories are written.",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="Force a specific TensorZero variant name (tensorzero::variant_name).",
    )
    parser.add_argument(
        "--preprocessing",
        default=None,
        help="Path to a Python file exporting preprocess(game_state: dict) -> dict.",
    )
    args = parser.parse_args()

    config = BotConfig(
        gateway_url=args.gateway_url,
        function_name=args.function_name,
        deck=args.deck,
        stake=args.stake,
        seed=args.seed,
        max_turns=args.max_turns,
        port=args.port,
        log_dir=Path(args.log_dir),
        variant=args.variant,
        preprocessing_path=Path(args.preprocessing) if args.preprocessing else None,
    )
    run_bot(config)


if __name__ == "__main__":
    main()
