from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

STATE_TO_CALLS: dict[str, tuple[str, ...]] = {
    "MENU": ("start",),
    "BLIND_SELECT": ("select", "skip"),
    "SELECTING_HAND": ("play", "discard"),
    "ROUND_EVAL": ("cash_out",),
    "SHOP": ("buy", "reroll", "next_round"),
    "GAME_OVER": ("menu",),
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
    model: str
    deck: str
    stake: str
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
        raise ValueError(f"Unsupported deck name for {field_name}: {value!r}.") from error


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
        return name, {
            "index": _require_int(arguments.get("index"), field_name="arguments.index")
        }

    raise ValueError(f"Validation is not implemented for call {name!r}.")


def build_turn_prompt(
    game_state: dict[str, Any], config: BotConfig, previous_error: str | None
) -> str:
    allowed_calls = allowed_calls_for_state(game_state)
    prompt = {
        "objective": "Choose the single best next BalatroBot API call for this turn.",
        "constraints": [
            "Return exactly one JSON object and no surrounding prose.",
            "Use one of the allowed function names for the current state.",
            "Card and shop indices are 0-based.",
            "Do not invent fields beyond name and arguments.",
            "Prefer legal, conservative actions over risky guesses.",
        ],
        "run_defaults": {
            "deck": normalize_deck(config.deck, field_name="config.deck"),
            "stake": normalize_stake(config.stake, field_name="config.stake"),
            "seed": config.seed,
        },
        "allowed_calls": allowed_calls,
        "response_shape": {
            "name": "BalatroBot function name",
            "arguments": {"field": "value"},
        },
        "examples": [
            {
                "name": "start",
                "arguments": {
                    "deck": normalize_deck(config.deck, field_name="config.deck"),
                    "stake": normalize_stake(config.stake, field_name="config.stake"),
                    "seed": config.seed,
                },
            },
            {"name": "select", "arguments": {}},
            {"name": "play", "arguments": {"cards": [0, 1, 2, 3, 4]}},
            {"name": "next_round", "arguments": {}},
        ],
        "previous_error": previous_error,
        "game_state": game_state,
    }
    return json.dumps(prompt, sort_keys=True)


def build_model_input(
    game_state: dict[str, Any], config: BotConfig, previous_error: str | None
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are controlling a BalatroBot client. "
                "Reply with exactly one JSON object containing `name` and `arguments`."
            ),
        },
        {
            "role": "user",
            "content": build_turn_prompt(game_state, config, previous_error),
        },
    ]


def choose_command(
    game_state: dict[str, Any],
    config: BotConfig,
    previous_error: str | None,
    *,
    logger: RunLogger | None = None,
    turn: int | None = None,
    attempt: int | None = None,
    state_name: str | None = None,
) -> tuple[str, dict[str, Any], str]:
    from openai import OpenAI

    model_input = build_model_input(game_state, config, previous_error)
    if logger is not None:
        logger.log(
            "model_input",
            {
                "turn": turn,
                "attempt": attempt,
                "state": state_name,
                "model": config.model,
                "input": model_input,
                "previous_error": previous_error,
            },
        )

    client = OpenAI()
    response = client.responses.create(
        model=config.model,
        input=model_input,
    )
    raw_output = response.output_text
    if logger is not None:
        logger.log(
            "model_output",
            {
                "turn": turn,
                "attempt": attempt,
                "state": state_name,
                "model": config.model,
                "output": raw_output,
            },
        )
    parsed = parse_model_command(raw_output)
    name, arguments = validate_command(parsed, game_state)
    return name, arguments, raw_output


def run_bot(config: BotConfig) -> None:
    from balatrobot.cli.client import BalatroClient

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
    client = BalatroClient(port=config.port)
    previous_error: str | None = None
    for turn_index in range(config.max_turns):
        game_state = client.call("gamestate")
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
            name, arguments, raw_output = choose_command(
                game_state,
                config,
                previous_error,
                logger=logger,
                turn=turn_index + 1,
                attempt=attempt + 1,
                state_name=state_name,
            )
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
            raise RuntimeError(
                f"Failed to produce a valid API call after 3 attempts in state {state_name}."
            )
        logger.log(
            "run_finished", {"turn": config.max_turns, "reason": "max_turns_reached"}
        )
        raise RuntimeError(f"Stopped after hitting max_turns={config.max_turns}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a BalatroBot client driven by the OpenAI Responses API."
    )
    parser.add_argument(
        "--model", default="gpt-5-mini", help="OpenAI model name to use."
    )
    parser.add_argument("--deck", default="Red Deck", help="Deck name for start_run.")
    parser.add_argument(
        "--stake", default="WHITE", help="Stake name for start_run."
    )
    parser.add_argument(
        "--seed", default=None, help="Optional Balatro seed for reproducible runs."
    )
    parser.add_argument("--port", type=int, default=12346, help="BalatroBot TCP port.")
    parser.add_argument(
        "--max-turns", type=int, default=250, help="Maximum API turns before aborting."
    )
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
