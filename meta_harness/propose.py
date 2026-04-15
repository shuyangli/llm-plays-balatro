"""Generate an improved harness candidate using GPT-5.4-nano."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).parent.parent

# How many events to include per trace shown to the proposer.
_MAX_TRACE_EVENTS = 40
# How many traces to include in the proposer context (worst N + best 1).
_WORST_TRACES = 2
_BEST_TRACES = 1

_DEFAULT_PREPROCESSING = '''\
"""
Preprocessing hook: runs before template rendering each turn.
Add computed features to the returned dict — do not remove existing keys.
Exceptions are caught automatically; the original game_state is used as fallback.
"""


def preprocess(game_state: dict) -> dict:
    return game_state
'''

PROPOSER_SYSTEM = """\
You are an expert Balatro player and prompt engineer.
Your job: improve a bot's harness so it plays Balatro better.

The harness has three components you can modify:
1. system.minijinja  — system prompt shown to the model every turn
2. turn_context.minijinja  — user message template (uses minijinja)
3. preprocessing.py  — Python executed before template rendering each turn
   - Must export: def preprocess(game_state: dict) -> dict
   - May add new keys to the dict; must never remove existing keys
   - The augmented dict replaces game_state_json in the template
   - Exceptions are silently caught; original game_state is used as fallback
   - Only use the Python standard library (no third-party imports)

Template variables available in both minijinja templates:
  {{ state_name }}               current game state name
  {{ allowed_calls_json }}       JSON list of valid actions right now
  {{ game_state_json }}          full game state (augmented by preprocessing)
  {{ previous_error }}           last validation/API error, or null
  {{ run_defaults_json }}        deck, stake, seed for the "start" action
  {{ supported_action_schemas_json }}  action argument specs
  {{ all_actions_json }}         all known BalatroBot actions (reference only)

You will receive:
1. The current best harness (all three files)
2. Aggregate performance scores
3. Compact game traces (game states + model choices) showing wins and failures
4. A history of previous attempts (avoid repeating them)

Rules:
- Identify the SINGLE most impactful failure mode visible in the traces
- Make one focused, surgical improvement — do not rewrite things that work
- All three output blocks must be complete and self-contained
- For preprocessing: useful additions include computing the best playable hand,
  summarising joker synergies, or flagging boss blind constraints in plain text

Output EXACTLY this structure (all four XML tags are required):

<analysis>
One or two sentences: what is the key failure mode and what you changed.
</analysis>

<system_prompt>
[Complete new system.minijinja content]
</system_prompt>

<turn_context>
[Complete new turn_context.minijinja content]
</turn_context>

<preprocessing>
[Complete new preprocessing.py content]
</preprocessing>
"""


def propose_candidate(
    best_candidate_dir: Path,
    *,
    model: str = "gpt-5.4-nano-2026-03-17",
    all_candidates: list[dict[str, Any]] | None = None,
) -> tuple[str, str, str, str]:
    """
    Generate improved harness files for a new candidate.

    Returns:
        (system_prompt_text, turn_context_text, preprocessing_py, analysis_text)
    """
    from openai import OpenAI

    harness_dir = best_candidate_dir / "harness"
    scores_path = best_candidate_dir / "scores.json"
    traces_dir = best_candidate_dir / "traces"

    system_text = (harness_dir / "system.minijinja").read_text(encoding="utf-8")
    turn_ctx_text = (harness_dir / "turn_context.minijinja").read_text(encoding="utf-8")
    preprocessing_text = _read_preprocessing(harness_dir)
    scores = json.loads(scores_path.read_text(encoding="utf-8")) if scores_path.exists() else {}

    trace_paths = _select_traces(traces_dir, scores)
    trace_summaries = [_summarize_trace(p) for p in trace_paths]

    user_message = _build_user_message(
        system_text, turn_ctx_text, preprocessing_text,
        scores, trace_summaries, all_candidates,
    )

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": PROPOSER_SYSTEM},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
    )

    raw = response.choices[0].message.content or ""
    return _parse_output(raw, system_text, turn_ctx_text, preprocessing_text)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_preprocessing(harness_dir: Path) -> str:
    pp = harness_dir / "preprocessing.py"
    return pp.read_text(encoding="utf-8") if pp.exists() else _DEFAULT_PREPROCESSING


# ---------------------------------------------------------------------------
# Trace selection and summarisation
# ---------------------------------------------------------------------------

def _select_traces(traces_dir: Path, scores: dict[str, Any]) -> list[Path]:
    """Return worst-N + best-1 game logs by ante_reached."""
    per_game = scores.get("per_game") or []
    if per_game:
        ranked = sorted(per_game, key=lambda m: m["ante_reached"])
        candidates = ranked[:_WORST_TRACES] + ranked[-_BEST_TRACES:]
        paths: list[Path] = []
        seen: set[str] = set()
        for m in candidates:
            p = Path(m["log_path"])
            if p.exists() and str(p) not in seen:
                paths.append(p)
                seen.add(str(p))
        if paths:
            return paths
    # Fallback: newest files in traces dir.
    return sorted(traces_dir.glob("balatrobot-openai-*.jsonl"))[-(_WORST_TRACES + _BEST_TRACES):]


def _summarize_trace(log_path: Path) -> str:
    """Build a compact event-by-event summary of a game log for the proposer."""
    events: list[dict[str, Any]] = []
    with log_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    condensed: list[dict[str, Any]] = []
    for event in events:
        ev = event.get("event", "")
        if ev == "run_started":
            condensed.append({"event": "run_started", "seed": event.get("seed")})
        elif ev == "run_finished":
            condensed.append({
                "event": "run_finished",
                "turn": event.get("turn"),
                "reason": event.get("reason"),
            })
        elif ev == "game_state":
            condensed.append({
                "event": "game_state",
                "turn": event.get("turn"),
                "state": event.get("state"),
                "ante": _extract_ante_from_payload(event.get("payload", {})),
                "summary": _key_fields(event.get("payload", {})),
            })
        elif ev == "model_choice":
            condensed.append({
                "event": "model_choice",
                "turn": event.get("turn"),
                "state": event.get("state"),
                "name": event.get("name"),
                "arguments": event.get("arguments"),
                "previous_error": event.get("previous_error"),
            })
        elif ev in ("model_error", "api_error", "preprocessing_error"):
            condensed.append({
                "event": ev,
                "turn": event.get("turn"),
                "error": event.get("error"),
            })

    # Keep first 10 events (context) + last 30 events (failure region).
    if len(condensed) > _MAX_TRACE_EVENTS:
        head = condensed[:10]
        tail = condensed[-(max(_MAX_TRACE_EVENTS - 10, 1)):]
        selected = head + [{"event": "...(truncated)..."}] + tail
    else:
        selected = condensed

    lines = [f"=== {log_path.name} ==="]
    for e in selected:
        lines.append(json.dumps(e, sort_keys=True))
    return "\n".join(lines)


def _extract_ante_from_payload(payload: dict[str, Any]) -> int | None:
    from meta_harness.parse_results import _extract_ante
    return _extract_ante(payload)


def _key_fields(gs: dict[str, Any]) -> dict[str, Any]:
    """Extract a small subset of game-state fields relevant for strategy analysis."""
    keep = {
        "ante", "round", "hands", "discards", "money", "jokers",
        "hand", "played_this_round", "boss", "round_info", "blind",
    }
    return {k: v for k, v in gs.items() if k in keep}


# ---------------------------------------------------------------------------
# Prompt assembly and output parsing
# ---------------------------------------------------------------------------

def _build_user_message(
    system_text: str,
    turn_ctx_text: str,
    preprocessing_text: str,
    scores: dict[str, Any],
    trace_summaries: list[str],
    all_candidates: list[dict[str, Any]] | None,
) -> str:
    parts = [
        "## Current Best Harness",
        "",
        "### system.minijinja",
        "```",
        system_text,
        "```",
        "",
        "### turn_context.minijinja",
        "```",
        turn_ctx_text,
        "```",
        "",
        "### preprocessing.py",
        "```python",
        preprocessing_text,
        "```",
        "",
        "## Performance Scores",
        "```json",
        json.dumps({k: v for k, v in scores.items() if k != "per_game"}, indent=2),
        "```",
        "",
    ]

    if all_candidates:
        history_entries = [
            {"id": c["id"], "avg_ante": c.get("avg_ante"), "analysis": c.get("analysis")}
            for c in all_candidates
        ]
        parts += [
            "## Previous Attempts (do not repeat these strategies)",
            "```json",
            json.dumps(history_entries, indent=2),
            "```",
            "",
        ]

    parts += ["## Game Traces", ""]
    for summary in trace_summaries:
        parts.append(summary)
        parts.append("")

    return "\n".join(parts)


def _parse_output(
    raw: str,
    fallback_system: str,
    fallback_turn_ctx: str,
    fallback_preprocessing: str,
) -> tuple[str, str, str, str]:
    """Extract XML-tagged sections from the proposer response."""
    analysis = _extract_tag(raw, "analysis") or "No analysis provided."
    system_prompt = _extract_tag(raw, "system_prompt") or fallback_system
    turn_context = _extract_tag(raw, "turn_context") or fallback_turn_ctx
    preprocessing = _extract_tag(raw, "preprocessing") or fallback_preprocessing
    return system_prompt.strip(), turn_context.strip(), preprocessing.strip(), analysis.strip()


def _extract_tag(text: str, tag: str) -> str | None:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1) if match else None
