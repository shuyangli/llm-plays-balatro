"""Register TensorZero variants and run Balatro games to collect metrics."""
from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

from meta_harness.parse_results import aggregate_scores, parse_game_log

PROJECT_DIR = Path(__file__).parent.parent

# TensorZero model used for all meta-harness candidates.
META_MODEL = "gpt_5_4_nano"


def register_variant(
    candidate_id: str,
    harness_dir: Path,
    *,
    model: str = META_MODEL,
) -> None:
    """
    Copy harness templates into the TensorZero config tree and add a variant
    block to tensorzero.toml.  Safe to call repeatedly — existing entries are
    left untouched.
    """
    variant_name = f"meta_{candidate_id}"
    templates_dst = (
        PROJECT_DIR
        / "config"
        / "functions"
        / "balatro_next_command"
        / "variants"
        / variant_name
        / "templates"
    )
    templates_dst.mkdir(parents=True, exist_ok=True)

    for template in ("system.minijinja", "turn_context.minijinja"):
        src = harness_dir / template
        if src.exists():
            (templates_dst / template).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    _append_variant_to_toml(variant_name, model, templates_dst)


def _append_variant_to_toml(variant_name: str, model: str, templates_dir: Path) -> None:
    toml_path = PROJECT_DIR / "config" / "tensorzero.toml"
    content = toml_path.read_text(encoding="utf-8")

    header = f"[functions.balatro_next_command.variants.{variant_name}]"
    if header in content:
        return  # Already registered.

    # Paths relative to the config directory (how TensorZero expects them).
    rel_base = templates_dir.relative_to(PROJECT_DIR / "config")
    turn_context_path = str(rel_base / "turn_context.minijinja").replace("\\", "/")
    system_path = str(rel_base / "system.minijinja").replace("\\", "/")

    new_block = (
        f"\n{header}\n"
        f'type = "chat_completion"\n'
        f'model = "{model}"\n'
        f'reasoning_effort = "high"\n'
        f"templates = {{ "
        f'turn_context = {{ path = "{turn_context_path}" }}, '
        f'system = {{ path = "{system_path}" }} '
        f"}}\n"
    )

    # Insert just before the [experimentation] block so the file stays tidy.
    experimentation_header = "[functions.balatro_next_command.experimentation]"
    if experimentation_header in content:
        content = content.replace(experimentation_header, new_block + experimentation_header)
    else:
        content += new_block

    toml_path.write_text(content, encoding="utf-8")


def restart_gateway(*, gateway_url: str = "http://localhost:3000", timeout_s: int = 120) -> None:
    """Restart the TensorZero Docker service and wait until healthy."""
    subprocess.run(
        ["docker", "compose", "restart", "gateway"],
        cwd=PROJECT_DIR,
        check=True,
        timeout=60,
    )
    _wait_for_health(gateway_url, timeout_s=timeout_s)


def _wait_for_health(gateway_url: str, *, timeout_s: int = 120) -> None:
    deadline = time.monotonic() + timeout_s
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"{gateway_url}/health", timeout=2)
            return
        except Exception as exc:
            last_exc = exc
            time.sleep(1)
    raise RuntimeError(
        f"TensorZero gateway at {gateway_url} did not become healthy within {timeout_s}s: {last_exc}"
    )


def run_game(
    variant_name: str,
    seed: str,
    log_dir: Path,
    *,
    preprocessing_path: Path | None = None,
    port: int = 12346,
    timeout_s: int = 1800,
) -> Path:
    """
    Invoke balatrobot-openai for one game and return the path to its JSONL log.
    Raises RuntimeError if no new log appears after the process exits.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    before = set(log_dir.glob("balatrobot-openai-*.jsonl"))

    cmd = [
        "uv",
        "run",
        "balatrobot-openai",
        "--variant",
        variant_name,
        "--seed",
        seed,
        "--log-dir",
        str(log_dir),
        "--port",
        str(port),
    ]
    if preprocessing_path is not None and preprocessing_path.exists():
        cmd += ["--preprocessing", str(preprocessing_path)]

    # check=False because GAME_OVER is a normal exit that raises RuntimeError in run_bot.
    subprocess.run(cmd, cwd=PROJECT_DIR, check=False, timeout=timeout_s)

    after = set(log_dir.glob("balatrobot-openai-*.jsonl"))
    new_logs = sorted(after - before)
    if not new_logs:
        raise RuntimeError(
            f"No JSONL log created in {log_dir} for variant={variant_name}, seed={seed}"
        )
    return new_logs[-1]


def evaluate_candidate(
    candidate_dir: Path,
    variant_name: str,
    seeds: list[str],
    *,
    port: int = 12346,
) -> dict[str, Any]:
    """
    Run one game per seed, parse results, persist scores.json, return aggregated scores.
    """
    traces_dir = candidate_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    preprocessing_path: Path | None = candidate_dir / "harness" / "preprocessing.py"
    if not preprocessing_path.exists():
        preprocessing_path = None

    all_metrics: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"  [eval] variant={variant_name} seed={seed}")
        log_path = run_game(
            variant_name, seed, traces_dir,
            preprocessing_path=preprocessing_path,
            port=port,
        )
        metrics = parse_game_log(log_path)
        metrics["seed"] = seed
        all_metrics.append(metrics)
        print(
            f"  [eval] → ante_reached={metrics['ante_reached']} "
            f"reason={metrics['run_finished_reason']} "
            f"errors={metrics['error_count']}"
        )

    scores = aggregate_scores(all_metrics)
    scores["per_game"] = all_metrics
    (candidate_dir / "scores.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    print(
        f"  [eval] Summary: avg_ante={scores['avg_ante']:.2f} "
        f"max_ante={scores['max_ante']} "
        f"error_rate={scores['error_rate']:.4f}"
    )
    return scores
