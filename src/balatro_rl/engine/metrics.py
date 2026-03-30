from __future__ import annotations

from balatro_rl.engine.models import TransitionMetrics
from balatro_rl.engine.serialization import to_primitive


def metrics_to_dict(metrics: TransitionMetrics) -> dict[str, object]:
    return to_primitive(metrics)
