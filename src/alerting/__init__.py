"""Alerting system for CR-Pipeline.

Provides:
- Training event notifications
- Convergence alerts
- Bottleneck alerts
- Fitness milestone alerts
- Error alerts
- Custom alert rules
- Multiple alert channels (console, file, webhook)
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# A message placeholder: a context key, an optional format spec, and up to
# three braces on each side, so both the current one-brace spelling and the
# older three-brace one are recognised.
_PLACEHOLDER = re.compile(r"\{{1,3}(\w+)(:[^}]*)?\}{1,3}")


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


class AlertType(Enum):
    """Types of alerts."""
    CONVERGENCE = auto()
    BOTTLENECK = auto()
    FITNESS_MILESTONE = auto()
    EARLY_STOP = auto()
    GPU_ERROR = auto()
    TRAINING_COMPLETE = auto()
    CUSTOM = auto()


@dataclass
class Alert:
    """Represents a single alert.

    Attributes:
        alert_type: Type of alert.
        level: Severity level.
        message: Alert message.
        timestamp: Alert timestamp.
        metadata: Additional alert data.
        acknowledged: Whether alert has been acknowledged.
    """
    alert_type: AlertType
    level: AlertLevel
    message: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False

    def to_dict(self) -> dict:
        return {
            "alert_type": self.alert_type.name,
            "level": self.level.name,
            "message": self.message,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "acknowledged": self.acknowledged,
        }


class AlertRule:
    """Defines a condition for triggering alerts."""

    def __init__(
        self,
        alert_type: AlertType,
        level: AlertLevel,
        condition: Callable[[Dict[str, Any]], bool],
        message_template: str,
        cooldown_sec: float = 60.0,
    ):
        self.alert_type = alert_type
        self.level = level
        self.condition = condition
        self.message_template = message_template
        self.cooldown_sec = cooldown_sec
        self._last_triggered: float = 0

    def should_trigger(self, context: Dict[str, Any]) -> bool:
        """Check if this rule should trigger given current context."""
        if not self.condition(context):
            return False
        if time.time() - self._last_triggered < self.cooldown_sec:
            return False
        return True

    def trigger(self, context: Dict[str, Any]) -> Alert:
        """Trigger the alert and return an Alert.

        A placeholder is a context key in braces, optionally with a format
        spec. Extra surrounding braces are consumed, so templates written in
        the older three-brace style still render cleanly: the previous
        substitution matched only the inner two braces and left the outer
        pair behind, which is why finished runs logged lines like
        "best fitness {1531.19}". A spec was never applied at all, so the
        elapsed time rendered as its own template text rather than a number.
        """
        self._last_triggered = time.time()

        def substitute(match: "re.Match") -> str:
            key, spec = match.group(1), match.group(2)
            if key not in context:
                return match.group(0)
            value = context[key]
            if spec:
                try:
                    return format(value, spec[1:])
                except (TypeError, ValueError):
                    return str(value)
            return str(value)

        message = _PLACEHOLDER.sub(substitute, self.message_template)
        return Alert(
            alert_type=self.alert_type,
            level=self.level,
            message=message,
            metadata=context,
        )


class AlertChannel:
    """Base class for alert delivery channels."""

    def send(self, alert: Alert) -> None:
        raise NotImplementedError


class ConsoleAlertChannel(AlertChannel):
    """Deliver alerts to console."""

    def send(self, alert: Alert) -> None:
        prefix = {
            AlertLevel.INFO: "[INFO]",
            AlertLevel.WARNING: "[WARNING]",
            AlertLevel.ERROR: "[ERROR]",
            AlertLevel.CRITICAL: "[CRITICAL]",
        }.get(alert.level, "[ALERT]")
        logger.info(f"{prefix} [{alert.alert_type.name}] {alert.message}")


class FileAlertChannel(AlertChannel):
    """Deliver alerts to a log file."""

    def __init__(self, log_path: str = "runs/alerts.log"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def send(self, alert: Alert) -> None:
        with open(self.log_path, "a") as f:
            f.write(json.dumps(alert.to_dict()) + "\n")


class AlertManager:
    """Manages alert rules, channels, and delivery.

    Monitors training context and triggers alerts based on rules.
    """

    def __init__(self):
        self.rules: List[AlertRule] = []
        self.channels: List[AlertChannel] = [ConsoleAlertChannel(), FileAlertChannel()]
        self.alert_history: List[Alert] = []

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self.rules.append(rule)

    def add_channel(self, channel: AlertChannel) -> None:
        """Add an alert delivery channel."""
        self.channels.append(channel)

    def check(self, context: Dict[str, Any]) -> List[Alert]:
        """Check all rules against current context and deliver alerts.

        Args:
            context: Current training context with metrics.

        Returns:
            List of triggered alerts.
        """
        triggered = []
        for rule in self.rules:
            if rule.should_trigger(context):
                alert = rule.trigger(context)
                triggered.append(alert)
                for channel in self.channels:
                    channel.send(alert)
        self.alert_history.extend(triggered)
        return triggered

    def get_default_rules(self) -> List[AlertRule]:
        """Get default alert rules for training monitoring."""
        return [
            # Convergence alert
            AlertRule(
                AlertType.CONVERGENCE,
                AlertLevel.INFO,
                lambda ctx: ctx.get("converged", False),
                "Training converged at generation {generation} with best fitness {best_fitness}",
                cooldown_sec=300,
            ),
            # Fitness milestone alert
            AlertRule(
                AlertType.FITNESS_MILESTONE,
                AlertLevel.INFO,
                lambda ctx: ctx.get("fitness_improved", False) and ctx.get("improvement_pct", 0) > 10,
                "Fitness improved by {improvement_pct:.1f}% at generation {generation}",
                cooldown_sec=60,
            ),
            # Early stopping alert
            AlertRule(
                AlertType.EARLY_STOP,
                AlertLevel.WARNING,
                lambda ctx: ctx.get("early_stop", False),
                "Early stopping triggered at generation {generation} - no improvement for {patience} generations",
                cooldown_sec=0,
            ),
            # Bottleneck alert
            AlertRule(
                AlertType.BOTTLENECK,
                AlertLevel.WARNING,
                lambda ctx: ctx.get("bottleneck_severity", "low") in ("high", "critical"),
                "Bottleneck detected: {bottleneck_type} with severity {bottleneck_severity}",
                cooldown_sec=120,
            ),
            # Training complete alert
            AlertRule(
                AlertType.TRAINING_COMPLETE,
                AlertLevel.INFO,
                lambda ctx: ctx.get("training_complete", False),
                "Training complete: best fitness {best_fitness} after {total_gens} generations in {elapsed_min:.1f} minutes",
                cooldown_sec=0,
            ),
        ]

    def save_history(self, path: str = "runs/alert_history.json") -> None:
        """Save alert history to file."""
        with open(path, "w") as f:
            json.dump([a.to_dict() for a in self.alert_history], f, indent=2)
