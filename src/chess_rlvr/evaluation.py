"""Evaluation metrics with an explicit denominator and uncertainty.

Project rule: no number without a denominator. Every rate is reported with a
Wilson confidence interval, because `3/4` and `300/400` are not the same evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson confidence interval for a proportion.

    Preferred over the normal interval because it never produces bounds outside
    [0, 1] and does not collapse when the proportion is 0 or 1, which is exactly
    the baseline case.
    """
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return (max(0.0, center - margin), min(1.0, center + margin))


@dataclass
class Count:
    """A success count with its denominator and confidence interval."""
    successes: int = 0
    total: int = 0

    def add(self, ok: bool) -> None:
        self.total += 1
        if ok:
            self.successes += 1

    @property
    def rate(self) -> float:
        return self.successes / self.total if self.total else 0.0

    def line(self, label: str) -> str:
        lo, hi = wilson(self.successes, self.total)
        return "%-26s %5d/%-5d = %6.2f%%  95%% CI [%.2f%%, %.2f%%]" % (
            label, self.successes, self.total, 100 * self.rate, 100 * lo, 100 * hi
        )


@dataclass
class EvaluationResult:
    """Aggregates the metrics of one evaluation run."""
    legal: Count = field(default_factory=Count)
    correct: Count = field(default_factory=Count)
    unparseable: Count = field(default_factory=Count)
    by_band: dict[str, Count] = field(default_factory=dict)
    by_theme: dict[str, Count] = field(default_factory=dict)

    def record(self, *, status: str, band: str, themes: tuple[str, ...]) -> None:
        self.legal.add(status in ("correct", "legal_good", "legal_bad"))
        self.correct.add(status == "correct")
        self.unparseable.add(status == "unparseable")

        self.by_band.setdefault(band, Count()).add(status == "correct")
        for theme in themes:
            self.by_theme.setdefault(theme, Count()).add(status == "correct")

    def report(self, title: str) -> str:
        parts = [f"=== {title}", self.legal.line("legal move"),
                 self.correct.line("correct move"),
                 self.unparseable.line("unparseable")]

        if self.by_band:
            parts.append("-- by rating band (correct) --")
            for band in sorted(self.by_band):
                parts.append(self.by_band[band].line(band))

        # Only themes with enough mass: a slice of 3 cases is not evidence.
        themes = [(t, c) for t, c in self.by_theme.items() if c.total >= 20]
        if themes:
            parts.append("-- by theme (correct, n>=20) --")
            for theme, count in sorted(themes, key=lambda kv: -kv[1].rate):
                parts.append(count.line(theme))

        return "\n".join(parts)

    def as_dict(self) -> dict:
        def block(c: Count) -> dict:
            lo, hi = wilson(c.successes, c.total)
            return {"successes": c.successes, "total": c.total,
                    "rate": round(c.rate, 4),
                    "ci95": [round(lo, 4), round(hi, 4)]}

        return {
            "legal": block(self.legal),
            "correct": block(self.correct),
            "unparseable": block(self.unparseable),
            "by_band": {k: block(v) for k, v in sorted(self.by_band.items())},
            "by_theme": {k: block(v) for k, v in sorted(self.by_theme.items())
                         if v.total >= 20},
        }

