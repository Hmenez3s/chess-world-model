#!/usr/bin/env python3
"""Estimate puzzle Elo by maximum likelihood and item bootstrap."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def probability(rating: int, elo: float) -> float:
    exponent = max(-700.0, min(700.0, math.log(10) * (rating - elo) / 400))
    return 1 / (1 + math.exp(exponent))


def estimate(items: list[dict]) -> float | None:
    """Root of the log-likelihood derivative; None for an empty/extreme sample."""
    n = len(items)
    correct = sum(bool(item["correct"]) for item in items)
    if n == 0 or correct in (0, n):
        return None

    def derivative(elo: float) -> float:
        return sum((int(bool(item["correct"])) - probability(int(item["rating"]), elo))
                   for item in items)

    low, high = -1000.0, 4000.0
    for _ in range(100):
        middle = (low + high) / 2
        if derivative(middle) > 0:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def upper_bound_zero(items: list[dict], z: float = 1.96) -> float | None:
    """With zero correct, the MLE is -infinity; the data still upper-bounds Elo.

    Wilson upper bound for 0/n is z^2 / (n + z^2). Find the rating R at which
    the expected success rate over the test's actual ratings equals this bound.
    A model above that R would have a 95% chance of solving at least one puzzle.
    """
    n = len(items)
    if n == 0:
        return None
    target = z * z / (n + z * z)

    def expected(elo: float) -> float:
        return sum(probability(int(item["rating"]), elo) for item in items) / n

    low, high = -2000.0, 4000.0
    for _ in range(100):
        middle = (low + high) / 2
        if expected(middle) < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def analyze(path: str, bootstrap: int, rng: random.Random) -> dict:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    items = data["items"]
    for item in items:
        if "rating" not in item or "correct" not in item:
            raise ValueError(f"{path}: each item must have rating and correct")
    n = len(items)
    correct = sum(bool(item["correct"]) for item in items)
    rate = correct / n if n else None
    elo = estimate(items)
    if elo is None:
        reason = ("no items" if n == 0 else
                  "0 solved: maximum likelihood tends to -infinity" if correct == 0 else
                  "100% solved: maximum likelihood tends to +infinity")
        ci = None
    else:
        samples = []
        for _ in range(bootstrap):
            sample = [items[rng.randrange(n)] for _ in range(n)]
            value = estimate(sample)
            if value is not None:
                samples.append(value)
        ci = ([percentile(samples, 0.025), percentile(samples, 0.975)]
              if samples else None)
        reason = None
    upper_bound = upper_bound_zero(items) if (n and correct == 0) else None
    return {"file": path, "n": n, "correct": correct, "rate": rate,
            "elo_puzzle": elo, "ci95": ci, "elo_max": upper_bound,
            "status": "estimated" if elo is not None else "undefined",
            "reason": reason, "bootstrap_valid": (len(samples) if elo is not None else 0)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/puzzle-elo.json")
    args = ap.parse_args()
    if args.bootstrap < 0:
        ap.error("--bootstrap must be >= 0")
    rng = random.Random(args.seed)
    results = [analyze(path, args.bootstrap, rng) for path in args.files]
    print(f"{'file':<32} {'n':>6} {'correct':>8} {'rate':>9} {'Puzzle Elo':>12} CI95")
    for result in results:
        rate = "undefined" if result["rate"] is None else f"{100*result['rate']:.2f}%"
        elo = ("undefined" if result["elo_puzzle"] is None else f"{result['elo_puzzle']:.1f}")
        if result["elo_puzzle"] is None and result.get("elo_max") is not None:
            elo = f"<= {result['elo_max']:.0f}"
        ci = "undefined" if result["ci95"] is None else f"[{result['ci95'][0]:.1f}, {result['ci95'][1]:.1f}]"
        print(f"{result['file']:<32} {result['n']:>6} {result['correct']:>8} {rate:>9} {elo:>12} {ci}")
        if result["reason"]:
            print("  reason:", result["reason"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved to", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
