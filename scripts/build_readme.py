#!/usr/bin/env python3
"""Build the English research README from measured JSON evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
TEMPLATE = ROOT / "docs" / "readme_template.md"
README = ROOT / "README.md"
MARKERS = (
    "{{START_VS_END}}",
    "{{GAME_ANALYSIS}}",
    "{{DIFFICULTY}}",
    "{{RECIPE}}",
    "{{LAYERS}}",
    "{{INFERENCE}}",
    "{{STAR}}",
    "{{GENERALIZATION}}",
    "{{MIRROR}}",
    "{{COST}}",
    "{{OTHER_METHODS}}",
)
EVIDENCE_FILES = (
    "eval-base.json", "eval-sft.json", "eval-sft-grpo.json",
    "eval-mav-30k.json", "eval-mav-70k.json", "eval-mav-verified.json",
    "eval-mav-grpo.json", "mirror-sft.json", "mirror-mav-verified.json",
    "games-random-base.json", "games-random-verified.json",
    "games-stockfish-base.json", "games-stockfish-verified.json",
    "game-analysis.json", "puzzle-elo.json", "inference-compute.json",
    "star-iter1-sampling.json", "eval-mav-star1.json", "eval-mav-ctrl1.json",
    "generalization-base.json", "generalization-trained.json",
    "train-mav-30k.json", "train-mav-70k.json", "other-methods.json",
)


class EvidenceError(RuntimeError):
    """An input is missing or does not contain the required evidence."""


def load(name: str):
    path = OUT / name
    if not path.is_file():
        raise EvidenceError(f"required evidence file is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read valid JSON from {path}: {exc}") from exc


def pct(value: float) -> str:
    return f"{value:.2f}%"


def ci(values) -> str:
    return "—" if values is None else f"[{values[0]:.2f}, {values[1]:.2f}]"


def integer_ci(values) -> str:
    return "" if values is None else f"[{values[0]:.0f}, {values[1]:.0f}]"


def table_start_vs_end() -> str:
    start, end = load("eval-base.json"), load("eval-mav-verified.json")
    random_start = load("games-random-base.json")
    random_end = load("games-random-verified.json")
    sf_start = load("games-stockfish-base.json")
    sf_end = load("games-stockfish-verified.json")

    def wdl(data):
        return f"{data['wins']}/{data['draws']}/{data['losses']}"

    rows = [
        '| | Start: Qwen2.5-1.5B-Instruct | End: world-model SFT + legal-move verification |',
        '|---|---|---|',
        f"| Puzzles solved (with 95% CI) | {pct(start['best_correct_pct'])} {ci(start['best_correct_ci95'])} | {pct(end['best_correct_pct'])} {ci(end['best_correct_ci95'])} |",
        f"| Legal move rate | {pct(start['best_legal_pct'])} | {pct(end['best_legal_pct'])} |",
        f"| Board after the move (exact) | cannot | {pct(end['board_exact_pct'])} |",
        f"| Games vs random player (W/D/L) | {wdl(random_start)} | {wdl(random_end)} |",
        f"| Games vs Stockfish `UCI_Elo=1320` (W/D/L) | {wdl(sf_start)} | {wdl(sf_end)} |",
        f"| Illegal moves in games | {pct(100 * sf_start['illegal_move_rate'])} | {pct(100 * sf_end['illegal_move_rate'])} |",
    ]
    return "\n".join(rows)


def table_game_analysis() -> str:
    data = load("game-analysis.json")
    rows = ["| Metric | Value |", "|---|---|"]
    metrics = (
        ("Games", str(data["games"])),
        ("Model moves", str(data["model_moves"])),
        ("Blunders (≥ 300 cp)", str(data["blunders"])),
        ("One blunder every N moves", f"{data['moves_per_blunder']:.1f}"),
        ("Mean move of first blunder", f"{data['mean_first_blunder_move']:.1f}"),
        ("Games with a blunder by move 5", f"{data['games_with_blunder_by_move_5']} of {data['games']}"),
    )
    rows.extend(f"| {name} | {value} |" for name, value in metrics)
    return "\n".join(rows)


def puzzle_elo_by_file() -> dict[str, dict]:
    return {Path(row["file"]).name: row for row in load("puzzle-elo.json")}


def table_difficulty() -> str:
    evaluation = load("eval-mav-verified.json")
    elo_row = puzzle_elo_by_file().get("eval-mav-verified.json")
    if elo_row is None or elo_row.get("elo_puzzle") is None:
        raise EvidenceError("puzzle-elo.json has no estimated Elo for eval-mav-verified.json")
    rating = elo_row["elo_puzzle"]
    rating_label = f"{rating:.0f}"
    grouped: dict[int, list[dict]] = {}
    for item in evaluation["items"]:
        lower = (int(item["rating"]) // 400) * 400
        grouped.setdefault(lower, []).append(item)

    rows = [
        f"| Rating band | n | Model | Expected for a {rating_label}-rated solver |",
        "|---|---:|---:|---:|",
    ]
    for lower, items in sorted(grouped.items()):
        if len(items) < 20:
            continue
        model = 100 * sum(bool(item["correct"]) for item in items) / len(items)
        expected = 100 * sum(
            1 / (1 + 10 ** ((item["rating"] - rating) / 400)) for item in items
        ) / len(items)
        rows.append(f"| {lower}–{lower + 399} | {len(items)} | {pct(model)} | {pct(expected)} |")
    return "\n".join(rows)


def table_recipe() -> str:
    stages = (
        ("Qwen2.5-1.5B-Instruct (no training)", "eval-base.json"),
        ("SFT (move only)", "eval-sft.json"),
        ("SFT → GRPO (Stockfish reward)", "eval-sft-grpo.json"),
        ("World-model SFT, 30k positions", "eval-mav-30k.json"),
        ("World-model SFT, 70k positions", "eval-mav-70k.json"),
        ("**+ legal-move verification (final)**", "eval-mav-verified.json"),
        ("+ GRPO (not significant)", "eval-mav-grpo.json"),
    )
    elo = puzzle_elo_by_file()
    rows = ["| Step | Puzzles solved | Puzzle Elo (comparative) |", "|---|---:|---:|"]
    for label, filename in stages:
        data = load(filename)
        result = elo.get(filename)
        if result is None:
            raise EvidenceError(f"puzzle-elo.json has no entry for {filename}")
        if result.get("elo_puzzle") is None:
            if result.get("elo_max") is None:
                elo_text = "—"
            else:
                elo_text = f"≤ {result['elo_max']:.0f} (0 solved)"
        else:
            elo_text = f"{result['elo_puzzle']:.0f} {integer_ci(result.get('ci95'))}"
        rows.append(f"| {label} | {pct(data['best_correct_pct'])} | {elo_text} |")
    return "\n".join(rows)


def table_layers() -> str:
    rows = [
        "| Step | Board exact | Color | Legal exact (F1) | Best = solution (CI) |",
        "|---|---:|---:|---:|---:|",
    ]
    stages = (
        ("Qwen2.5-1.5B-Instruct (no training)", "eval-base.json", False),
        ("SFT (move only)", "eval-sft.json", False),
        ("SFT → GRPO (Stockfish reward)", "eval-sft-grpo.json", False),
        ("World-model SFT, 30k positions", "eval-mav-30k.json", True),
        ("World-model SFT, 70k positions", "eval-mav-70k.json", True),
        ("**+ legal-move verification (final)**", "eval-mav-verified.json", True),
        ("+ GRPO (not significant)", "eval-mav-grpo.json", True),
    )
    for label, filename, has_world_model in stages:
        data = load(filename)
        if not has_world_model:
            board = color = legal = "n/a"
        else:
            board = pct(data["board_exact_pct"])
            color = pct(data["color_correct_pct"])
            legal = f"{pct(data['legal_exact_pct'])} ({data['legal_f1_mean']:.3f})"
        rows.append(
            f"| {label} | {board} | {color} | {legal} | "
            f"{pct(data['best_correct_pct'])} {ci(data['best_correct_ci95'])} |"
        )
    return "\n".join(rows)


def table_inference() -> str:
    data = load("inference-compute.json")
    k = data["k"]
    methods = (
        ("Greedy", "greedy"),
        ("Sampled (T = 0.8)", "sampled"),
        (f"Self-consistency (majority of {k})", "majority"),
        ("Depth-2 minimax with own values", "search_depth2"),
        (f"pass@{k} (upper bound for RL / rejection sampling)", "pass_at_k"),
    )
    rows = ["| Decoding | Accuracy | 95% CI |", "|---|---:|---:|"]
    rows.extend(
        f"| {label} | {pct(data[key]['pct'])} | {ci(data[key]['ci95'])} |"
        for label, key in methods
    )
    return "\n".join(rows)


def table_star() -> str:
    sampling = load("star-iter1-sampling.json")
    end = load("eval-mav-verified.json")
    star = load("eval-mav-star1.json")
    control = load("eval-mav-ctrl1.json")
    rows = [
        "| Model | Puzzles solved (test, n=600) | 95% CI |",
        "|---|---:|---:|",
        f"| End model, no extra training | {pct(end['best_correct_pct'])} | "
        f"{ci(end['best_correct_ci95'])} |",
        f"| + STaR on its own correct traces | {pct(star['best_correct_pct'])} | "
        f"{ci(star['best_correct_ci95'])} |",
        f"| + control: same budget, original data | {pct(control['best_correct_pct'])} | "
        f"{ci(control['best_correct_ci95'])} |",
        "",
        f"Training puzzles sampled: {sampling['puzzles']}, {sampling['k']} answers each "
        f"(T = {sampling['temperature']}). Correct answers sampled: "
        f"{pct(sampling['sample_accuracy_pct'])}. Usable traces kept: "
        f"{sampling['kept_examples']} ({pct(sampling['pass_at_k_pct'])} of the puzzles).",
    ]
    return "\n".join(rows)

def table_generalization() -> str:
    start = load("generalization-base.json")
    end = load("generalization-trained.json")
    pairs = (
        ("Connect Four: win now", start["connect4"]["win"], end["connect4"]["win"]),
        ("Connect Four: block the threat", start["connect4"]["block"], end["connect4"]["block"]),
        ("Connect Four: total", start["connect4"]["total"], end["connect4"]["total"]),
        ("GSM8K (math)", start["gsm8k"], end["gsm8k"]),
    )
    rows = ["| Task | Start | End | Δ (pp) |", "|---|---:|---:|---:|"]
    for label, before, after in pairs:
        if label == "Connect Four: total":
            label += f" (chance: {after['chance_pct']:.2f}%)"
        delta = after["accuracy_pct"] - before["accuracy_pct"]
        rows.append(
            f"| {label} | {pct(before['accuracy_pct'])} | "
            f"{pct(after['accuracy_pct'])} | {delta:+.1f} pp |"
        )
    return "\n".join(rows)


def table_mirror() -> str:
    rows = ["| Model | Original | Mirrored | Change (pp) | Discarded |", "|---|---:|---:|---:|---:|"]
    pairs = (
        ("SFT (move only)", "eval-sft.json", "mirror-sft.json"),
        ("World-model SFT + verification", "eval-mav-verified.json", "mirror-mav-verified.json"),
    )
    for label, original_file, mirrored_file in pairs:
        original, mirrored = load(original_file), load(mirrored_file)
        change = mirrored["best_correct_pct"] - original["best_correct_pct"]
        rows.append(
            f"| {label} | {pct(original['best_correct_pct'])} | "
            f"{pct(mirrored['best_correct_pct'])} | {change:+.1f} pp | "
            f"{mirrored.get('mirror_discarded', 0)} |"
        )
    return "\n".join(rows)


def table_cost() -> str:
    rows = ["| Step | Examples | Steps | Time (min) | Peak VRAM (GB) |", "|---|---:|---:|---:|---:|"]
    for label, filename in (
        ("World-model SFT, 30k", "train-mav-30k.json"),
        ("World-model SFT, +40k (continued)", "train-mav-70k.json"),
    ):
        data = load(filename)
        log = data["log"]
        final = log[-1] if isinstance(log, list) else log
        steps = final.get("step", final.get("epoch", "—"))
        minutes = final["train_runtime"] / 60
        rows.append(
            f"| {label} | {data['examples']} | {steps} | {minutes:.0f} | "
            f"{data['peak_vram_gb']:.2f} |"
        )
    return "\n".join(rows)


def table_other_methods() -> str:
    rows = ["| Method | Puzzles solved | Note |", "|---|---:|---|"]
    for label, filename in (
        ("SFT (move only)", "eval-sft.json"),
        ("SFT → GRPO (Stockfish reward)", "eval-sft-grpo.json"),
    ):
        data = load(filename)
        rows.append(f"| {label} | {pct(data['best_correct_pct'])} (n={data['n']}) | leak-free test split |")
    for data in load("other-methods.json"):
        rows.append(
            f"| {data['method']} | {pct(data['best_correct_pct'])} (n={data['n']}) | {data['note']} |"
        )
    return "\n".join(rows)


BUILDERS = {
    "{{START_VS_END}}": table_start_vs_end,
    "{{GAME_ANALYSIS}}": table_game_analysis,
    "{{DIFFICULTY}}": table_difficulty,
    "{{RECIPE}}": table_recipe,
    "{{LAYERS}}": table_layers,
    "{{INFERENCE}}": table_inference,
    "{{STAR}}": table_star,
    "{{GENERALIZATION}}": table_generalization,
    "{{MIRROR}}": table_mirror,
    "{{COST}}": table_cost,
    "{{OTHER_METHODS}}": table_other_methods,
}


def build_readme() -> str:
    for filename in EVIDENCE_FILES:
        load(filename)
    if not TEMPLATE.is_file():
        raise EvidenceError(f"required template is missing: {TEMPLATE}")
    text = TEMPLATE.read_text(encoding="utf-8")
    missing = [marker for marker in MARKERS if marker not in text]
    if missing:
        raise EvidenceError("template is missing required marker(s): " + ", ".join(missing))
    for marker, builder in BUILDERS.items():
        text = text.replace(marker, builder())
    remaining = [marker for marker in MARKERS if marker in text]
    if remaining:
        raise EvidenceError("unfilled template marker(s): " + ", ".join(remaining))
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate inputs and generated README without writing")
    args = parser.parse_args()
    try:
        generated = build_readme()
        if args.check:
            print("evidence, template markers, and generated README are valid")
        else:
            README.write_text(generated, encoding="utf-8")
            print(f"wrote {README}")
        return 0
    except (EvidenceError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"build_readme.py: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
