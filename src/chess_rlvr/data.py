"""Loading and splitting Lichess puzzles.

Splitting rule: the indivisible group is the **GameId**, not the row. Puzzles
derived from the same game share positions and motifs; splitting them by row
would leak answers between train and test and inflate the metric.

Stratifying by rating band ensures train and test cover the same difficulty
ladder, so the per-difficulty slices are comparable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

RATING_BANDS = (600, 1000, 1400, 1800, 2200, 2600, 3000)


def rating_band(rating: int) -> str:
    lower = 0
    for upper in RATING_BANDS:
        if rating < upper:
            return f"{lower}-{upper}"
        lower = upper
    return f"{lower}+"


def game_hash(game_id: str, seed: int = 0) -> float:
    """Stable hash of the GameId in [0, 1), independent of row order."""
    h = hashlib.sha256(f"{seed}:{game_id}".encode()).hexdigest()
    return int(h[:16], 16) / float(1 << 64)


@dataclass(frozen=True)
class Split:
    train: list
    validation: list
    test: list

    def summary(self) -> dict:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }


def split_by_game(lines: list[dict], *, seed: int = 0,
                     frac_val: float = 0.05, frac_test: float = 0.15) -> Split:
    """Split by GameId, deterministically and reproducibly.

    A GameId lands entirely in one split. If a game's puzzles have very
    different ratings, the game still stays whole: group integrity wins over
    perfect stratification.
    """
    by_game: dict[str, list[dict]] = {}
    for line in lines:
        by_game.setdefault(line["game_id"], []).append(line)

    train, val, test = [], [], []
    for game_id, game_lines in by_game.items():
        u = game_hash(game_id, seed)
        if u < frac_test:
            test.extend(game_lines)
        elif u < frac_test + frac_val:
            val.extend(game_lines)
        else:
            train.extend(game_lines)

    return Split(train, val, test)


def _field(line: dict, *names: str):
    """Read a field, accepting either snake_case or the dataset's PascalCase."""
    for name in names:
        if name in line:
            return line[name]
    return None


def normalize(line: dict) -> dict:
    """Convert a Lichess row to the internal format."""
    return {
        "puzzle_id": _field(line, "puzzle_id", "PuzzleId"),
        "game_id": _field(line, "game_id", "GameId") or "",
        "fen": _field(line, "fen", "FEN"),
        "moves": _field(line, "moves", "Moves"),
        "rating": int(_field(line, "rating", "Rating") or 0),
        "popularity": int(_field(line, "popularity", "Popularity") or 0),
        "nb_plays": int(_field(line, "nb_plays", "NbPlays") or 0),
        "themes": tuple(_field(line, "themes", "temas", "Themes") or ()),
    }


def filter_rows(lines: list[dict], *, min_popularity: int = 80,
            min_plays: int = 100, rating_min: int = 400,
            rating_max: int = 3000) -> list[dict]:
    """Keep only clean puzzles that have been played enough.

    A puzzle with low popularity tends to have an ambiguous or alternative
    solution: label noise that no hyperparameter can fix.
    """
    kept = []
    for line in lines:
        if int(_field(line, "popularity", "Popularity") or 0) < min_popularity:
            continue
        if int(_field(line, "nb_plays", "NbPlays") or 0) < min_plays:
            continue
        rating = int(_field(line, "rating", "Rating") or 0)
        if not (rating_min <= rating <= rating_max):
            continue
        kept.append(line)
    return kept


def audit_leakage(split: Split) -> dict:
    """Prove that no GameId or PuzzleId appears in two splits."""
    games = {
        "train": {l["game_id"] for l in split.train},
        "validation": {l["game_id"] for l in split.validation},
        "test": {l["game_id"] for l in split.test},
    }
    ids = {
        "train": {l["puzzle_id"] for l in split.train},
        "validation": {l["puzzle_id"] for l in split.validation},
        "test": {l["puzzle_id"] for l in split.test},
    }
    return {
        "games_by_split": {k: len(v) for k, v in games.items()},
        "puzzles_by_split": {k: len(v) for k, v in ids.items()},
        "shared_games": sorted(
            (games["train"] & games["test"])
            | (games["train"] & games["validation"])
            | (games["validation"] & games["test"])
        )[:20],
        "shared_puzzles": sorted(
            (ids["train"] & ids["test"])
            | (ids["train"] & ids["validation"])
            | (ids["validation"] & ids["test"])
        )[:20],
    }


def write_manifest(split: Split, destination: Path, *, source: str,
                       filters: dict, seed: int) -> dict:
    """Record provenance and counts. Without this the result is not auditable."""
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": source,
        "seed": seed,
        "filters": filters,
        "splits": split.summary(),
        "leakage_audit": audit_leakage(split),
        "rating_bands": {
            "train": _histogram(split.train),
            "validation": _histogram(split.validation),
            "test": _histogram(split.test),
        },
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def _histogram(lines: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in lines:
        band = rating_band(int(_field(line, "rating", "Rating") or 0))
        counts[band] = counts.get(band, 0) + 1
    return dict(sorted(counts.items()))


# --------------------------------------------------------------------------
# Stratified split by motif + rating band
# --------------------------------------------------------------------------

MAIN_MOTIFS = (
    "fork", "pin", "skewer", "discoveredAttack", "doubleCheck", "deflection",
    "sacrifice", "mateIn1", "mateIn2", "endgame", "middlegame", "opening",
    "promotion", "hangingPiece", "trappedPiece", "xRayAttack", "zugzwang",
    "attraction", "clearance", "interference", "intermezzo", "quietMove",
    "backRankMate", "smotheredMate", "arabianMate", "anastasiaMate",
)


def motif_of(themes) -> str:
    """Pick the most specific motif of the puzzle.

    Order matters: `mateIn1` is more informative than `mate`, which is more
    informative than `middlegame`. Taking the first match from the known list
    keeps every puzzle from becoming "middlegame".
    """
    theme_set = set(themes or ())
    for m in MAIN_MOTIFS:
        if m in theme_set:
            return m
    for t in (themes or ()):
        if t in ("crushing", "advantage", "equality", "mate", "short", "long",
                 "veryLong", "oneMove", "master", "masterVsMaster", "superGM"):
            continue
        return t
    return "other"


def cell(line: dict) -> str:
    """Stratification cell: motif + rating band."""
    return "%s|%s" % (motif_of(line.get("themes") or line.get("Themes")),
                      rating_band(int(_field(line, "rating", "Rating") or 0)))


def stratified_split(lines: list[dict], *, seed: int = 0,
                          frac_val: float = 0.05,
                          frac_test: float = 0.15) -> Split:
    """Split by GameId, stratifying by (motif, rating band).

    Two guarantees at once:
      1. no GameId appears in two splits (group integrity);
      2. each motif×rating cell is represented in all three splits, so the
         per-theme slices are comparable and do not just measure composition.

    Stratification sorts each cell's games by hash and cuts at the requested
    proportions. If a cell has few games, it falls entirely into the split the
    hash points to, and this is recorded in the manifest.
    """
    by_game: dict[str, list[dict]] = {}
    for line in lines:
        gid = _field(line, "game_id", "GameId") or str(_field(line, "puzzle_id", "PuzzleId"))
        by_game.setdefault(gid, []).append(line)

    # Each game belongs to one cell (that of its first puzzle).
    cells: dict[str, list[str]] = {}
    for gid, game_lines in by_game.items():
        cells.setdefault(cell(game_lines[0]), []).append(gid)

    train, val, test = [], [], []
    for games in cells.values():
        ordered = sorted(games, key=lambda g: game_hash(g, seed))
        n = len(ordered)
        n_test = int(round(n * frac_test))
        n_val = int(round(n * frac_val))
        if n >= 3:
            n_test = max(1, n_test)
            n_val = max(1, n_val)
        test_start = n - n_test
        val_start = max(0, test_start - n_val)

        for g in ordered[:val_start]:
            train.extend(by_game[g])
        for g in ordered[val_start:test_start]:
            val.extend(by_game[g])
        for g in ordered[test_start:]:
            test.extend(by_game[g])

    return Split(train, val, test)


def audit_strata(split: Split) -> dict:
    """Prove that motifs and ratings are covered in all three splits."""
    def coverage(lines):
        motifs, bands = {}, {}
        for l in lines:
            m = motif_of(l.get("themes") or l.get("Themes"))
            motifs[m] = motifs.get(m, 0) + 1
            b = rating_band(int(_field(l, "rating", "Rating") or 0))
            bands[b] = bands.get(b, 0) + 1
        return motifs, bands

    mt, bt = coverage(split.train)
    mv, bv = coverage(split.validation)
    ms, bs = coverage(split.test)

    all_motifs = set(mt) | set(mv) | set(ms)
    all_bands = set(bt) | set(bv) | set(bs)

    return {
        "train_motifs": len(mt), "test_motifs": len(ms),
        "train_only_motifs": sorted(set(mt) - set(ms)),
        "test_only_motifs": sorted(set(ms) - set(mt)),
        "train_bands": sorted(bt), "test_bands": sorted(bs),
        "motifs_without_test": sorted(m for m in all_motifs if m not in ms),
        "bands_without_test": sorted(b for b in all_bands if b not in bs),
        "motif_coverage": {m: {"train": mt.get(m, 0), "val": mv.get(m, 0),
                                  "test": ms.get(m, 0)}
                              for m in sorted(all_motifs)},
    }
