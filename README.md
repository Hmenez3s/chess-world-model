# chess-world-model

Teaching a small LLM (Qwen2.5-1.5B-Instruct) to **model the chess board before choosing a
move**. LoRA fine-tuning on a single **RTX 3060 (12 GB)**; nothing trained from scratch.

## Start vs. end

Leak-free test split (split by game), 600 Lichess puzzles, same sample for both models.

| | Start: Qwen2.5-1.5B-Instruct | End: world-model SFT + legal-move verification |
|---|---|---|
| Puzzles solved (with 95% CI) | 0.00% [0.00, 0.64] | 16.00% [13.28, 19.15] |
| Legal move rate | 3.33% | 99.17% |
| Board after the move (exact) | cannot | 93.83% |
| Games vs random player (W/D/L) | 0/6/0 | 1/5/0 |
| Games vs Stockfish `UCI_Elo=1320` (W/D/L) | 0/1/7 | 0/0/8 |
| Illegal moves in games | 96.31% | 5.21% |

**What improved:** the model now reads and updates the board (93.83% exact), almost never
plays an illegal move, and solves 16% of tactical puzzles instead of none.

**What did not:** it still loses every game to Stockfish at its weakest setting. Its real
playing strength is **below 1320 Elo** (0/8 games; the 95% upper bound is about 1193).

### Why it solves puzzles but loses games

| Metric | Value |
|---|---|
| Games | 8 |
| Model moves | 192 |
| Blunders (≥ 300 cp) | 30 |
| One blunder every N moves | 6.4 |
| Mean move of first blunder | 5.1 |
| Games with a blunder by move 5 | 7 of 8 |

It blunders (loses three pawns or more) about once every six moves, and in 7 of 8 games the
first blunder comes **within the first five moves**. The training data are puzzle positions —
middlegame and endgame tactics — so the model never saw an opening. A puzzle asks for one
forcing move in a position that is known to contain a tactic; a game asks for dozens of
moves without a single blunder.

### A note on "puzzle Elo"

| Rating band | n | Model | Expected for a 1460-rated solver |
|---|---:|---:|---:|
| 1200–1599 | 120 | 21.67% | 43.99% |
| 1600–1999 | 308 | 15.26% | 12.57% |
| 2000–2399 | 172 | 13.37% | 2.61% |

A rated solver gets fewer puzzles right as they get harder. This model barely does: its
accuracy is almost flat across difficulty. Fitting an Elo to the 16% it solves on this
(hard, mean rating 1837) test set gives ~1460, but that number depends on how hard the set
is and **does not measure playing strength**. It is reported in the per-layer table only, for
comparison between training steps.

## Training recipe

| Step | Puzzles solved | Puzzle Elo (comparative) |
|---|---:|---:|
| Qwen2.5-1.5B-Instruct (no training) | 0.00% | ≤ 825 (0 solved) |
| SFT (move only) | 4.67% | 1190 [1109, 1254] |
| SFT → GRPO (Stockfish reward) | 6.17% | 1246 [1180, 1302] |
| World-model SFT, 30k positions | 8.00% | 1300 [1237, 1353] |
| World-model SFT, 70k positions | 13.67% | 1421 [1370, 1475] |
| **+ legal-move verification (final)** | 16.00% | 1460 [1408, 1509] |
| + GRPO (not significant) | 16.83% | 1473 [1421, 1518] |

Supervising only the final move (SFT) and then reinforcing it (GRPO) barely moves the
rating: the model learns the *format* of an answer, not the game. The jump comes from
**world-model SFT**.

## Method: world-model SFT

Adapted from *Mastering Board Games by External and Internal Planning with Language Models*
(DeepMind, [arXiv 2412.12119](https://arxiv.org/abs/2412.12119)). Instead of asking for the
move alone, each training example asks the model to write out the consequences of the
previous move. Input:

```
Previous FEN: r4rk1/2q3p1/p1bbp3/1p3p2/4NP2/5Q1P/PPPB2P1/4RRK1 w - - 0 20
Move played: e4d6
```

Target:

```
board:
8 r . . . . r k .
7 . . q . . . p .
6 p . b N p . . .      <- the knight left e4 and captured the bishop on d6
5 . p . . . p . .
4 . . . . . P . .      <- e4 is now empty
3 . . . . . Q . P
2 P P P B . . P .
1 . . . . R R K .
  a b c d e f g h
to move: black
legal: a6a5 a8a7 ... c6f3 ... c7d6 ... (35 moves)
top5: c6f3:85 c7d6:40 a8d8:31 f8d8:31 f8f6:25
best: c6f3
```

1. **State transition** — the board after the move (uppercase = white, lowercase = black).
2. **Legal moves** of the new position.
3. **Action values** — win probability for the top 5 moves (Stockfish labels, used only to
   build training data).
4. **Best move.**

The best move here, `c6f3` (bishop takes queen), only exists because the knight **left e4**
and opened the diagonal. A model that does not update the board cannot see it — which is why
supervising the state improves the choice. Predicting a value per move instead of only the
best move is also what [ChessBench](https://arxiv.org/abs/2402.04494) found to work best
(83.3% vs 65.7% puzzle accuracy in the same model).

**Legal-move verification.** The model writes its own legal-move list and top-5 before the
answer, so an illegal `best:` can be repaired from its own output: take the first legal move of
its top-5, else the first legal move of its list. No external information is used. In games
this cuts illegal moves from 35.86% to 5.21%.

### Per-layer results

| Step | Board exact | Color | Legal exact (F1) | Best = solution (CI) |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct (no training) | n/a | n/a | n/a | 0.00% [0.00, 0.64] |
| SFT (move only) | n/a | n/a | n/a | 4.67% [3.25, 6.66] |
| SFT → GRPO (Stockfish reward) | n/a | n/a | n/a | 6.17% [4.51, 8.38] |
| World-model SFT, 30k positions | 86.33% | 99.71% | 0.00% (0.555) | 8.00% [6.09, 10.45] |
| World-model SFT, 70k positions | 93.83% | 99.90% | 5.17% (0.799) | 13.67% [11.15, 16.65] |
| **+ legal-move verification (final)** | 93.83% | 99.90% | 5.17% (0.799) | 16.00% [13.28, 19.15] |
| + GRPO (not significant) | 93.67% | 99.90% | 5.00% (0.797) | 16.83% [14.05, 20.04] |

## Inference-time compute

Same model, no extra training, 200 test puzzles.

| Decoding | Accuracy | 95% CI |
|---|---:|---:|
| Greedy | 18.00% | [13.29, 23.91] |
| Sampled (T = 0.8) | 17.50% | [12.86, 23.36] |
| Self-consistency (majority of 6) | 20.00% | [15.05, 26.09] |
| Depth-2 minimax with own values | 14.00% | [9.87, 19.49] |
| pass@6 (upper bound for RL / rejection sampling) | 42.50% | [35.85, 49.43] |

- **Self-consistency helps a little** (majority of 6 samples).
- **Searching with its own value estimates hurts.** Depth-2 minimax, where the model scores
  the opponent's replies with its own top-5 values, turns more right answers into wrong ones
  than the reverse (13 vs 5). The world model is accurate enough to *simulate* a move, not
  yet accurate enough to *evaluate* the opponent's position.
- **pass@6 is 42.5%.** The correct move is already among 6 sampled answers more than twice
  as often as greedy decoding finds it. So the model can *generate* the right move; it cannot
  reliably *choose* it. That gap is what search or self-training is supposed to close.

### Can the model learn from its own reasoning?

If the right move shows up when sampling, the trace that found it can be used as training
data — rejection-sampling fine-tuning (STaR / expert iteration). One iteration, same
base adapter, same budget:

| Model | Puzzles solved (test, n=600) | 95% CI |
|---|---:|---:|
| End model, no extra training | 16.00% | [13.28, 19.15] |
| + STaR on its own correct traces | 17.50% | [14.67, 20.74] |
| + control: same budget, original data | 17.17% | [14.36, 20.39] |

Training puzzles sampled: 800, 4 answers each (T = 0.8). Correct answers sampled: 11.31%. Usable traces kept: 218 (27.25% of the puzzles).

A trace was kept only if the model's own board was exactly right **and** its raw `best:` was
the solution. The gain is not distinguishable from the control, which is the same training
budget spent on the original data. The reason is in the sampling column: on training puzzles
the model produced a usable trace for only about a quarter of them, so there is little
correct reasoning to learn from. More sampling would raise that yield, but the ceiling is the
same 42.5% pass rate — and the control result says the extra training, not the self-generated
reasoning, is what moved the number.

**What the reasoning evidence says together:** the world-model trace is what carries the model
from 0% to 16%, and it is the reason its board state is 94% exact. But its *move values* are
not good enough to support choosing between alternatives: minimax on its own values loses
ground, and self-training on its own successes adds nothing over plain training. Fixing that
means better value estimates — which needs value data the model does not currently have —
not more search over the values it has.

## Generalization

Same battery for the start and end models. Connect Four is a game the model never saw (rules
in the prompt); GSM8K measures forgetting.

| Task | Start | End | Δ (pp) |
|---|---:|---:|---:|
| Connect Four: win now | 15.00% | 20.00% | +5.0 pp |
| Connect Four: block the threat | 17.00% | 9.00% | -8.0 pp |
| Connect Four: total (chance: 15.28%) | 16.00% | 14.50% | -1.5 pp |
| GSM8K (math) | 51.00% | 20.50% | -30.5 pp |

- **No transfer to a new game.** On Connect Four the final model stays at chance.
- **Catastrophic forgetting on math.** GSM8K drops by more than half; the model starts
  answering math questions in the chess format (`board:` followed by a "grid"). Mixing in
  general data was not tested.

**Memorization check:** on mirrored positions (same tactic, flipped coordinates and colors)
accuracy does not drop, so the model did not memorize the puzzles.

| Model | Original | Mirrored | Change (pp) | Discarded |
|---|---:|---:|---:|---:|
| SFT (move only) | 4.67% | 4.67% | +0.0 pp | 0 |
| World-model SFT + verification | 16.00% | 17.33% | +1.3 pp | 0 |

## Cost

| Step | Examples | Steps | Time (min) | Peak VRAM (GB) |
|---|---:|---:|---:|---:|
| World-model SFT, 30k | 30000 | 1875 | 148 | 5.41 |
| World-model SFT, +40k (continued) | 71704 | 2200 | 174 | 5.41 |

## Limitations

- **Small scale.** 70k positions and LoRA on a 1.5B model; reference work uses hundreds of
  millions of positions and trains from scratch.
- **Weak in games.** Playing strength is below 1320; it blunders about once every six moves,
  mostly in the opening, which the puzzle-only training never covered. Training on positions
  from full games is the obvious fix and was not done.
- **No learned search yet.** Search with the model's own values hurts (see above); training on
  search traces (MAV-IS, Searchformer, Stream of Search) was not done.
- **Negative generalization.** No transfer to Connect Four; forgetting on GSM8K.

## Reproduce

```bash
pip install -e ".[train]"
export STOCKFISH_PATH=~/bin/stockfish

# 1. leak-free splits from the Lichess open database
python scripts/prepare_splits.py --limit 0

# 2. world-model data (train split only), labeled by Stockfish
python scripts/generate_mav.py --positions 32000 --workers 12 --out data/mav-32k.jsonl
python scripts/generate_mav.py --positions 40000 --skip 32000 --workers 12 --out data/mav-72k.jsonl
cat data/mav-32k.jsonl data/mav-72k.jsonl > data/mav-all.jsonl

# 3. training (RTX 3060: ~2.5 h + ~2.9 h)
python scripts/train_sft.py --base-adapter none --data data/mav-32k.jsonl \
       --samples 30000 --batch 8 --grad-accum 2 --lr 2e-4 --max-length 512 --out out/mav-30k
python scripts/train_sft.py --base-adapter out/mav-30k --data data/mav-all.jsonl \
       --batch 8 --grad-accum 2 --lr 1e-4 --max-length 512 --max-steps 2200 --out out/mav-final

# 4. start vs. end on the same test split
python scripts/evaluate.py --format direct --puzzles 600 --out out/eval-base.json
python scripts/evaluate.py --adapter out/mav-final --puzzles 600 --verify --out out/eval-mav-verified.json
python scripts/puzzle_elo.py out/eval-base.json out/eval-mav-verified.json
python scripts/play_games.py --adapter out/mav-final --opponent stockfish --elo 1320 \
       --games 8 --verify --out out/games-stockfish-verified.json

# 5. README is generated from the JSON files in out/ (fails if any is missing)
python scripts/build_readme.py
```

The trained adapter (LoRA r=64, 275 MB) is attached to the [release](../../releases):

```bash
tar xzf adapter-mav-final.tar.gz -C out/mav-final
python scripts/evaluate.py --adapter out/mav-final --puzzles 50 --verify
```

## Other methods tried

| Method | Puzzles solved | Note |
|---|---:|---|
| SFT (move only) | 4.67% (n=600) | leak-free test split |
| SFT → GRPO (Stockfish reward) | 6.17% (n=600) | leak-free test split |
| SFT + free-text chain of thought | 9.17% (n=600) | earlier sample, before the leak-free split |
| Chain of thought + GRPO | 9.67% (n=600) | earlier sample, before the leak-free split |
| Checks-and-captures sweep + GRPO | 9.17% (n=600) | earlier sample, before the leak-free split |

- **GRPO on the bare answer** raises legality but not accuracy: the model optimizes the easy
  part of the reward.
- **Free-text chain of thought** ("situation, candidates, comparison") doubles accuracy within
  the same model but does not beat plain SFT: the model writes a plausible comparison without
  reading the board.
- **Teaching a checks-and-captures sweep** puts the solution in the candidate list more often
  but makes the final choice worse.

All of these target the choice. World-model SFT targets reading the board, and it is the only
one that moved the rating.

## Data and licenses

Puzzles: [Lichess open database](https://database.lichess.org/#puzzles) (CC0). Labels generated
with [Stockfish](https://stockfishchess.org/) (GPLv3), used only to build data and measure; not
distributed here. Code: MIT.
