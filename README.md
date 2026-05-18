# neural-chess

A chess engine powered by a residual convolutional neural network, trained on a large corpus of human games. **Two architecture generations live in this repo:**

- **v1** — 12-plane input, 10×256 residual CNN, ~12M params, trained on 22M positions from ~800k historical games. Policy head only. [Architecture details below](#v1-architecture-original).
- **v2** — 21-plane history-aware input, 16×256 residual CNN + wider policy/value heads, ~19M params, trained on a 40M-position license-clean mixed-Elo corpus. Adds a value head and a board-rotation trick so the model can play both colors. [Architecture details below](#v2-architecture-current).

Both versions share the same `PolicyEngine` runtime interface so a single `play.py` / `uci.py` works for either. Version is auto-detected from the checkpoint.

## Project principles

Three load-bearing rules govern every design decision:

1. **Human games only.** Training data must come from human-played games. No engine games, no self-play, no engine-annotated labels, no pretrained engine weights.
2. **Pen-and-paper signal only.** Supervision targets are limited to what's observable in the games (positions, moves played, outcomes). No computed chess features (material counts, pawn-structure heuristics, king-safety scores, mobility, etc.) — the model must learn what matters from raw observation.
3. **Single forward pass.** Inference is one forward pass producing one move. No external search loops (MCTS, alpha-beta, "run N times pick best").

These rules cap the realistic strength ceiling (no MCTS means no Magnus-level play) but make the artifact interesting on its own terms: how strong can a forward-pass-only model trained on human games get?

---

## v2 architecture (current)

### Input: 21 planes, history-aware

The input is `(21, 8, 8)`:

| Planes | What | Why |
|-------:|------|-----|
| 0–5 | Own pieces (P, N, B, R, Q, K) | Standard piece encoding |
| 6–11 | Opponent pieces (P, N, B, R, Q, K) | Independent from own pieces — cleaner gradients than sign-encoding |
| 12–15 | Castling rights (K/Q each color, broadcast) | Pen-and-paper observation; needed for legal-move semantics the model would otherwise miss |
| 16 | En passant target square (one-hot) | Same — passes Principle 2 trivially |
| 17 | Halfmove clock (normalized, broadcast) | For 50-move rule awareness |
| 18 | Fullmove number (normalized, broadcast) | Game-phase signal |
| 19 | Side to move (broadcast 0/1) | Even after rotation (below), the value head benefits |
| 20 | Repetition count (broadcast) | Detects threefold repetition |

When black is to move, the **board is mirrored vertically** (rank flip — `python-chess`'s `board.mirror()`) and the player's planes are swapped. This is a pure coordinate transformation (no chess theory smuggled in — passes Principle 2) but is essential for the model to learn both colors with one set of weights.

### Output: AlphaZero-style 8×8×73 move encoding

The policy head produces `(8, 8, 73) = 4,672` logits — a per-from-square plane of 73 move *types*:
- 56 sliding moves (8 directions × 7 max distance)
- 8 knight moves
- 9 underpromotions (3 promo types × 3 capture directions)

Legal-move masking renormalizes the distribution before sampling/argmax. The same encoding lets the model represent any chess move (including promotions and castling) without special-casing.

### Model: ResNet encoder + policy head + value head

```
Input (21, 8, 8)
     │
     ▼
Input stem: Conv 21→256, 3×3 + BN + ReLU
     │
     ▼
Residual tower: 16 blocks × 256 ch
   each block: Conv → BN → ReLU → Conv → BN → (+skip) → ReLU
     │
     ├──────────────────────┐
     ▼                      ▼
Policy head:                Value head:
  Conv 256→64, 1×1            Conv 256→4, 1×1
  + BN + ReLU                 + BN + ReLU
  → Conv 64→73, 1×1           → flatten → Linear(4·64, 128)
  → (8, 8, 73) logits         → ReLU → Linear(128, 1) → tanh
                              → scalar in [-1, 1]
```

Parameter budget (the current `T_FINAL` config, ~19M total):

| Stage | Approx params | Share |
|-------|--------------:|------:|
| Input stem | 48,640 | 0.3% |
| 16 residual blocks × 256 ch | ~18.9M | 99.4% |
| Policy head (64 mid + 73 out, 1×1 convs) | ~21k | 0.1% |
| Value head (4 mid + 4·64→128→1 dense) | ~33k | 0.2% |
| **Total** | **~19.0M** | **100%** |

### Training signal

Two losses combined per batch:
- **Policy**: cross-entropy of the legal-move-masked logits against the move actually played (1-of-4672 label).
- **Value**: MSE against the game result from the side-to-move's perspective: `+1` for win, `-1` for loss, `0` for draw.

No auxiliary heads, no engine-derived signal — see Principle 2.

### What v2 changes vs v1

| | v1 | v2 |
|--|------|------|
| Input planes | 12 (pieces only) | 21 (pieces + castling + EP + clocks + side + repetition) |
| Output | 4,096 (from, to) | 4,672 (8×8×73 AlphaZero-style) |
| Plays both colors | Half (board rotation in dataset, no value head) | Yes (mirror trick + value head) |
| Value head | No | Yes (scalar tanh) |
| Castling / EP awareness | No (only via legal-move mask geometry) | Yes (explicit input planes) |
| Training filter | Decisive games only, winner's moves only | All games (win/loss/draw), both sides' moves, tier-mixed Elo |
| Dataset size | 22M positions, 800k games | 40M positions, ~490k games (3-tier mix) |

The most important practical effect is that v2 plays both colors correctly with one model — v1 effectively only learned white because the dataset was filtered to winner's moves and the board rotation didn't expose enough black-as-moving-player signal.

---

## v1 architecture (original)

The v1 model is a 12-plane residual CNN with a single policy head. Documented in detail in earlier commits of this README; the short version:

- **Input** (12, 8, 8): own/opponent piece planes, sign-encoded board rotated for black.
- **Tower**: 10 residual blocks × 256 channels (~11.8M params).
- **Policy head**: two 1×1 convs (256→32→64) reshaped into `(64, 8, 8) = 4096` (from, to) logits.
- **No value head, no castling/EP planes.**
- Trained for ~50 epochs on `data/v1/*.npz` (22M positions); cosine LR schedule; FP16 mixed precision.

The trained v1 best checkpoint serves as the head-to-head reference opponent during v2 evaluation.

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows (use bin/activate on Linux/macOS)

pip install -r requirements.txt
```

For CUDA acceleration (recommended for training):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

You will also need [Stockfish](https://stockfishchess.org/download/) installed at `bin/stockfish.exe` (Windows) or `bin/stockfish` for engine-vs-engine evaluation. The repo ships with `bin/stockfish.exe` for Windows.

---

## Data

The `data/` and `model/` directories are **gitignored** — too large to push, and easy to regenerate. Use the instructions below to reconstruct.

### v1 data (legacy)

Pre-parsed NPZ files in `data/v1/*.npz`, one per year/era. Each file is a `np.savez_compressed` archive containing:

| Array | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `X` | `(N, 384)` | int8 | Board features — 6 sign-encoded piece planes of 8×8, flattened |
| `Y` | `(N, 4096)` | int8 | One-hot move labels over (from, to) pairs |
| `meta` | `(1,)` | int32 | Row count |

Reproducing v1 data from raw PGN: drop `.PGN` files into `data/v1/` and run:

```bash
python src/v1/parse_pgn.py
```

This writes `data/v1/<name>.npz` next to each `.PGN`. Filter: decisive games only, winner's moves only, board rotated for black-to-move. Draws and en passant / castling rights are discarded.

### v2 data — full reconstruction guide

The v2 pipeline is **three sequential steps** living in `data/v2/_acquire/`. They are deliberately idempotent and resumable — partial failures pick up where they left off.

#### Step 0 — Disk budget

| Stage | Approx size |
|-------|------------:|
| Raw archives (Lichess Elite + monthly + TWIC zips/zsts) | ~6 GB |
| Filtered tier PGNs | ~1.1 GB |
| 40M-position training shard (binary memmap) | ~10.8 GB |
| 8M shard (smaller smoke-test) | ~2.2 GB |

Reserve **~25 GB** of free disk space for a comfortable v2 reconstruction.

#### Step 1 — Download raw sources

```bash
python data/v2/_acquire/download.py
```

Downloads (per [data/v2/README.md](data/v2/README.md)):
- **Lichess Elite** (nikonoel) — all monthly archives. CC0.
- **Lichess monthly** (`database.lichess.org`) — three most recent months of standard rapid+. CC0.
- **TWIC** (theweekinchess) — issues 1500 through latest. "Personal use" license — do not redistribute the raw PGNs or the filtered output.

Progress is checkpointed to `data/v2/_acquire/download_progress.json`; re-running resumes incomplete downloads.

> ⚠ Hardcoded paths: the acquire scripts use absolute paths under `D:/dev/neural-chess/data/v2`. If your repo lives elsewhere, edit `ROOT`/`RAW` constants at the top of each script.

#### Step 2 — Filter, dedup, index

```bash
python data/v2/_acquire/filter.py
```

Streams the raw archives (priority: Elite > TWIC > Lichess monthly for dedup), applies filters:

| Filter | Criterion | Why |
|--------|-----------|-----|
| Both Elos parse | `WhiteElo`/`BlackElo` integer | Need rating for tier assignment |
| Elo agreement | Both within ±400 of each other | Mismatched-strength games dominated by gap |
| Time control | Initial ≥ 180 seconds | Skip bullet; speed-dominated |
| Termination | `Normal` only | Drop time forfeits, rule violations |
| Result | `1-0` / `0-1` / `1/2-1/2` | Need definite outcome for value head |
| Length | ≥ 10 plies | Skip noise-resigned games |
| Variant | Standard chess only | We don't model variants |

Writes:
- `data/v2/filtered/tier_top_2400plus.pgn` (Elite + TWIC top)
- `data/v2/filtered/tier_mid_1900-2400.pgn` (Lichess monthly mid)
- `data/v2/filtered/tier_low_1600-1900.pgn` (Lichess monthly low)
- `data/v2/games_index.parquet` — per-game metadata (Elos, source, result)
- `data/v2/manifest.json` — counts + provenance

Progress checkpointed to `_acquire/filter_progress.json`. Resumable.

#### Step 3 — Generate training shards (featurize)

```bash
# 8M smoke-test shard (smaller, faster iteration; ~5 min to generate)
python -m src.v2.dataset --out-dir data/v2/training_T1_rot --positions 8000000

# 40M production shard (used to train current v2-final; ~25 min to generate)
python -m src.v2.dataset --out-dir data/v2/training_T1_rot_40M --positions 40000000
```

Reads from `data/v2/filtered/`, applies 21-plane featurization with the rotation trick, and writes three memory-mapped binary files:

```
data/v2/training_T1_rot_40M/
    X.bin          # int8,  shape = (n, 21, 8, 8) — 21-plane input
    Y_policy.bin   # int32, shape = (n,)         — move index (0..4671)
    Y_value.bin    # int8,  shape = (n,)         — outcome from side-to-move's view (+1/0/-1)
    meta.json      # {n_samples, tier_counts, ...}
```

Tier mix defaults to 65% top / 25% mid / 10% low (configurable via `--top-pct` / `--mid-pct` / `--low-pct`). The generator reads PGN games in file order; the same `--positions` target with the same `--seed` produces deterministic shards, and a larger target is a strict superset of a smaller one.

#### Validate

```bash
python -m src.v2.test_dataset_hardening   # 5 smoke tests; takes ~4 sec
```

---

## Training

### v2 (current)

```bash
python -m src.v2.train \
    --shard-dir data/v2/training_T1_rot_40M \
    --save-dir model/v2/my_run \
    --epochs 12 --max-epochs 10 \
    --batch-size 1024 \
    --blocks 16 --channels 256 \
    --policy-channels 64 --value-channels 4 --value-hidden 128 \
    --keep-last-n 5
```

Notes:
- `--epochs` is the **cosine LR horizon**. Training continues until `--max-epochs` (or a `.stop` file appears in the project root).
- Auto-resumes from the latest checkpoint in `--save-dir`. Pass `--no-resume` to start fresh.
- Mixed-precision FP16 by default; ~85 min/epoch on a 4080 Super for the 40M / 16×256 config.
- Checkpoints are dicts `{model, optimizer, scheduler, epoch, arch, config}` — resume restores full training state.

Full flag reference:

| Flag | Default | Description |
|------|---------|-------------|
| `--shard-dir` | (required) | Directory with `X.bin`/`Y_policy.bin`/`Y_value.bin`/`meta.json` |
| `--save-dir` | (required) | Checkpoint output directory |
| `--save-name` | `model` | Checkpoint name prefix (`model_e0001.pt` etc.) |
| `--batch-size` | 1024 | |
| `--lr` | 1e-3 | Peak LR; cosine-decays toward 0 over `--epochs` |
| `--weight-decay` | 1e-4 | AdamW weight decay |
| `--epochs` | 50 | Cosine LR horizon |
| `--max-epochs` | 999 | Hard stop after this many epochs |
| `--blocks` | 12 | Residual tower depth |
| `--channels` | 256 | Channels per residual block |
| `--policy-channels` | 32 | Policy head 1×1 conv compression width |
| `--value-channels` | 1 | Value head 1×1 conv compression target |
| `--value-hidden` | 64 | Value head MLP hidden width |
| `--lookahead-k` | 0 | (Experimental, disabled by default) Differentiable lookahead branching factor — ablation showed no benefit, see `memory/future-architecture-roadmap.md` |
| `--value-weight` | 1.0 | Loss combiner: `loss = policy_ce + value_weight * value_mse` |
| `--keep-last-n` | 0 | If > 0, keep only the N most recent checkpoints |
| `--resume` | (auto) | Explicit checkpoint path |
| `--no-resume` | off | Ignore existing checkpoints |
| `--stop-file` | `.stop` | Path of the graceful-exit sentinel |

### v1 (legacy)

```bash
python src/v1/train.py
```

Runs until you create a `.stop` file. Auto-resumes from latest in `model/v1/`. Same `--resume` / `--no-resume` semantics. See `python src/v1/train.py --help` for the full v1 flag list.

---

## Playing

The `play.py` CLI works with both v1 and v2 checkpoints — version is auto-detected from the file.

### vs Stockfish

```bash
python play.py model/v2/T_FINAL/model_e0005.pt engine --games 10 --depth 5 --skill 10
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `-n`, `--games` | 10 | Games to play |
| `-d`, `--depth` | 0 | Stockfish search depth |
| `-s`, `--skill` | 0 | Stockfish skill level (0–20) |
| `--color` | white | Color the model plays |
| `--stockfish` | `bin/stockfish.exe` | Path to Stockfish executable |
| `-t`, `--temperature` | 0.5 | Starting softmax temperature (0 = greedy argmax) |
| `--temp-decay` | 0.05 | `temp = start * exp(-decay * ply)` |

### Interactive

Play in the terminal (you're white, model is black):

```bash
python play.py model/v2/T_FINAL/model_e0005.pt interactive
```

UCI move notation (`e2e4`, `g1f3`, `e7e8q` for promotions). Type `quit` to exit.

### Session-based play (for AI agents)

`src/play_session.py` is a stateful one-move-at-a-time harness suitable for an external player (a human at the CLI, or an LLM agent) submitting moves between separate process invocations:

```bash
# Start a game; agent plays white, neural-chess plays black
python -m src.play_session start \
    --white "agent:my-label" \
    --black "neural-chess:model/v2/T_FINAL/model_e0005.pt" \
    --game-id mygame

# Submit moves one at a time
python -m src.play_session move --game-id mygame --uci e2e4

# Show current state
python -m src.play_session show --game-id mygame
```

Player specs: `interactive` / `agent[:LABEL]` / `neural-chess:<path>` / `stockfish[:depth[:skill[:path]]]`. State persists in `tmp/play_sessions/<game-id>.json`.

---

## Evaluation

### v2 — SF ladder + h2h vs v1

```bash
python eval/v2/eval_v2.py <epoch> \
    --save-dir model/v2/T_FINAL --save-name model \
    --eval-csv eval/v2/eval_T_FINAL.csv \
    --no-gate
```

Runs five Stockfish tiers in sequence (100-game splits for the lower three, 10-game for the upper two), then a head-to-head against the v1 best (`model/v1/checkpoints/model_e0009.pt` by default):

| Tier | Depth | Skill | Games |
|------|------:|------:|------:|
| sf_easy | 1 | 0 | 100 |
| sf_med | 3 | 5 | 100 |
| sf_hard | 5 | 10 | 100 |
| sf_magnus | 8 | 20 | 10 |
| sf_ultra | 16 | 20 | 10 |

By default, higher tiers are gated by ≤10% loss at the previous tier (a sanity guard). `--no-gate` runs all tiers regardless. Use `--skip-h2h` or `--skip-sf` to limit which sections run.

Results append to the per-model CSV with the schema `epoch,timestamp,opponent,games,won_pct,draw_pct,lost_pct,white_won_pct,black_won_pct,illegal_pct,moves,elapsed_s`.

### v1 — multi-checkpoint vs SF

```bash
python eval/v1/evaluate.py            # all checkpoints, default settings
python eval/v1/evaluate.py --quick    # 1 checkpoint, 5 games per setting
python eval/v1/evaluate.py --models 5 # evenly spaced 5 checkpoints
```

Other v1 analysis scripts (opening diversity, castling/EP rates, temperature sweep) live in `eval/v1/`.

---

## UCI / Arena GUI

To use with a UCI-compatible chess GUI like [Arena](http://www.playwitharena.de):

1. **Engines → Manage → New**
2. Browse to `.venv\Scripts\python.exe`
3. **Command Line Parameters**: `<repo_root>\uci.py <path_to_checkpoint.pt>`
4. Both v1 and v2 checkpoints work; both colors supported.

---

## Repository layout

```
neural-chess/
├── play.py                  # play vs SF or interactive (v1+v2 dispatch via inference_api)
├── uci.py                   # UCI driver for Arena/other GUIs
├── requirements.txt
├── bin/                     # Stockfish + pgn-extract binaries (Windows)
├── data/                    # gitignored — see "Data" section
│   ├── v1/                  # legacy NPZ training files
│   └── v2/
│       ├── README.md        # canonical v2 dataset spec
│       ├── _acquire/        # download.py / filter.py / etc.
│       ├── raw/             # raw downloaded archives
│       ├── filtered/        # tier_top/mid/low.pgn after filter step
│       ├── training_T1_rot/        # 8M shard
│       └── training_T1_rot_40M/    # 40M shard
├── model/                   # gitignored — checkpoints per version
│   ├── v1/checkpoints/
│   └── v2/T_FINAL/          # current v2 run
├── src/
│   ├── inference_api.py     # PolicyEngine abstract base + load_policy_engine factory
│   ├── engine.py            # Stockfish/UCI helpers
│   ├── game_loop.py         # engine-vs-engine game runner
│   ├── stats.py             # play stats aggregator
│   ├── uci_protocol.py      # UCI protocol implementation for uci.py
│   ├── play_session.py      # stateful session-based play harness
│   ├── v1/
│   │   ├── model.py / featurize.py / dataset.py / train.py / inference.py / parse_pgn.py
│   └── v2/
│       ├── model.py / featurize.py / moves.py / dataset.py / train.py / inference.py
│       ├── lookahead.py     # disabled experimental lookahead block (ablated, kept for reference)
│       └── test_dataset_hardening.py
├── eval/
│   ├── v1/                  # evaluate.py, eval_one.py, opening/castling/temp analyses
│   └── v2/                  # eval_v2.py + per-model CSVs
├── logs/                    # gitignored — training/eval logs
└── tmp/                     # gitignored — play_sessions/, scratch
```

---

## Provenance and license

- **Code**: see repo license.
- **v1 training data**: derived from open historical PGNs (TWIC + Lichess) — derived NPZ shards may be redistributed.
- **v2 training data**: mixed sources. Lichess content is CC0; **TWIC content is "personal use only" and the filtered tier PGNs / shards derived from it must NOT be redistributed wholesale.** Trained weights derived from this mix are personal-use only.
- **No engine-derived training signal** (Stockfish evaluations, MCTS rollouts, etc.) is used anywhere in either version, per the project principles.
