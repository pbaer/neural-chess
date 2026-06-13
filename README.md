# neural-chess

> 🎮 **[Play the model & explore it live in your browser → pbaer.github.io/neural-chess](https://pbaer.github.io/neural-chess/)** — the *Neural Chess* web tool: play the tiny v3.1 model with one-shot inference, and inspect every weight and activation from the architecture diagram down to a single scalar.

A chess engine trained on a large corpus of human games. **Three architecture generations live in this repo:**

- **v1** — 12-plane input, 10×256 residual CNN, ~12M params, trained on 22M positions from ~800k historical games. Policy head only. [Architecture details below](#v1-architecture-original).
- **v2** — 21-plane history-aware input, residual CNN + policy **and value** heads, a board-rotation trick so the model plays both colors, and a license-clean mixed-Elo corpus. Scaled across several sizes; best v2 is **`v2-37M`** (20×320, ~37M params, 100M positions). [Architecture details below](#v2-architecture-prior-generation).
- **v3 (current)** — same 21-plane input/output, but the CNN tower is replaced by a **square-token attention tower**: transformer blocks over the 64 squares with a learned 2D board-geometry bias. The current **v3.1** form is a *pure* transformer with no conv stem (the first big models used a small conv stem before ablations showed it's dead weight). Markedly more parameter-efficient: `v3-18M` beats the 2×-larger v2-37M, and **`v3-37M`** (37.4M, acc 62.70%) is the strongest single model. [Architecture details below](#v3-architecture-current).

A separate, validated win is the **"tau" data recipe** — a position-aggregated corpus (averaged value + soft-policy move histogram + frequency-tempered sampling) that adds ~+79 Elo at equal params; **`v3-18M-tau` ≈ `v3-37M` at half the parameters**. [Details below](#the-tau-data-recipe-position-aggregated). Naming: `v3-<paramsize>-tau`.

- **v3.1 — a leaner architecture (no conv stem).** Teaching-model work showed the v3 conv stem is droppable: a **pure square-token transformer** (`stem_kernel=1, stem_blocks=0`) is *smaller and stronger* at small scale, and a full ablation pass ("v3.2") confirmed the result is tight — every remaining component is load-bearing. **v3.1 is the validated lean architecture** and the recommended shape for the next big model (`v3.1-37M-tau`). [Details below](#v31--the-lean-no-conv-architecture-teaching-models--true-minimum).
- **Neural Chess web tool (`viz/`).** A browser app that runs a hand-written TypeScript forward pass of the tiny v3.1 hero and lets you **play** it *and* **inspect** every weight/activation from the architecture diagram down to a single scalar. [Details below](#neural-chess-web-tool-viz) · [viz/README.md](viz/README.md).

All three generations share the same `PolicyEngine` runtime interface so a single `play.py` / `uci.py` works for any; the architecture is auto-detected from the checkpoint.

**Optional MCTS at inference.** An AlphaZero-style PUCT search (Monte Carlo Tree Search guided by the network — `src/mcts.py`) wraps any v2 or v3 model, using only the model's own policy priors and value estimate — no hand-coded chess heuristics. It's an opt-in `play.py --mcts` flag, not the default. See [Playing](#playing) and the [MCTS results](#mcts-results). ("PUCT" is the standard exploration formula the tree uses to decide which move to search deeper.)

### Naming convention

Model directories are named `v{N}-{params_M}M[-{tag}]` where `{params_M}` is the param count rounded to the nearest million. The optional `-{tag}` (e.g. `-a`/`-b`) only appears when two runs land on the same rounded million but had different architecture experiments. This keeps the directory name honest — "19M" is a fact, not a claim about whether the run was "final" or "best." v3 follows the same scheme under `model/v3/` (`v3-18M`, `v3-37M`), plus the **`-tau` suffix** for a model trained on the [position-aggregated "tau" data recipe](#the-tau-data-recipe-position-aggregated) (e.g. `v3-18M-tau`). Earlier v2 experiment-flavored names (`T0a`, `T2_FAT`, `T_FINAL`, etc.) were retroactively renamed under this convention; the lineage is preserved in `memory/v2-build-log.md`.

## Project principles

Three load-bearing rules govern every design decision:

1. **Human games only.** Training data must come from human-played games. No engine games, no self-play, no engine-annotated labels, no pretrained engine weights.
2. **Pen-and-paper signal only.** Supervision targets are limited to what's observable in the games (positions, moves played, outcomes). No computed chess features (material counts, pawn-structure heuristics, king-safety scores, mobility, etc.) — the model must learn what matters from raw observation.
3. **Single forward pass by default — with one principled search exception.** The default inference mode is one forward pass producing one move; no hand-coded search heuristics ever. The **one carve-out** is AlphaZero-style PUCT MCTS *whose tree uses only the model's own policy `P` and value `V`* — no material terms, piece-square tables, or pruning heuristics. That injects no chess knowledge the model didn't already have; the tree just lets the model think harder. Alpha-beta with a hand-coded eval, or "run the model N times and average," remain disallowed. See `memory/project-principles.md` for the full carve-out.

These rules keep the experiment honest: how strong a chess engine can we build from human-game observation plus (optionally) the model amplifying its own judgment via search — with zero external chess intelligence injected?

---

## v3 architecture (current)

v3 keeps v2's **input (21 planes, rotation) and output (4,672 move encoding) unchanged** — only the inner tower changes from a CNN to self-attention over the 64 squares. The thesis: a CNN builds long-range board understanding only by stacking many conv layers, whereas self-attention lets any square influence any other in one layer.

**The current, validated form is "v3.1": a pure square-token transformer with _no_ conv stem.** Later ablations showed the conv stem is dead weight — the model is *smaller and stronger* without it. The first v3 big models (`v3-18M`/`v3-37M`) were trained with a small conv stem *before* that finding; the diagram below shows the current no-conv form, and the [v3.1 section](#v31--the-lean-no-conv-architecture-teaching-models--true-minimum) has the evidence.

```
Input (21, 8, 8)
     │
     ▼
Per-square embed: 21 planes → C, 1×1 per square   (v3.1 — no conv stem)
     │
     ▼
Tokenize (B,C,8,8) → (B,64,C)  +  learned positional embedding   (token = square index)
     │
     ▼
Transformer tower: `n_blocks` pre-norm blocks
   each: LayerNorm → GeoSelfAttention (+skip) → LayerNorm → FFN ×4 (+skip)
     │
     ├──────────────────────┐
     ▼                      ▼
Policy head:                Value head:
  Linear C→73 per token       mean-pool 64 tokens → Linear C→128
  → (8,8,73) = 4672 logits    → ReLU → Linear→1 → tanh  ∈ [-1,1]
```

**GeoSelfAttention** adds a *learned 2D relative-position bias* indexed by the (Δrank, Δfile) displacement between squares — this encodes only board **geometry** (the transformer analog of a CNN's translation-equivariance), the one principle-sensitive choice. It provides a uniform learnable slot for all 225 displacements and singles out none; the model must *discover* from games which offsets matter (e.g. knight `(±1,±2)`). It bakes in **no** piece-movement rule (Principle 2). Toggle via `config.geometry_bias`.

| Config | d_model × blocks | Params | Trained on | Notes |
|--------|------------------|-------:|------------|-------|
| `v3-18M` | 256 × 20 | 18.3M | 100M positions | beat v2-19M +150 Elo & 2× v2-37M on SF |
| **`v3-37M`** (best raw) | **384 × 18** | **37.4M** | **100M positions** | acc **62.70%**; strongest single model |
| `v3-18M-tau` | 256 × 20 | 18.3M | 72M-unique aggregated | the [tau recipe](#the-tau-data-recipe-position-aggregated); ≈ v3-37M at half params |

Config-driven via `ChessConfigV3` (`src/v3/model.py`); knobs `--d-model / --n-heads / --n-blocks / --ffn-mult / --stem-blocks / --no-geometry-bias / --checkpoint-every` in `src/v2/train.py` (with `--arch v3`). Training signal, losses, and the rotation trick are identical to v2.

> **The diagram above shows the current no-conv form (v3.1).** The conv stem is the *legacy* path — enabled by `stem_kernel=3, stem_blocks=2` and used only by the first big models (`v3-18M`/`v3-37M`) in the table above, which predate the ablation. Setting `stem_kernel=1, stem_blocks=0` (the default for the teaching/hero models and the next big model) gives the pure square-token transformer, which is *smaller and stronger* — attention + the relative geometry bias already cover locality. Evidence: [v3.1 — the lean (no-conv) architecture](#v31--the-lean-no-conv-architecture-teaching-models--true-minimum).

## v2 architecture (prior generation)

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

Legend: **Conv** = convolution layer; **BN** = batch normalization (stabilizes training by normalizing each layer's outputs); **ReLU** = rectified linear unit (the standard nonlinearity, `max(0, x)`); **ResNet** = residual network (the `+skip` shortcut adds a block's input to its output, which keeps gradients flowing through deep stacks); **1×1** convs mix channels without looking at neighboring squares.

The encoder depth/width and head widths are all config-driven (`ChessConfigV2`), so v2 spans a scale ladder. The diagram above shows the `v2-19M` config (16×256); the **best v2, `v2-37M`, is 20×320 with wider heads** (policy 128, value 8 / hidden 256). Parameter budgets:

| Config | Blocks×Ch | Policy / Value heads | Params | Trained on |
|--------|-----------|----------------------|-------:|------------|
| `v2-19M` | 16×256 | 64 / 4·128 | 19.0M | 40M positions |
| **`v2-37M`** (best) | **20×320** | **128 / 8·256** | **37.1M** | **100M positions** |

In both, ~99% of parameters live in the residual tower; the heads are tiny by comparison. Tunable via `--blocks / --channels / --policy-channels / --value-channels / --value-hidden` in `src/v2/train.py`.

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
| Dataset size | 22M positions, 800k games | up to 100M positions, 3-tier Elo mix (8M/40M/100M shards) |

The most important practical effect is that v2 plays both colors correctly with one model — v1 effectively only learned white because the dataset was filtered to winner's moves and the board rotation didn't expose enough black-as-moving-player signal.

---

## v1 architecture (original)

The v1 model is a 12-plane residual CNN with a single policy head. Documented in detail in earlier commits of this README; the short version:

- **Input** (12, 8, 8): own/opponent piece planes, sign-encoded board rotated for black.
- **Tower**: 10 residual blocks × 256 channels (~11.8M params).
- **Policy head**: two 1×1 convs (256→32→64) reshaped into `(64, 8, 8) = 4096` (from, to) logits.
- **No value head, no castling/EP planes.**
- Trained for ~50 epochs on `data/v1/*.npz` (22M positions); cosine learning-rate (LR) schedule; FP16 (16-bit floating-point) mixed precision.

The trained v1 best checkpoint serves as the head-to-head reference opponent during v2 evaluation.

---

## The "tau" data recipe (position-aggregated)

A validated way to make a model stronger by improving the **data**, not the architecture (it applies to any v2/v3 model). The standard shards label each position with *that one game's* single outcome and the *one* move played; openings are also seen millions of times (frequency-weighted). The tau recipe instead **aggregates the corpus by unique position** and stores richer, lower-noise targets:

- **Averaged value `Q`** — mean game outcome over *all* games reaching the position (a denoised win-probability target), instead of one game's noisy ±1/0.
- **Soft-policy move histogram** — the full distribution of human moves from the position (a richer label than a single move; KL target).
- **count** — how many games reached it, used for **frequency-tempered sampling** `P(sample) ∝ count^τ`. `τ=1` reproduces today's frequency weighting; `τ=0` is uniform over unique positions; **`τ≈0.5`** down-weights over-represented openings without flattening to the noisy long tail.

All three are still *observed* quantities (averages/histograms of real human games) — Principle-1/2 clean, no engine signal. Position identity is the polyglot Zobrist key (placement + side + castling + capturable-EP; transpositions merge, colors don't).

```bash
# 1) Aggregate the filtered PGNs into a position-keyed corpus (memmap artifact)
python -m src.v3.aggregate --out-dir data/v2/agg_100M --instances 100000000 --cap-unique 72000000

# 2) Train any v3 config on it with the recipe knobs
python -m src.v3.train_agg --agg-dir data/v2/agg_100M \
    --save-dir model/v3/v3-18M-tau --save-name v3-18M-tau \
    --value-mode avg --policy-mode soft --tau 0.5 \
    --d-model 256 --n-blocks 20 --epochs 10 --epoch-size 40000000 \
    --num-workers 16 --save-every-steps 5000
```

`aggregate.py` streams `(row, move)` pairs to a flat array and builds the move histogram via a vectorized sort/group (so the full ~72M-unique corpus aggregates in ~10 GB RAM). `train_agg.py` adds a numpy-based frequency sampler (torch's `WeightedRandomSampler` caps at 2²⁴ categories — too few for 70M positions) and per-worker memmap sharing. Naming convention for recipe models: **`v3-<paramsize>-tau`**.

**Result:** `v3-18M-tau` gains **~+79 Elo** over a same-size model on the old data, the advantage *grows* with data volume, and it **≈ matches the 2×-larger `v3-37M`** at low temperature / under MCTS / on the Stockfish ladder — at half the parameters. Full write-up in `memory/position-aggregated-dataset.md`.

---

## v3.1 — the lean (no-conv) architecture, teaching models & true-minimum

A line of tiny "teaching" models (small enough that **every weight/activation is browsable** — see the [web tool](#neural-chess-web-tool-viz)) drove a real architecture finding that feeds back into the big models. Full campaign report: `eval/v3/teaching_models_report.md`; durable findings in `memory/teaching-models-tau.md` and `memory/v3.2-ablation-verdict.md`.

**v3.1 = drop the conv stem.** Replacing the conv stem with a 1×1 per-square embed (`stem_kernel=1, stem_blocks=0`) gives a **pure square-token transformer**. At small scale it's *smaller and stronger*: the 116k `v3.1-eq` (d32/h4/b8, no conv) beats Stockfish-easiest **77%**, vs 58% for the 159k conv-stem `v3-nano-tau` (+19pp at −42k params); at equal params, spending the conv's budget on more attention blocks also wins. Attention + the relative geometry bias already cover locality — the conv stem is dead weight.

**v3.2 ablation = v3.1 is tight.** A full hunt for further dead weight found none: removing the geometry bias is catastrophic (sf_easy 77→5%), removing the positional embedding costs −17pp, halving heads −19pp, and cross-block weight-sharing is fatal — every component earns its keep. Trading FFN width for depth is a wash (a *style* knob — deeper plays better vs stronger opponents at the same prediction accuracy — not a strength win). **Conclusion: v3.1 is the architecture; there is no "v3.2."** Full table: `memory/v3.2-ablation-verdict.md`.

**True minimum.** Scaling v3.1 down, the smallest net that still beats Stockfish-easiest is **d24/b8 ≈ 70.6k params** (~52% wins, confirmed over 520 games). Below it the strength cliff is steep and **width-bound** (d20→37%, d16→19%, d12→11%); depth can't rescue narrow width. Checkpoint: `model/v3/truemin/T1-d24b8/`.

### Fast in-RAM trainer (tau recipe, bit-packed)

`src/v3/pack_agg.py` + `src/v3/train_agg_fast.py` are a **~13–25× faster** path for the tau recipe. 20 of the 21 input planes are binary after int8 truncation, so the ~92 GB `agg_100M` corpus bit-packs to a RAM-resident **~26 GB** form (`data/v2/agg_100M_packed/`, round-trip bit-exact) that loads fully into memory, and the GPU bit-unpacks each batch — eliminating the disk-I/O bottleneck that capped `train_agg.py` at ~2k samp/s. Exposes the v3.1/v3.2 knobs `--stem-kernel --stem-blocks --ffn-mult --value-hidden --no-geometry-bias --no-pos-emb --share-blocks`. Used for every teaching model; the recommended trainer for the next big model.

## Neural Chess web tool (`viz/`)

A public, static, **educational** browser app (React + Vite + TypeScript) that runs a **hand-written TypeScript forward pass** (no ONNX; PyTorch-parity-verified, identical argmax) of the 116k v3.1 hero and lets you:
- **play** it — one-shot inference per move, with a value-head readout and an optional top-5 move picker, and
- **inspect** it — a "Model Inspector" telescope from the architecture diagram down to individual attention weights tied to the 64 board squares, with contextual explanations.

It is **architecture-version-neutral**: it never reads a `.pt`, only a self-describing **Model Capsule** (`viz/scripts/export/export_model.py` → `capsule.json` graph + `weights.bin` + `config.json`), so it renders whatever architecture the capsule declares. See `viz/README.md` and `viz/IMPLEMENTATION_PLAN.md`.

## Roadmap — next big model

The next big-model run stacks the three validated levers: **`v3.1-37M-tau`** = the v3.1 (no-conv) architecture + the tau data recipe + ~37M capacity (matching the current best raw model `v3-37M`, so the comparison isolates the gain). Train with `train_agg_fast.py` in BF16, ~10–16 epochs, `--save-every-steps`; eval on the full SF ladder + h2h vs v3-37M; projected ≈ +79 Elo from tau alone. **Caveat:** no-conv is validated only at ~116k, so an 18M conv-vs-no-conv A/B is cheap insurance before committing the full run. Also pending: ship the already-trained `v3-18M-tau` as the efficient model. Full plan + risks (incl. the degraded training box, RMA pending): `memory/future-architecture-roadmap.md`.

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

Pre-parsed NPZ files (NumPy's compressed `.npz` array format) in `data/v1/*.npz`, one per year/era. Each file is a `np.savez_compressed` archive containing:

| Array | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `X` | `(N, 384)` | int8 | Board features — 6 sign-encoded piece planes of 8×8, flattened |
| `Y` | `(N, 4096)` | int8 | One-hot move labels over (from, to) pairs |
| `meta` | `(1,)` | int32 | Row count |

Reproducing v1 data from raw PGN (Portable Game Notation, the standard chess game-record text format): drop `.PGN` files into `data/v1/` and run:

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
| 8M shard (smaller smoke-test) | ~11 GB |
| 40M-position training shard (binary memmap) | ~51 GB |
| 100M shard (trained v2-37M / v3-37M) | ~126 GB |
| Position-aggregated "tau" corpus (`agg_100M`, 72M unique) | ~92 GB |

Reserve **~20 GB** for an 8M smoke-test reconstruction (~60 GB through the 40M shard); the full 100M shard and the tau corpus need **~220 GB** between them.

#### Step 1 — Download raw sources

```bash
python data/v2/_acquire/download.py
```

Downloads (per [data/v2/README.md](data/v2/README.md)):
- **Lichess Elite** (nikonoel) — all monthly archives. CC0 (Creative Commons public-domain dedication).
- **Lichess monthly** (`database.lichess.org`) — three most recent months of standard rapid+. CC0.
- **TWIC** (The Week in Chess, theweekinchess.com) — issues 1500 through latest. "Personal use" license — do not redistribute the raw PGNs or the filtered output.

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

# 40M shard (trained v2-19M; ~25 min to generate)
python -m src.v2.dataset --out-dir data/v2/training_T1_rot_40M --positions 40000000

# 100M shard (trained the current-best v2-37M; ~95 min, ~126 GB on disk)
python -m src.v2.dataset --out-dir data/v2/training_T1_rot_100M --positions 100000000
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

`src/v2/train.py` trains **both** v2 (CNN) and v3 (attention) — pass `--arch v3` for the attention tower. The tau-recipe trainer is the separate `src/v3/train_agg.py` (see [the tau recipe](#the-tau-data-recipe-position-aggregated)).

### v3 (current)

The command that trained the strongest model, `v3-37M` (attention, 384×18, 100M shard):

```bash
python -m src.v2.train --arch v3 \
    --shard-dir data/v2/training_T1_rot_100M \
    --save-dir model/v3/v3-37M --save-name model \
    --d-model 384 --n-heads 8 --n-blocks 18 \
    --epochs 10 --max-epochs 10 --batch-size 1024 --lr 6e-4 \
    --checkpoint-every 1 --num-workers 8 --save-every-steps 3500
```

Notes specific to v3: `--checkpoint-every 1` enables gradient checkpointing (keeps the 37M tower at ~5.3 GB VRAM); `--d-model / --n-heads / --n-blocks / --ffn-mult / --stem-blocks / --no-geometry-bias` configure the tower. Everything else (BF16, cosine LR, auto-resume, `.stop`, losses) is shared with v2 below. v3 checkpoints carry `arch='v3'` and load via the same `play.py` / eval path. **On the current (degraded) training box, disable CPU Turbo Boost before long runs and use `--save-every-steps`** — see `memory/training-hardware-stability.md` / `CrashAnalysis.md`.

### v2 (prior generation)

The exact command that trained `v2-37M`:

```bash
python -m src.v2.train \
    --shard-dir data/v2/training_T1_rot_100M \
    --save-dir model/v2/v2-37M \
    --epochs 8 --max-epochs 8 \
    --batch-size 2048 \
    --blocks 20 --channels 320 \
    --policy-channels 128 --value-channels 8 --value-hidden 256 \
    --num-workers 8 --keep-last-n 8
```

Notes:
- `--epochs` is the **cosine LR horizon** (anneals LR to 0 at that epoch). Training continues until `--max-epochs` (or a `.stop` file appears in the project root). Set them equal to run the full cosine.
- Auto-resumes from the latest checkpoint in `--save-dir`. Pass `--no-resume` to start fresh.
- **Mixed precision is BF16** (bfloat16 — a 16-bit float with the same exponent range as 32-bit, so it can't overflow; auto-detected, falls back to FP16 only if unsupported). BF16 is required for deep towers — FP16 overflows and silently NaNs BatchNorm buffers on ≥16-block models. Gradient clipping (caps the gradient size each step at norm 1.0) is always on.
- **`--num-workers 8`** is strongly recommended for large shards that exceed RAM page-cache (the 100M shard is 126 GB on disk) — it gives ~7× faster data loading. Works on Windows thanks to the memmap pickle fix in `ChessDatasetV2`.
- ~8 h/epoch on a 4080 Super for the 100M / 20×320 config.
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
| `--value-hidden` | 64 | Value head MLP (multi-layer perceptron) hidden width |
| `--num-workers` | 0 | DataLoader workers (use 8 for large memmap shards; ~7× faster loading) |
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
python play.py model/v2/v2-37M/model_e0007.pt engine --games 10 --depth 5 --skill 10
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

### With MCTS

Add `--mcts` to wrap a model in PUCT search (any v2 or v3 model — needs a value head):

```bash
python play.py model/v2/v2-37M/model_e0007.pt engine --games 10 --depth 16 --skill 20 \
    --mcts --mcts-sims 50 --mcts-cpuct 1.5
```

| Flag | Default | Description |
|------|---------|-------------|
| `--mcts` | off | Use PUCT MCTS instead of raw single-shot policy |
| `--mcts-sims` | 70 | Simulations per move (50 ≈ best in-budget; see [MCTS results](#mcts-results)) |
| `--mcts-cpuct` | 1.5 | PUCT exploration constant (1.5 is the swept optimum) |

### Interactive

Play in the terminal (you're white, model is black):

```bash
python play.py model/v2/v2-37M/model_e0007.pt interactive          # raw
python play.py model/v2/v2-37M/model_e0007.pt interactive --mcts   # with search
```

UCI (Universal Chess Interface) move notation (`e2e4`, `g1f3`, `e7e8q` for promotions). Type `quit` to exit.

### Session-based play (for AI agents)

`src/play_session.py` is a stateful one-move-at-a-time harness suitable for an external player (a human at the CLI, or an LLM agent) submitting moves between separate process invocations:

```bash
# Start a game; agent plays white, neural-chess plays black
python -m src.play_session start \
    --white "agent:my-label" \
    --black "neural-chess:model/v2/v2-37M/model_e0007.pt" \
    --game-id mygame

# Submit moves one at a time
python -m src.play_session move --game-id mygame --uci e2e4

# Show current state
python -m src.play_session show --game-id mygame
```

Player specs: `interactive` / `agent[:LABEL]` / `neural-chess:<path>` / `stockfish[:depth[:skill[:path]]]`. State persists in `tmp/play_sessions/<game-id>.json`.

---

## Evaluation

### SF ladder + head-to-head (works for v2 **and** v3)

`eval/v2/eval_v2.py` is architecture-agnostic (it drives `play.py`, which auto-detects the checkpoint arch), so it evaluates v3 models too — just point `--save-dir`/`--save-name` at a v3 checkpoint:

```bash
# v3 example
python eval/v2/eval_v2.py 9 \
    --save-dir model/v3/v3-37M --save-name model \
    --eval-csv eval/v3/eval_v3-37M.csv --no-gate

# v2 example
python eval/v2/eval_v2.py <epoch> \
    --save-dir model/v2/v2-37M --save-name model \
    --eval-csv eval/v2/eval_v2-37M.csv \
    --no-gate
```

`--sf-games N` / `--magnus-games N` raise the per-tier game counts for tighter signal (defaults 100 / 10).

Runs the Stockfish ladder, a uniform-random-mover baseline, and a head-to-head against the v1 best (`model/v1/checkpoints/model_e0009.pt` by default):

| Opponent | Depth | Skill | Games | Purpose |
|----------|------:|------:|------:|---------|
| sf_easy | 1 | 0 | 100 | sanity floor |
| sf_med | 3 | 5 | 100 | club level |
| sf_hard | 5 | 10 | 100 | the discriminating tier |
| sf_magnus | 8 | 20 | 10 | strong |
| sf_ultra | 16 | 20 | 10 | ~full-strength SF |
| random | — | — | 100 | conversion/finishing test (want 100% + low plies/game) |

By default, higher SF tiers are gated by ≤10% loss at the previous tier; `--no-gate` runs all. `--skip-h2h` / `--skip-sf` / `--skip-random` limit which sections run. Results append to a CSV with schema `epoch,timestamp,opponent,games,won_pct,draw_pct,lost_pct,white_won_pct,black_won_pct,illegal_pct,moves,elapsed_s`.

Other eval/analysis tools in `eval/v2/`:
- **`mcts_sweep.py`** — sweep MCTS settings (sims, c_puct) vs the SF ladder + raw self
- **`vs_random.py`** — multi-model random-mover gauntlet
- **`analyze_v2.py`** — parameter-utilization analysis (BN gammas, per-block contribution)
- **`startpos_value.py`** — what value each model assigns the start position vs the empirical corpus white-advantage
- **`mcts_vs_ultra_timed.py`** — timed match harness (sums each side's thinking time)

v3 / tau-recipe tools in `eval/v3/`:
- **`bakeoff_h2h.py`** — head-to-head between two checkpoints (used for the v3-vs-v2 bake-offs)
- **`agg_sweep.py` / `agg_report.py`** — sweep + rank tau data treatments (value/policy/τ) at matched compute
- **`agg_h2h.py` / `agg_h2h_mcts.py`** — generic policy / MCTS head-to-head between any two checkpoints
- **`bench_v3_18M_tau.py`** — comprehensive benchmark (SF ladder + large-N h2h with Elo±CI + MCTS)

### v1 — multi-checkpoint vs SF

```bash
python eval/v1/evaluate.py            # all checkpoints, default settings
python eval/v1/evaluate.py --quick    # 1 checkpoint, 5 games per setting
python eval/v1/evaluate.py --models 5 # evenly spaced 5 checkpoints
```

Other v1 analysis scripts (opening diversity, castling/EP rates, temperature sweep) live in `eval/v1/`.

---

## Results

### Current best: `v3-37M` (attention) and `v3-18M-tau` (efficiency)

**`v3-37M`** is the strongest single model — top-1 move accuracy **62.70%**. Raw single forward pass vs Stockfish: sf_easy 94%, sf_med **96%**, sf_hard **81%** wins (vs v2-37M's 65%); with **MCTS-50 it reaches 0.60 vs sf_magnus (d8)**, the best Magnus-tier result in the project. Head-to-head it beats the same-size v2-37M CNN **0.645 (≈ +100 Elo)** — and at 18M, `v3-18M` beat v2-19M by **≈ +150 Elo**, confirming attention is more parameter-efficient than the CNN.

**`v3-18M-tau`** (the [tau data recipe](#the-tau-data-recipe-position-aggregated) at 18M) shows data ≈ capacity: **+79 Elo** over a same-size old-data model, and it **matches `v3-37M`** at low temperature (h2h ≈ even at temp 0.3) and **under MCTS-50 (40W/0D/40L, dead even)** — at half the parameters. The 2× model only pulls ahead under higher-temperature sampling. SF ladder: med 0.96, hard 0.84.

> Nuance: v3's edge over v2 is a **policy** win — at matched data the value head is a near-tie with the CNN (signal-limited). The tau recipe's averaged value is what improves the value head (and thus MCTS).

### `v2-37M` (CNN, prior best — raw, single forward pass)

| Opponent | W / D / L | Notes |
|----------|-----------|-------|
| sf_easy (d1/s0) | 92 / 5 / 3 | |
| sf_med (d3/s5) | 88 / 10 / 2 | |
| sf_hard (d5/s10) | 65 / 21 / 14 | discriminating tier |
| sf_magnus (d8/s20) | occasional wins/draws | ~best raw result is 10/10/80 |
| sf_ultra (d16/s20) | occasional draw | ~full-strength SF; mostly losses |
| random | ~99 / 1 / 0 | near-perfect conversion |
| **v1 best (200 games)** | **200 / 0 / 0** | total domination |

Top-1 move-prediction accuracy 60.9%. `v2-37M` clearly surpasses the earlier `v2-19M` at every discriminating tier (e.g. sf_hard 52→65% wins).

### MCTS results

PUCT search on top of a v2 model, swept for the strongest settings (full sweep in `memory/v2-build-log.md`):

- **Best in-budget config: 50 sims @ c_puct 1.5** (≈170 ms/move on `v2-37M`, within Stockfish-ultra's per-move time). c_puct is an inverted-U with a clean peak at 1.5; the search benefit floors in around 15 sims and saturates by ~50.
- **On `v2-37M`, MCTS-50 sweeps sf_easy/med/hard at 100%** and cracks sf_magnus (d8) to ~37% wins **within budget** — where `v2-19M` at 50 sims scored 0% and needed ~800 over-budget sims to compete.
- **Search effectiveness scales with base-model strength:** the stronger `v2-37M` extracts far more per simulation than `v2-19M`, and is the first model to draw against sf_ultra (d16) from the search track.
- **The d16 "ultra" tier remains a model-quality ceiling** — search amplifies the model but cannot bridge a ~1000-Elo gap. Beating it needs a stronger base model (the case for v3), not just more simulations.

Run with `play.py --mcts` or sweep with `eval/v2/mcts_sweep.py`. MCTS uses only the model's own policy `P` and value `V` (Principle 3 carve-out — no injected chess heuristics).

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
│       ├── training_T1_rot*/        # featurized shards (8M / 40M / 100M) for v2/v3
│       ├── agg_100M/        # position-aggregated "tau" corpus (avg value + move histogram + count)
│       └── agg_100M_packed/ # bit-packed RAM-resident "tau" corpus (~25 GB; ~26 GB resident) for train_agg_fast.py
├── model/                   # gitignored — checkpoints per version
│   ├── v1/checkpoints/
│   ├── v2/v2-{2..37}M*/     # the v2 CNN scale ladder; v2-37M (20×320, 100M data) is best v2
│   └── v3/                  # attention tower: v3-18M, v3-37M (best raw), v3-18M-tau (tau recipe)
│       ├── v3.1/            # pure-transformer (no conv) ablation: v3.1-eq (116k hero), v3.1-pm
│       ├── v3.2/            # v3.2 ablation runs (R1..R8 — all confirmed v3.1 is tight)
│       ├── truemin/         # true-minimum scale pass (smallest v3.1 beating sf_easy = d24/b8 ~70.6k)
│       ├── v3-micro-tau/, v3-nano-tau/   # teaching models (conv-stem era)
│       └── v3.1-nano/       # exported-hero checkpoint (= v3.1-eq) backing the web-tool capsule
├── src/
│   ├── inference_api.py     # PolicyEngine abstract base + load_policy_engine factory (arch auto-detect)
│   ├── engine.py            # Stockfish/UCI helpers
│   ├── game_loop.py         # engine-vs-engine game runner
│   ├── mcts.py              # AlphaZero-style PUCT MCTS + MCTSEngine wrapper
│   ├── random_engine.py     # uniform-random-mover baseline opponent
│   ├── stats.py             # play stats aggregator
│   ├── uci_protocol.py      # UCI protocol implementation for uci.py
│   ├── play_session.py      # stateful session-based play harness
│   ├── v1/
│   │   ├── model.py / featurize.py / dataset.py / train.py / inference.py / parse_pgn.py
│   ├── v2/
│   │   ├── model.py / featurize.py / moves.py / dataset.py / train.py / inference.py
│   │   ├── lookahead.py     # disabled experimental lookahead block (ablated; still needed to load v2-3M)
│   │   └── test_dataset_hardening.py
│   └── v3/
│       ├── model.py / inference.py     # attention tower (+ v3.1/v3.2 toggles) + V3PolicyEngine
│       ├── aggregate.py     # build the position-aggregated "tau" corpus
│       ├── train_agg.py     # train on the tau corpus (avg value + soft policy + count^τ)
│       ├── pack_agg.py      # bit-pack agg_100M → agg_100M_packed (~92 GB → ~26 GB, RAM-resident)
│       └── train_agg_fast.py  # fast in-RAM bit-packed tau trainer (~13–25×; v3.1/v3.2 flags)
├── eval/
│   ├── v1/                  # evaluate.py, eval_one.py, opening/castling/temp analyses
│   ├── v2/                  # eval_v2.py, mcts_sweep.py, vs_random.py, analyze_v2.py, startpos_value.py, mcts_vs_ultra_timed.py
│   └── v3/                  # bakeoff_h2h.py, agg_sweep/report/h2h/h2h_mcts.py, bench_v3_18M_tau.py
│                            #   teaching/ablation orchestrators: run_micro_campaign, run_nano_search,
│                            #   run_v3_1_ablation, run_v3_2_explore, run_v3_2_batch2, run_v3_truemin
│                            #   + teaching_models_report.md
├── viz/                     # Neural Chess web tool (React + Vite + TS) — see viz/README.md
│   ├── src/core/            # presentation-agnostic: engine + trace + model-graph + content + game
│   ├── src/presentations/desktop/   # React desktop UI (play + Model Inspector telescope)
│   ├── public/weights/v3.1-nano/    # the hero Model Capsule (capsule.json + weights.bin + config.json)
│   └── scripts/export/export_model.py   # checkpoint → versioned Model Capsule
├── logs/                    # gitignored — training/eval logs
└── tmp/                     # gitignored — play_sessions/, scratch
```

---

## Provenance and license

- **Code**: see repo license.
- **v1 training data**: derived from open historical PGNs (TWIC + Lichess) — derived NPZ shards may be redistributed.
- **v2 training data**: mixed sources. Lichess content is CC0; **TWIC content is "personal use only" and the filtered tier PGNs / shards derived from it must NOT be redistributed wholesale.** Trained weights derived from this mix are personal-use only.
- **No engine-derived training signal** (Stockfish evaluations, MCTS rollouts, etc.) is used anywhere in either version, per the project principles.
