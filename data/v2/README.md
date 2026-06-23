# Data v2 — PGN corpus + filtered intermediate dataset

This document is the authoritative spec for the next-generation training
dataset. It supersedes the original 31 NPZ files in `data/` (which are
preserved alongside as the legacy v1 dataset).

> **Downstream artifacts (built from `filtered/`):** two featurizations consume
> this corpus. (1) The standard **per-position shards** (`training_T1_rot*`,
> `src/v2/dataset.py`) — one row per ply with that game's single move + outcome;
> used by every v2/v3 model. (2) The **position-aggregated "tau" corpus**
> (`agg_100M`, `src/v3/aggregate.py`) — rows keyed by unique position with an
> *averaged* value, a *move histogram*, and a *count* for frequency-tempered
> sampling. The tau corpus is a validated strength win (`v3-18M-tau`); see the
> repo README "tau data recipe" section and `memory/position-aggregated-dataset.md`.
> A bit-packed, RAM-resident variant (`agg_100M_packed`, built by `src/v3/pack_agg.py`)
> backs the fast trainer `src/v3/train_agg_fast.py` (~13–25× speedup; round-trip bit-exact).

## Goals

1. **History-aware training inputs.** The current v1 dataset bakes positions
   to (6, 8, 8) sign-encoded planes and discards everything else — castling
   rights, en passant target, halfmove clock, repetition state. The v2
   pipeline stops at PGN/filtered-PGN so any future featurization can use
   whatever the chess rules need.

2. **Quality-graduated mix.** Train not only on masters (which lack the
   "exploit a blunder" signal) but on a deliberate blend across rating tiers.
   Strong amateur and intermediate games provide examples of blunders being
   punished — the missing class of pattern in pure-master training data.

3. **License-clean throughout.** Every game must come from a source whose
   license permits training and (ideally) redistribution of derived artifacts.
   No Stockfish-derived signal of any kind appears in the dataset; the goal
   is to test whether our model can rival Stockfish at hard difficulty
   *without* having been taught by it.

4. **Sized for a scale ladder, not a single target.** The filtered corpus
   should be deep enough to support training shards at multiple sizes
   (~5M, ~15M, ~40M, ~100M positions), enabling small smoke-test models
   to iterate quickly before scaling up. The 100M target is the upper
   ladder rung; the filter step keeps everything that passes quality
   gates so we can generate shards at any size from one corpus. Storage
   uses memory-mapped binary at training time so RAM is not a constraint.

5. **Re-runnable.** The PGN → filtered intermediate is the canonical form.
   Featurization (the future step 3) reads from `filtered/` and writes new
   training shards; changing the input encoding doesn't require re-downloading
   or re-parsing the source PGNs.

## Non-goals

- **No Stockfish or any other engine evaluation** is incorporated into the
  dataset. The target labels are: (a) the move the human/player chose, and
  (b) the game's final result. Nothing more.
- **No engine annotations preserved.** PGN comments, NAGs (`!`, `??`, `$1`),
  and `[%eval ...]` annotations are stripped during the filter step. We don't
  want any signal of "Stockfish thought this was a good move" leaking in.
- **No move-time / clock annotations** kept. Interesting signal but adds
  parsing complexity and the model has no use for it currently.
- **No variation trees** — only the main line of each game is processed.

## Sources

| Source | URL | License | Role | Estimated raw size |
|--------|-----|---------|------|-------------------:|
| **Lichess Elite Database** | <https://database.nikonoel.fr> | CC0 (inherits from Lichess) | Top tier (2400+). Pre-filtered, curated. | ~3 GB compressed, ~10 GB raw |
| **Lichess Standard monthly** | <https://database.lichess.org/standard/> | CC0 | Mid + low tiers (filter at parse) | 3 months × ~800 MB = ~2.4 GB compressed |

**Skipped sources** (with reasons documented for future reference):
- **Ajedrez Data** — license not explicitly stated, risky for derived work
- **Caissabase original** — domain dead (squatted by an unrelated site)
- **Lumbra's Gigabase** — Mega.nz gating defeats automation without megacmd
- **PGN Mentor** — per-player only, no year bundles; heavy overlap with above
- **Computer chess (TCEC, etc.)** — engine-derived, violates non-goal #1

**License notes for our use:**
- Both sources are CC0 / public domain. We can do anything with the raw PGN,
  the filtered/ corpus, and the trained weights — with no attribution required
  and no redistribution restrictions (though we'll attribute Lichess anyway).
  The entire corpus is CC0; there are no "personal use" caveats anywhere in it.

## Tier mix target (for training-time sample selection, not filter-time pruning)

**Important shift in design**: tier mix is a *training-shard-generation*
concern (step 3), not a filter-time concern (step 2). The filter step should
keep **every game that passes the Elo/quality filters**, regardless of how
much that exceeds any target — this gives us a deep buffer to sample from
at training time, and lets us produce shards at different sizes for
different model scales without redoing step 2.

The reasoning: the model architecture is going to be developed via a scale
ladder (small smoke-test models first, scaling up only after the approach
is validated — see [[future-architecture-roadmap]]). Different ladder rungs
want different training-shard sizes (5M, 15M, 40M, 100M positions). Pruning
at filter time would lock us into one size.

The target tier proportions that step 3 will use when generating shards:

| Tier | Elo band | Source | Proportion | Purpose |
|------|----------|--------|-----------:|---------|
| Top | 2400+ | Lichess Elite | ~65% | Strong opening, strategic, endgame |
| Mid | 1900–2400 | Lichess monthly | ~25% | Blunders + exploitation (both sides still competent) |
| Low | 1600–1900 | Lichess monthly | ~10% | More dramatic blunders, clearer punishments |

A 100M-position shard at these proportions implies roughly 1.4M games at
~70 plies/game. A 5M-position shard implies ~70K games. Same filtered/ corpus,
different sampled subsets.

**Filter-time policy:** keep all games that pass the quality criteria below.
Do not enforce a max per tier. If the top tier yields 5M games and we
expected 910K, that's fine — store them all, the buffer is useful. The
manifest should record the actual game count per tier after filtering, not
a downsampled count.

## Filter criteria (applied during step 2)

Applied to every candidate game from every source:

| Filter | Criterion | Rationale |
|--------|-----------|-----------|
| **Both Elos present** | `WhiteElo` and `BlackElo` exist and parse as integer | We need rating for tier assignment |
| **Elo agreement** | Both Elos within ±400 of each other | Mismatched-strength games are dominated by the gap, not by the players' chess |
| **Tier assignment** | min(WhiteElo, BlackElo) → tier band | Use the weaker player's strength; prevents 2500 vs 1500 games from sneaking into "top" |
| **Time control** | Initial time ≥ 180 seconds (skip bullet) | Bullet is dominated by speed, not chess quality |
| **Termination** | `Normal` only (or absent — treated as normal) | Drop time forfeits, abandonments, rule violations |
| **Result** | `1-0`, `0-1`, or `1/2-1/2` — *not* `*` | Need a definite outcome for value-head label |
| **Length** | Game has ≥ 10 plies | Skip noise-resigned-immediately games |
| **Variant** | Standard chess only (skip Crazyhouse, Atomic, etc. if PGN tags say so) | We don't model variants |
| **Annotations** | Strip all `{...}`, `%eval`, `%clk`, NAGs (`!`, `??`, `$N`) before saving | No engine signal allowed |

## Dedup strategy

Same game can appear in multiple sources (Lichess Elite is a subset of
Lichess monthly, etc.). Dedup priority, highest-to-lowest:

1. Lichess Elite (highest curation quality, oldest source generally)
2. Lichess monthly (we use this for mid/low tiers anyway)

**Dedup key**: `sha1(white_name + black_name + date + result + first_20_plies_uci)`.

Implementation: build an in-memory hash set as games are processed in
priority order; skip any game whose key has already been seen.

## Per-position selection (foreshadowing step 3)

Step 3 is not in scope for this download/filter run, but documenting now so
the filter step preserves what's needed:

- **All moves from both sides** are included as training samples (not just
  the winner's, as in v1). This roughly doubles the per-game position yield.
- **Value-head label**: each position is labeled with the player-to-move's
  eventual game outcome (+1 win, 0 draw, −1 loss). Doesn't require Stockfish.
- **Position duplication is preserved**: the same opening position appearing
  in many games is signal, not noise (positions that win more often are, on
  average, better; the data reflects this).
- **Skip the first ~2 plies** of each game during featurization (they
  contribute no learning signal beyond "1.e4 is popular"). Decided at
  featurization time, not here.

## On-disk layout

```
data/
├── (existing 31 *.npz files)            ← legacy v1, preserved as-is
└── v2/
    ├── README.md                        ← this file
    │
    ├── raw/                             ← step 1 output; can be deleted after step 2 verified
    │   ├── lichess_elite/
    │   │   ├── lichess_elite_2024-01.pgn.zst
    │   │   ├── lichess_elite_2024-02.pgn.zst
    │   │   └── ...                      (all available months from nikonoel; ~3 GB total)
    │   └── lichess_standard/
    │       ├── lichess_db_standard_2025-12.pgn.zst   (3 recent months)
    │       ├── lichess_db_standard_2026-01.pgn.zst
    │       └── lichess_db_standard_2026-02.pgn.zst
    │
    ├── filtered/                        ← step 2 output; the canonical intermediate
    │   ├── tier_top_2400plus.pgn        ← ~910K games (Lichess Elite, deduped, filtered)
    │   ├── tier_mid_1900-2400.pgn       ← ~350K games (Lichess monthly, filtered)
    │   └── tier_low_1600-1900.pgn       ← ~140K games (Lichess monthly, filtered)
    │
    ├── games_index.parquet              ← one row per surviving game (see schema below)
    │
    └── manifest.json                    ← see schema below
```

### `games_index.parquet` schema

One row per game in `filtered/`. ~1.4M rows, expected size ~50 MB.

| Column | Type | Notes |
|--------|------|-------|
| `game_id` | str (40) | sha1 hex of dedup key |
| `tier` | str | `"top"` / `"mid"` / `"low"` |
| `source` | str | `"lichess_elite"` / `"lichess_monthly"` |
| `file_path` | str | which file in `filtered/` |
| `pgn_offset_bytes` | int64 | byte offset of game in `file_path` |
| `pgn_length_bytes` | int32 | byte length of game in `file_path` |
| `date` | date (ISO) | from `Date` tag |
| `white` | str | from `White` tag |
| `black` | str | from `Black` tag |
| `white_elo` | int16 | parsed |
| `black_elo` | int16 | parsed |
| `result` | str | `"1-0"` / `"0-1"` / `"1/2-1/2"` |
| `eco` | str (4) | `ECO` tag if present, else `""` |
| `time_control` | str | raw `TimeControl` tag value |
| `n_plies` | int16 | count of plies in main line |

### `manifest.json` schema

```jsonc
{
  "spec_version": "v2.0",
  "generated_at_utc": "2026-05-17T...",
  "git_commit": "<sha>",                  // commit at time of generation
  "sources": [
    {
      "name": "lichess_elite",
      "url": "https://database.nikonoel.fr",
      "license": "CC0",
      "raw_files": [...],                 // names + sha256 of downloaded files
      "raw_games_total": 6000000,         // approximate
      "raw_size_bytes": 3200000000
    },
    // ... lichess_monthly ...
  ],
  "filter_criteria": {                    // mirror of the filter section of this README
    "min_initial_time_seconds": 180,
    "elo_diff_max": 400,
    "min_plies": 10,
    "allowed_terminations": ["Normal"],
    "tier_bands": {
      "top":  [2400, null],
      "mid":  [1900, 2400],
      "low":  [1600, 1900]
    }
  },
  "tiers": {
    "top":  { "games": N, "size_bytes": N },
    "mid":  { "games": N, "size_bytes": N },
    "low":  { "games": N, "size_bytes": N }
  },
  "totals": {
    "games": N,
    "size_bytes": N,
    "approx_positions_if_all_moves": N
  },
  "dedup_stats": {
    "raw_games_seen": N,
    "duplicates_removed": N,
    "kept": N
  }
}
```

## Steps

### Step 1 — Download raw sources

In scope for this run.

1. **Lichess Elite**: download all monthly archives currently available from
   `database.nikonoel.fr`. The site lists them at the root URL; mirror what's
   there. Use HTTPS directly (no proxies / scraping).
2. **Lichess Standard monthly**: download the 3 most recent months from
   `database.lichess.org/standard/list.txt`. Save as `.pgn.zst`.
3. **Verify** all downloads with sha256 and PGN-parses-cleanly checks.
4. **Record sources, sha256, sizes, game counts** in `manifest.json`.

### Step 2 — Filter, dedup, index

In scope for this run.

1. **Decompress on the fly** when reading; do not bloat disk with extracted
   .pgn copies. Use `zstandard` for `.zst`, `zipfile` for `.zip`.
2. **Apply filters** in the priority order listed in Dedup strategy.
3. **For each surviving game**:
   - Strip annotations (comments, NAGs, eval/clock)
   - Assign tier from `min(white_elo, black_elo)`
   - Append cleaned PGN to the appropriate `filtered/tier_*.pgn`
   - Record metadata row for `games_index.parquet`
4. **Do NOT down-sample at this step.** Keep every game that passes the
   quality filters. The `filtered/` corpus is intentionally a buffer that
   exceeds any single training run's needs; sampling to fit a specific
   training shard size is step 3's responsibility. Record actual per-tier
   game counts in the manifest.
5. **Write `games_index.parquet`** and complete `manifest.json`.
6. **Validation**:
   - Reload each `tier_*.pgn` with python-chess; confirm every game parses.
   - Cross-check counts in `games_index.parquet` match file game counts.
   - Print tier totals, mix percentages, date ranges, Elo distributions.

After step 2 validation, `raw/` can be deleted (~13 GB recovered) if disk
pressure exists. `filtered/` + `games_index.parquet` are sufficient for any
future featurization (step 3).

### Step 3 — Featurize into training shards (parameterized)

**NOT in scope for this run.** Will be planned after the Phase 1
architecture is settled (see [[future-architecture-roadmap]]). When we get
there, this step should be **parameterized by target shard size** rather
than producing one fixed-size output:

```
generate_shards.py --positions 5M --tier-mix top:0.65,mid:0.25,low:0.10 --out training_T0/
generate_shards.py --positions 100M --tier-mix top:0.65,mid:0.25,low:0.10 --out training_T3/
```

Same filtered PGN, different shard sizes. This supports the scale-ladder
training methodology (small smoke-test models on small shards first, scale
up only after the approach is validated). The featurization itself is fast
enough (~1-2 hr single-threaded, parallelizable) that re-running it for a
new size is cheap.

The shard format (planned, subject to architecture-time decisions): 
`training_*/X.bin`, `training_*/Y_policy.bin`, `training_*/Y_value.bin`,
`training_*/Y_aux.bin` (classical-heuristic auxiliary features per
position), `training_*/future_ply_index.bin` (pointers into the shard for
plies N+1..N+3 of the same game, supporting differentiable-lookahead
supervision), `training_*/shard_index.bin` (game-id per sample for
traceability), `training_*/meta.json`. All memmap-friendly.

## Tools

- `python-chess` (already in requirements) for PGN parsing and validation
- `pyarrow` for parquet output (NEW dependency)
- `zstandard` for `.zst` decompression (NEW dependency)
- `requests` or `urllib` for downloads (urllib is stdlib, prefer it)
- `bin/pgn-extract.exe` (already in repo) can be used for fast PGN filtering
  if it turns out to be faster than python-chess; verify before relying on it

## Cost estimates

| Step | Time | Disk |
|------|-----:|-----:|
| Step 1 download | ~45 min total (Lichess Elite ~10m, monthly ~30-60m) | ~20-25 GB during download (Lichess monthly is bigger than initial estimates) |
| Step 2 filter | ~3-5 hours single-threaded (no downsample, full corpus kept) | ~15-20 GB filtered output |
| Step 3 featurize (per shard size) | ~1-2 hr for 5M shard, ~10-20 hr for 100M shard | ~7 GB per 5M shard, ~150 GB for 100M shard |

## Open questions / future revisions

- Whether to also include older Lichess monthly archives (pre-2024) for
  metagame diversity. Current plan: only 3 recent months. Easy to extend.
- Whether to apply opening-balance enforcement (currently no; we let natural
  popularity stand). Could be added at featurization time.
- Whether to include any `Variant` tags beyond standard chess. Currently no.
