# Neural Chess — Tiny Teaching Models: Campaign Report

_2026-06-06. Goal: the smallest faithful square-token transformer that (a) plays real chess (≥ Stockfish‑easiest on a single forward pass) and (b) is small enough that **every weight/activation** can be explored in a browser. All training uses the validated "tau" recipe (averaged value + soft-policy histogram + count^τ=0.5 sampling) on the RAM-resident packed corpus._

## TL;DR

| model | config | params | top‑1 | sf_easy W% | sf_med W% | note |
|---|---|---|---:|---:|---:|---|
| v3‑micro‑tau (M1‑len20) | d64·h8·b10, conv stem | 695k | 0.527 | **90** | 75 | strong teaching model; "just train longer" |
| v3‑nano‑tau | d32·h8·b8, conv stem | 159k | 0.454 | 58 | 40 | minimal *conv-arch* model that clears 50% |
| **v3.1‑eq (HERO)** | **d32·h4·b8, NO conv stem** | **116k** | **0.466** | **77** | **43** | **smaller AND stronger — conv stem dropped** |
| v3.1‑pm | d32·h4·b11, no conv stem | 157k | 0.479 | 74 | 46 | equal-param control |

**Three headline findings:**
1. **The data path, not the GPU, was the bottleneck** — a bit-packed RAM-resident corpus gave a **~13–25× training speedup**; these tiny models are launch/bandwidth-bound (GPU power, not util%, is the honest utilization signal).
2. **Training length dominates the tweak space** — for a deeply-underfit tiny model, more epochs beats every cleverer knob; lowering `value_loss_weight` (a plausible idea) actively *hurt*.
3. **The conv stem is dead weight (worse: net-harmful) at this scale** — removing it (→ "v3.1", a pure transformer) made the model **smaller and ~+16–19pp stronger**. Likely generalizes to the big models.

---

## 1. Fast in-RAM trainer (infrastructure)

**Problem.** Training on `data/v2/agg_100M` (97 GB int8 `X` memmap, 72M unique positions) was **disk-I/O-bound at ~2,000 samp/s** — random ~1.3 KB reads starved a near-idle GPU.

**Fix (`src/v3/pack_agg.py`).** 20 of the 21 input planes are binary `{0,1}` after the int8 truncation (only plane 18 = fullmove/100 ∈ {0,1,2}); bit-pack the binary planes → **97 GB → 16 GB**, which fits in the box's 64 GB RAM. Round-trip verified **bit-exact**. Dense top‑32 soft-policy targets are precomputed (`trunc>32: 0` → lossless). Artifact: `data/v2/agg_100M_packed/`.

**Fix (`src/v3/train_agg_fast.py`).** Loads the packed corpus fully into RAM, gathers batches by vectorized RAM indexing, **bit-unpacks on the GPU**, no DataLoader. **~27k samp/s (micro) / ~50k (nano)** = ~13–25× over the I/O baseline. Now GPU-bound.

**Throughput findings (the "improve GPU utilization" thread).**
- **batch 1024 is optimal**; bigger is *worse* (8192 → ~16k samp/s) — the d-dim matmuls are memory-bandwidth-bound, not launch-amortizable.
- `torch.compile` (inductor) needs **Triton, absent on this Windows box**; the Triton-free `cudagraphs` backend gave **~0%**. So ~27k samp/s is the honest eager-mode ceiling.
- **Power draw is the trustworthy utilization signal**, not util%: micro draws ~204 W, nano ~97 W of the 320 W limit — the GPU literally can't spend more watts on work this small. (nvidia-smi util% *and* Task Manager both report "busy" misleadingly for tiny bursty-kernel models.) Raising the power cap to 320 W changed nothing.

---

## 2. Micro campaign — training-tweak sweep (fixed arch, OFAT)

Architecture fixed at d64·h8·b10 (~695k); only training knobs varied, seed 0, same corpus.

| run | tweak vs baseline | top‑1 | sf_easy | sf_med | sf_hard | h2h avg† |
|---|---|---:|---:|---:|---:|---:|
| M0‑base | 12 ep, vlw 1.0, lr 1e‑3 | 0.5135 | 83 | 63 | 42.5 | 0.464 |
| **M1‑len20** | **20 epochs** | 0.5272 | **90** | **75** | 45 | **0.592** |
| M2‑vlw050 | value_loss_weight 0.5 | 0.5132 | 81.7 | 57.5 | 35.8 | 0.459 |
| M3‑vlw025 | value_loss_weight 0.25 | 0.5137 | 86.7 | 62.5 | 37.5 | 0.468 |
| M4‑combo | 16 ep + vlw 0.35 + wd 0 | 0.5226 | 80.8 | 66.7 | 40.8 | 0.511 |
| M5‑lr15 | lr 1.5e‑3 + warmup 600 | 0.5201 | 83.3 | 69.2 | 42.5 | 0.506 |

†avg head-to-head score across the 5 intra-family pairings (temp 0.3, 100 games each). All six score ~0.13–0.22 vs the 25×-larger `v3‑18M‑tau` yardstick (≈ a 200-Elo gap, as expected).

**Lessons.**
- **Training LENGTH is the dominant lever** (the model is deeply underfit). 12→20 ep: top‑1 +1.4pp, sf_easy 83→90, sf_med 63→75, wins every head-to-head.
- **Lowering `value_loss_weight` did NOT help** (M2/M3 ≈ baseline or worse on SF *and* h2h). The "sharper policy → stronger single-pass" hypothesis is **refuted**; keep vlw = 1.0.
- Higher peak LR (M5) and the combo (M4) are modest second-tier; the combo's gains came from its *length* component, not vlw.

---

## 3. Nano size search — the strength cliff (conv architecture)

How small can the (conv-stem) model get and still beat sf_easy ≥50%? 16-epoch tau recipe, descending size:

| config | params | top‑1 | sf_easy W% | illegal% |
|---|---:|---:|---:|---:|
| d16×4 | 30k | 0.322 | **4** | 3.7 |
| d16×10 | 53k | 0.354 | 12 | 2.5 |
| d24×8 | 92k | 0.401 | 29 | 1.4 |
| d32×8 (= v3‑nano‑tau) | 159k | 0.454 | **58** | 0.5 |

**Strength does NOT degrade gracefully — it falls off a cliff.** Below a ~150k floor the policy can't avoid hanging pieces and even depth‑1 Stockfish punishes every blunder (4% wins at 30k). The binding constraint at the bottom is **width** (`d_model`): d16 stays ≤12% even at 10 blocks (narrow residual-stream bandwidth; depth only half-helps). The conv-architecture answer was **159k** (58%).

---

## 4. v3.1 — the conv-stem ablation (architecture win)

Is the conv stem doing anything attention + the geometry bias don't? Replace it with a 1×1 per-square embed (`stem_kernel=1, stem_blocks=0`) = a **pure square-token transformer**. Same d32 family, recipe, seed.

| model | params | conv stem? | top‑1 | sf_easy W% | sf_med W% |
|---|---:|---|---:|---:|---:|
| v3‑nano‑tau | 159k | yes (2 blocks) | 0.454 | 58 | 40 |
| **v3.1‑eq** | **116k** | **no** | 0.466 | **77** | 43 |
| v3.1‑pm | 157k | no | 0.479 | 74 | 46 |

- **Equal-config:** dropping the stem makes it **42k smaller AND +19pp** (116k @ 77% vs 159k @ 58%).
- **Equal-param (decisive):** at ~158k, spending the conv's params on 3 more attention blocks instead wins **74% vs 58%** (+16pp). The conv stem doesn't pull its weight — attention + the relative geometry bias already cover locality.

**Verdict:** **drop the conv stem.** "v3.1" (pure transformer) is the better tiny architecture and *cleaner to teach*. Consequences: (1) the nano frontier drops well below 116k (true minimal v3.1 likely ~50–80k); (2) per the inductive-bias asymmetry (a locality prior helps small models *most*, yet here it *hurt* the smallest), the big models (micro, and plausibly the 18M/37M line) likely improve by dropping it too — pending a confirmation run.

*Caveats:* one seed per point (gaps are ~4σ, so direction is solid); the ablation removed conv+BatchNorm+residual-blocks together, so "conv vs BN as the culprit" isn't isolated — but the practical verdict (drop the stem) is unambiguous.

---

## 5. Conclusions & what's next

- **Hero teaching model = `v3.1‑eq`** (d32·h4·b8, no conv stem, **116k**, 77% vs sf_easy) — strong, tiny, pure transformer.
- **Strong teaching model = `v3‑micro‑tau` / M1‑len20** (695k, 90% vs sf_easy) for when more strength is wanted.
- **Recipe wisdom:** train long; keep vlw=1.0; tau τ=0.5.

**Resolved (2026-06-07) — see §6:**
1. ✅ **v3.2 architecture exploration** — done. No improvement found; **v3.1 is tight** (every component load-bearing, weight-sharing fatal, depth a style-knob). No "v3.2."
2. ✅ **True-minimum scale pass** — done. Smallest v3.1 beating sf_easy = **d24/b8 ≈ 70.6k** (~52%).
3. ⏸ **Confirm conv removal at micro / 18M scale** — still deferred; folded into the next-big-model plan as an optional 18M conv-vs-no-conv A/B (`memory/future-architecture-roadmap.md`).
4. ✅ **Neural Chess web tool** — M1–M3 built & iterating around the v3.1‑eq hero (`viz/`).

## 6. Follow-up: v3.2 ablation & true-minimum (2026-06-07)

**v3.2 — is there more dead weight beyond the conv stem?** A two-batch ablation on `v3.1‑eq` (d32/h4/b8, no conv; sf_easy 77.0 / sf_med 43.3 / top1 0.466). Each run 16ep tau recipe, SF 120g.

| run | change | params | sf_easy | sf_med | top‑1 | read |
|---|---|---:|---:|---:|---:|---|
| R1 | FFN×4→×2 | 83k | 65.8 | 35.0 | 0.449 | −11pp, not free |
| R2 | FFN×2 + b8→11 | 111k | 72.5 | 46.7 | 0.468 | ≈ wash |
| R3 | geometry bias OFF | 109k | **5.0** | 0.8 | 0.345 | catastrophic — essential |
| R4 | pos_emb OFF | 114k | 60.0 | 30.8 | 0.453 | −17pp — essential |
| R8 | heads 4→2 | 113k | 57.5 | 29.2 | 0.441 | −19pp — heads matter |
| R7 | width↔depth d48/b4 | 128k | 65.8 | 40.0 | 0.448 | depth > width |
| R6a/R6b | weight-sharing ×8/×16 | 21k | 0.8 / 7.5 | 0.0 | 0.32 | fatal — blocks must be distinct |
| R5 | LEAN ffn1/b15 (max depth) | 118k | 66.7 | 50.0 | 0.467 | top‑1 flat — depth is a style knob |

**Verdict: v3.1 is the architecture — no "v3.2."** Every component is load-bearing; the only non-negative lever (depth↔FFN) is a *style* knob (top‑1 flat 0.466→0.468→0.467 across b8/b11/b15; deeper trades sf_easy for sf_med), not a strength win. Full writeup: `memory/v3.2-ablation-verdict.md`. (Added two faithful-miniature toggles to `src/v3/model.py` for this: `use_pos_emb`, `share_blocks`.)

**True-minimum scale pass** — scale v3.1 *down* (d_model/n_blocks only; sharing is fatal, so the floor is *distinct* blocks). Smallest that still beats sf_easy ≥50%:

| shape | params | sf_easy W% | clears? |
|---|---:|---:|---|
| d32/b8 (v3.1‑eq ref) | 116k | 77.0 | — |
| **d24/b8** | **70.6k** | **~52** | ✅ (confirmed, 520 games) |
| d20/b8 | 52k | 37.5 | ❌ |
| d16/b8 | 37k | 19.2 | ❌ |
| d12/b8 | 25k | 10.8 | ❌ |

The cliff is steep and **width-bound**; depth can't rescue narrow width. **True minimum = d24/b8 ≈ 70.6k** (`model/v3/truemin/T1-d24b8/`). Orchestrators: `run_v3_2_explore.py`, `run_v3_2_batch2.py`, `run_v3_truemin.py`.

**Decision:** the shipped hero stays **v3.1‑eq (116k)** — the model the web tool visualizes; d24/b8 is the documented research floor, not shipped.

## Appendix — key paths
- Trainer/infra: `src/v3/pack_agg.py`, `src/v3/train_agg_fast.py`; corpus `data/v2/agg_100M_packed/`.
- Orchestrators: `eval/v3/run_micro_campaign.py`, `eval/v3/run_nano_search.py`, `eval/v3/run_v3_1_ablation.py`.
- Checkpoints: `model/v3/v3-micro-tau/`, `model/v3/micro/M1-len20/`, `model/v3/v3-nano-tau/`, `model/v3/v3.1/{v3.1-eq,v3.1-pm}/`.
- Eval CSVs: `eval/v3/campaign_*_sf.csv`, `eval/v3/campaign_rr.csv`, `eval/v3/nano_*_sf.csv`, `eval/v3/v3_1_*_sf.csv`. Logs in `eval/v3/logs/` and `logs/`.
