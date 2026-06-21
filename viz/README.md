# Neural Chess

A public, static, **educational** web app that lets a learner (high-school-calculus level) **see and play** a tiny square-token chess transformer — every value inspectable from the architecture diagram down to a single scalar weight and the literal arithmetic of one operation.

The tool is **architecture-version-neutral**: it never reads a PyTorch checkpoint directly. A Python exporter derives a versioned, self-describing **Model Capsule** — `manifest.json` (incl. an architecture `graph` of typed stages) + `weights.bin` + `config.json` — and the tool renders whatever architecture the Model Capsule declares. "v3"/"v3.1" appears only in a model's `arch` metadata and its id; nothing in the tool's code is bound to a specific version.

> **Status: BUILT, RUNNING & DEPLOYED (small tweaks going forward).** Milestones **M1 (parity engine) → M2 (play loop + board) → M3 (visualization telescope)** are complete and have been iterated extensively, plus an in-browser MCTS engine, value-adaptive play, a move-assistant, a calibrated Elo strength estimate, and mobile polish. `tsc --noEmit` is clean and the full test suite passes (695 tests, including the PyTorch forward-parity assertions — identical argmax + value error < 1e-4 vs the reference model). Live at [pbaer.github.io/neural-chess](https://pbaer.github.io/neural-chess/) (GitHub Pages, auto-deploys on `viz/**` pushes to `master`). Run locally: `npm install` then `npm run dev` in `viz/`, open `http://localhost:5173/neural-chess/`. The full design + milestone roadmap live in [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md).

Current features: play vs. the 116k v3.1 hero in either **one-shot** (single forward pass per move) or **MCTS** ("think harder", PUCT search using only the model's own P/V; default 100 sims with early cutoff on clear positions, up to 300) mode, with a value-head readout, **value-adaptive move variety** (a sigmoid of the value head — sharpen when losing, relax when winning, never play a strong-player-bad move), and a **move-assistant** that shows the model's top suggestions from the human's side. The Model Configuration panel also shows a live **estimated Elo** for the current mode/sims/variety, calibrated by playing the model's GPU twin against Stockfish 18 (see the repo README and `src/core/strength/elo.ts`). Plus an always-open **Model Inspector** — per-stage expand/collapse sections drilling from the architecture graph → real activation/weight heatmaps (tied to the 64 board squares) → an individual scalar, with contextual explanations, a policy-head view (top-N move arrows + piece-to-move board shading), and a light/dark theme. All driven off the capsule `graph` + `kind`-keyed registries (version-neutral).

There's also a **story page** (`story.html` → `src/story/`) — a narrative, general-audience account of the project (principles, the journey, what worked and what didn't, with inline SVG visuals) linked from the main page's subtitle. It's a second Vite entry, so `npm run build` emits both `index.html` and `story.html`.

## Decisions locked
- **Inference:** hand-written TypeScript forward pass (no ONNX), in a Web Worker, with a PyTorch parity test (identical argmax move required).
- **Stack:** React + Vite + TypeScript; SVG/Canvas2D/WebGL render split under one shared zoom camera.
- **Hero model:** the capsule under `public/weights/v3.1-nano/` is **`D-a10-t2`** — a 116k pure-transformer (no conv stem) net **distilled** from the `v3-37M` teacher (policy-equal to the human-trained `v3.1-eq`, stronger value head → markedly stronger under MCTS). See the repo README's *Distillation* section.
- **Search:** in-browser PUCT MCTS is **built** (it was originally a future seam) — uses only the model's own policy priors and value, no chess heuristics (Principle-3 carve-out).
- **Scope:** read-only (no weight editing); play vs the model with a "watch it think" forward-pass trace; the Model Inspector telescope.
- **Assets/licensing:** *cburnett* chess pieces (BSD, see `THIRD_PARTY.md`); a footer carries license (MIT) + attribution + repo link.
- **Target:** desktop/laptop-first, but mobile layout is now supported and actively polished.

## Architectural principle: core ⟂ presentation
`src/core/` is **presentation-agnostic** (engine + trace, model-architecture graph, content data, game/play logic, state) and exposes a stable contract. `src/presentations/desktop/` is one **consumer** of that contract. A future mobile or alternative visualization approach reuses the entire core unchanged — so nothing in the core hardcodes desktop assumptions.

## Layout
See `IMPLEMENTATION_PLAN.md` §"Revised directory tree" for the annotated structure and the 6-milestone roadmap (M1 engine+parity → … → M6 polish/deploy).
