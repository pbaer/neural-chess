# Neural Chess

A public, static, **educational** web app that lets a learner (high-school-calculus level) **see and play** a tiny square-token chess transformer — every value inspectable from the architecture diagram down to a single scalar weight and the literal arithmetic of one operation.

The tool is **architecture-version-neutral**: it never reads a PyTorch checkpoint directly. A Python exporter derives a versioned, self-describing **Model Capsule** — `manifest.json` (incl. an architecture `graph` of typed stages) + `weights.bin` + `config.json` — and the tool renders whatever architecture the Model Capsule declares. "v3"/"v3.1" appears only in a model's `arch` metadata and its id; nothing in the tool's code is bound to a specific version.

> **Status: BUILT & RUNNING (iterating on polish).** Milestones **M1 (parity engine) → M2 (play loop + board) → M3 (visualization telescope)** are complete and have been iterated extensively. `tsc --noEmit` is clean and the full test suite passes (656 tests, including 576 PyTorch forward-parity assertions — identical argmax + value error < 1e-4 vs the reference model). Run it: `npm install` then `npm run dev` in `viz/`, open `http://localhost:5173/neural-chess/`. The full design + milestone roadmap live in [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md).

Current features: play vs. the 116k v3.1 hero (one-shot inference per move, value-head readout, optional top-5 move picker with board arrows, temperature control), and an always-open **Model Inspector** — per-stage expand/collapse sections drilling from the architecture graph → real activation/weight heatmaps (tied to the 64 board squares) → an individual scalar, with contextual explanations and a light/dark theme. All driven off the capsule `graph` + `kind`-keyed registries (version-neutral).

## Decisions locked for v1
- **Inference:** hand-written TypeScript forward pass (no ONNX), in a Web Worker, with a PyTorch parity test (identical argmax move required).
- **Stack:** React + Vite + TypeScript; SVG/Canvas2D/WebGL render split under one shared zoom camera.
- **Hero model:** `v3.1-nano` — the pure-transformer (no conv stem) ~116k-param net (smaller *and* stronger than the conv variant).
- **Scope:** one-shot inference (MCTS is a future seam); shows the value head; read-only (no weight editing in v1); play vs the model with a "watch it think" forward-pass trace.
- **Target:** desktop/laptop-optimized; mobile is out of v1.

## Architectural principle: core ⟂ presentation
`src/core/` is **presentation-agnostic** (engine + trace, model-architecture graph, content data, game/play logic, state) and exposes a stable contract. `src/presentations/desktop/` is one **consumer** of that contract. A future mobile or alternative visualization approach reuses the entire core unchanged — so nothing in the core hardcodes desktop assumptions.

## Layout
See `IMPLEMENTATION_PLAN.md` §"Revised directory tree" for the annotated structure and the 6-milestone roadmap (M1 engine+parity → … → M6 polish/deploy).
