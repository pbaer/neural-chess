# Neural Chess

A public, static, **educational** web app that lets a learner (high-school-calculus level) **see and play** a tiny square-token chess transformer — every value inspectable from the architecture diagram down to a single scalar weight and the literal arithmetic of one operation.

The tool is **architecture-version-neutral**: it never reads a PyTorch checkpoint directly. A Python exporter derives a versioned, self-describing **Model Capsule** — `capsule.json` (a manifest incl. an architecture `graph` of typed stages) + `weights.bin` + `config.json` — and the tool renders whatever architecture the Model Capsule declares. "v3"/"v3.1" appears only in a model's `arch` metadata and its id; nothing in the tool's code is bound to a specific version.

> **Status: BUILT, RUNNING & DEPLOYED (small tweaks going forward).** Milestones **M1 (parity engine) → M2 (play loop + board) → M3 (visualization telescope)** are complete and have been iterated extensively, plus an in-browser MCTS engine, value-adaptive play, a move-assistant, a calibrated Elo strength estimate, and mobile polish. `tsc --noEmit` is clean and the full test suite passes (730 tests, including the PyTorch forward-parity assertions — identical argmax + value error < 1e-4 vs the reference model). Live at [pbaer.github.io/neural-chess](https://pbaer.github.io/neural-chess/) (GitHub Pages, auto-deploys on `viz/**` pushes to `master`). Run locally: `npm install` then `npm run dev` in `viz/`, open `http://localhost:5173/neural-chess/`. The original design + milestone roadmap are preserved (as a historical record) in [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md).

Current features: play vs. the 116k v3.1 hero in either **one-shot** (single forward pass per move) or **MCTS** ("think harder", PUCT search using only the model's own P/V; default 100 sims with early cutoff on clear positions, up to 300) mode, with a value-head readout, **value-adaptive move variety** (a sigmoid of the value head — sharpen when losing, relax when winning, never play a strong-player-bad move), and a **move-assistant** that shows the model's top suggestions from the human's side. The Model Configuration panel also shows a live **estimated Elo** for the current mode/sims/variety, calibrated by playing the model's GPU twin against Stockfish 18 (see the repo README and `src/core/strength/elo.ts`). Plus an always-open **Model Inspector** — per-stage expand/collapse sections drilling from the architecture graph → real activation/weight heatmaps (tied to the 64 board squares) → an individual scalar, with contextual explanations, a policy-head view (top-N move arrows + piece-to-move board shading), and a light/dark theme. All driven off the capsule `graph` + `kind`-keyed registries (version-neutral).

There's also a **story page** (`story.html` → `src/story/`) — a narrative, general-audience account of the project (principles, the journey, what worked and what didn't, with inline SVG visuals) linked from the main page's subtitle. It's a second Vite entry, so `npm run build` emits both `index.html` and `story.html`.

## Decisions locked
- **Inference:** hand-written TypeScript forward pass (no ONNX), in a Web Worker, with a PyTorch parity test (identical argmax move required).
- **Stack:** React + Vite + TypeScript; SVG boards/arrows plus Canvas2D heatmaps (the originally planned WebGL zoom camera was never needed).
- **Hero model:** the capsule under `public/weights/v3.1-nano/` is **`v3.1-clean-distilled`** — a 116k pure-transformer (no conv stem) net **distilled** from a `v3.1-37M` teacher trained entirely on CC0 Lichess data (policy-equal to the human-trained `v3.1-eq`, much stronger value head → markedly stronger under MCTS). The whole pipeline is license-clean. See the repo README's *Distillation* section.
- **Search:** in-browser PUCT MCTS is **built** (it was originally a future seam) — uses only the model's own policy priors and value, no chess heuristics (Principle-3 carve-out).
- **Scope:** read-only (no weight editing); play vs the model with a "watch it think" forward-pass trace; the Model Inspector telescope.
- **Assets/licensing:** *cburnett* chess pieces (BSD, see `THIRD_PARTY.md`); a footer carries license (MIT) + attribution + repo link.
- **Target:** desktop/laptop-first, but mobile layout is now supported and actively polished.

## Architectural principle: core ⟂ presentation
`src/core/` is **presentation-agnostic** (engine + trace, model-architecture graph, content data, game/play logic, state) and exposes a stable contract. `src/presentations/desktop/` is one **consumer** of that contract. A future mobile or alternative visualization approach reuses the entire core unchanged — so nothing in the core hardcodes desktop assumptions.

## Layout
`src/core/` (engine, model graph, game/search/state logic) + `src/presentations/desktop/` (the UI) + `src/feedback/` and `src/story/` (self-contained pages); tests live next to the code and under `tests/`. (`IMPLEMENTATION_PLAN.md` has the original planned tree, kept as a historical record.)

## Feedback → agentic fix workflow

The app has a built-in loop from **user feedback** to a **shipped change**, driven by an AI coding agent (Claude Code). For the maintainer's own issues it is **fully automatic** — file an issue, and the fix implements, validates, and deploys itself with no further action.

### 1. A user submits feedback (in-app → GitHub issue)

A muted **Feedback** link sits in the footer of both pages (play + story). It opens a small dialog (🐞 Bug / 💡 Feature, title, details) and, on submit, opens a **prefilled GitHub "new issue" URL** in a new tab — the reporter reviews it and clicks *Submit* under their own GitHub account. There is **no backend and no secret**: the site is a static GitHub Pages deploy, so the widget never calls the GitHub API. Bug reports auto-attach environment context (hero model id, FEN, last model move, move number, browser).

- Widget: `src/feedback/Feedback.tsx` (+ `feedback.css`); URL builder is `buildIssueUrl()`.
- Issue Forms for people filing directly on GitHub: `.github/ISSUE_TEMPLATE/{bug_report,feature_request}.yml`. Labeled `feedback` (plus `bug` / `enhancement`).

### 2. The agent implements, validates, and ships

`.github/workflows/agent-implement.yml` runs on `issues: [opened]` — **exactly one run per issue**:

- **Maintainer's own issues auto-run.** Claude implements the change and adds deterministic tests (per `CLAUDE.md`). The workflow then runs an **independent validation gate** — `npm ci && npm run typecheck && npm test && npm run build` — and **only if green** commits straight to `master` (message `Agent: <issue title> (#n)`) and triggers the Pages deploy. No PR, no review, no approval.
- **If validation fails** (or the agent can't complete it), nothing is committed or deployed and the issue gets a comment with the run link. Worst case, file another issue.
- **If the request violates a project principle** (human-games-only, no engine signal, CC0/no-TWIC), the agent makes no changes and comments why.
- **Anyone else's issue** is handled on demand: run the workflow manually (Actions → *Agent — implement issue* → **Run workflow**, or `gh workflow run agent-implement.yml -f issue=<n>`).

### Guard rails

- A ruleset on `master` blocks branch **deletion and force-pushes** (normal pushes are allowed); the maintainer is on its bypass list.
- The agent's `--allowedTools` permit `Bash` (for `npm`) — it runs on maintainer-gated issues only, so the trust boundary is "which issues get run," which the auto-path limits to `@pbaer`'s own.

### Validation & test-coverage policy (deterministic, no LLM)

Every change must be validated by tests that run **without an LLM or network** — for speed and cost, and because the workflow's gate re-runs them before shipping:

- `npm run typecheck` · `npm test` · `npm run build` — all must pass; the suite includes the PyTorch **forward-parity** assertions.
- Component/UX tests use **jsdom + React Testing Library**; files named `*.dom.test.tsx` run in jsdom automatically, plain `*.test.ts` run in fast Node (see `vitest.config.ts`). Examples: `src/feedback/Feedback.dom.test.tsx`, `src/presentations/desktop/play/PlayUi.dom.test.tsx`.
- `npm run test:coverage` reports coverage (v8). New/changed behavior is expected to be well-covered.

The full agent contract — principles, required commands, and the coverage rule — lives in **`CLAUDE.md`** at the repo root. Setup (the `CLAUDE_CODE_OAUTH_TOKEN` secret and the `master` ruleset) is already in place.
