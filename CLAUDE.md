# Neural Chess — repository guide for the coding agent

This file is read automatically by Claude Code (including the GitHub coding agent
that resolves issues). Follow it when implementing changes.

Most issues filed through the app's **Feedback** button are about the in-browser
tool, which lives entirely under **`viz/`** (React + Vite + TypeScript). The
Python training/eval code (`src/`, `model/`, `data/`, `eval/`) is generally out
of scope for feedback-driven work — don't touch it unless the issue is explicitly
about it.

## Project principles (do not violate)

These are load-bearing. If an issue asks for something that breaks one, do **not**
implement it — comment on the issue explaining the conflict and stop.

1. **Human games only.** The model is trained solely on human games. No engine
   games, self-play, or copying a stronger program's judgments.
2. **Observed signal only.** Supervision uses only pen-and-paper observables
   (positions, moves, outcomes). No computed chess features (material, king
   safety, …) as inputs or labels.
3. **One forward pass at inference**, with a single carve-out: AlphaZero-style
   PUCT MCTS that uses *only* the model's own policy/value (no chess heuristics).

Also: the corpus and weights are **CC0**; do not reintroduce TWIC data anywhere.
The web tool is **architecture-version-neutral** — it reads the Model Capsule
(`viz/public/weights/**`), never a `.pt` checkpoint.

## Validating a change (REQUIRED before opening / updating a PR)

Run everything from the `viz/` directory. All must pass:

```bash
npm ci
npm run typecheck   # tsc --noEmit — must be clean
npm test            # full deterministic suite (vitest)
npm run build       # tsc + vite build for both entries (index.html + story.html)
```

- The suite includes **PyTorch forward-parity** tests (`tests/parity/`) that assert
  the TS engine matches the reference model (identical argmax, value error < 1e-4).
  Do not weaken or skip them. Only touch the golden fixtures
  (`viz/tests/parity/fixtures/**/golden.*`) if you deliberately regenerate them
  with the Python exporter — otherwise leave them alone. `viz/public/weights/**`
  is the shipped Model Capsule (`capsule.json` + `weights.bin`) — dev/test
  fixtures must not live there.
- All tests must be **deterministic and run without an LLM or network** (for speed
  and cost). Never add a test that calls a model API.

## Test-coverage requirement (every behavior change needs tests)

A PR that changes or adds functionality **must** include tests that cover it.
Reviews will reject changes with new untested behavior.

- **UX / React component changes** → add a component test named `*.dom.test.tsx`
  (jsdom + React Testing Library). Follow the existing patterns:
  - `src/feedback/Feedback.dom.test.tsx` — dialog open/close, form validation,
    the composed submit URL, all close paths (a full interactive example).
  - `src/presentations/desktop/play/PlayUi.dom.test.tsx` — rendering pure
    presentational components from plain props.
  Files matching `*.dom.test.tsx` run in jsdom automatically; plain `*.test.ts`
  files run in fast Node. Assert real user-visible behavior (roles, text, enabled/
  disabled state, what a click produces) — not implementation details.
- **Core / logic changes** (`src/core/**`) → add a Node `*.test.ts` next to the
  code, exercising inputs → outputs directly (see `src/core/**/*.test.ts` and
  `tests/play/playloop.smoke.test.ts`).
- Check coverage of what you changed with `npm run test:coverage`. New/changed
  files should be well-covered (the existing UX modules sit at ~100% lines); don't
  leave new branches or components untested.

## Conventions

- Match the surrounding code's style, naming, and comment density. TypeScript is
  strict; keep it clean under `tsc --noEmit`.
- Respect the **core ⟂ presentation** split: `src/core/` is presentation-agnostic
  (engine, model graph, game/play logic, state) and must not import from
  `src/presentations/`. UI lives in `src/presentations/desktop/`.
- Keep changes minimal and focused on the issue. Don't do unrelated refactors.

## Shipping (how the agent workflow uses your changes)

The GitHub agent workflow (`.github/workflows/agent-implement.yml`) is fully
automatic: when @pbaer opens an issue, you implement it, then the workflow runs
an independent validation gate (`npm ci && npm run typecheck && npm test &&
npm run build`) and — only if green — commits straight to `master` and deploys.
There are no pull requests and no review step.

- **Do not run git, push, or deploy yourself** — just leave your finished,
  validated changes in the working tree; the workflow handles committing and
  shipping. (The only exception is the decline path: if the request violates a
  principle, make no code changes and `gh issue comment` your reasoning, then stop.)
- **Your validation is the gate.** The workflow re-runs typecheck/test/build and
  will refuse to ship (and comment on the issue) if they fail, so make sure they
  genuinely pass before you finish. A change with no tests, or failing ones, does
  not ship.
