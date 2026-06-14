// ModelConfig — the single "Model Configuration" block. Picks the play mode
// (One-Shot vs MCTS) and shows ONLY the params relevant to that mode:
//   One-Shot → Move variety.
//   MCTS     → Move variety + Max simulations + c_puct + Early-cutoff threshold.
// "Move variety" is shared by both modes; selection is value-adaptive and bounded
// (sharpens when losing, relaxes when winning, never plays an unreasonable move).

import {
  MCTS_MAX_SIMS,
  MCTS_MIN_SIMS,
  MCTS_MIN_CUTOFF,
  MCTS_MAX_CUTOFF,
  type GameState,
  type GameStore,
} from '../../../core/index.ts';

export interface ModelConfigProps {
  store: GameStore;
  state: GameState;
  disabled: boolean;
}

export function ModelConfig({ store, state, disabled }: ModelConfigProps) {
  const mctsOn = state.mcts.enabled;
  const m = state.mcts;
  const set = store.getState;

  return (
    <div className="model-config">
      <div className="model-config-title">Model Configuration</div>

      <div className="mode-toggle" role="tablist" aria-label="Play mode">
        <button
          role="tab"
          aria-selected={!mctsOn}
          className={'btn mode-btn' + (!mctsOn ? ' btn-active' : '')}
          onClick={() => set().setMctsEnabled(false)}
          disabled={disabled}
        >
          One-Shot
        </button>
        <button
          role="tab"
          aria-selected={mctsOn}
          className={'btn mode-btn' + (mctsOn ? ' btn-active' : '')}
          onClick={() => set().setMctsEnabled(true)}
          disabled={disabled}
        >
          MCTS
        </button>
      </div>
      <div className="model-config-hint">
        {mctsOn
          ? 'AlphaZero-style PUCT search using only the model’s priors (P) and value (V) — no chess heuristics. Runs up to the max sims, cutting off early once one move is clearly best.'
          : 'One forward pass per move: the model’s policy picks the move directly.'}
      </div>

      {/* Move variety — shared by both modes. */}
      <div className="mcts-slider-row">
        <span className="mcts-slider-label" title="How much the model varies its play (value-adaptive)">
          Move variety
        </span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={state.variety}
          onChange={(e) => set().setVariety(parseFloat(e.target.value))}
          disabled={disabled}
          aria-label="Move variety"
        />
        <span className="mcts-slider-val">{state.variety <= 0 ? 'top' : state.variety.toFixed(2)}</span>
      </div>
      <div className="model-config-hint">
        {state.variety <= 0
          ? 'Deterministic: always the model’s single best move.'
          : 'Sharpens to the best move when losing, explores more when winning — but never a move a strong player would reject.'}
      </div>

      {mctsOn && (
        <div className="mcts-settings">
          <label className="mcts-slider-row">
            <span className="mcts-slider-label" title="Upper bound on simulations per move">
              Max simulations
            </span>
            <input
              type="range"
              min={MCTS_MIN_SIMS}
              max={MCTS_MAX_SIMS}
              step={10}
              value={m.sims}
              onChange={(e) => set().setMctsSettings({ sims: parseInt(e.target.value, 10) })}
              disabled={disabled}
            />
            <span className="mcts-slider-val">{m.sims}</span>
          </label>

          <label className="mcts-slider-row">
            <span className="mcts-slider-label" title="PUCT exploration constant">
              c_puct
            </span>
            <input
              type="range"
              min={0}
              max={4}
              step={0.1}
              value={m.cPuct}
              onChange={(e) => set().setMctsSettings({ cPuct: parseFloat(e.target.value) })}
              disabled={disabled}
            />
            <span className="mcts-slider-val">{m.cPuct.toFixed(1)}</span>
          </label>

          <label className="mcts-slider-row">
            <span
              className="mcts-slider-label"
              title="Stop early once the top move reaches this share of visits"
            >
              Early cutoff
            </span>
            <input
              type="range"
              min={MCTS_MIN_CUTOFF}
              max={MCTS_MAX_CUTOFF}
              step={0.05}
              value={m.cutoffThreshold}
              onChange={(e) => set().setMctsSettings({ cutoffThreshold: parseFloat(e.target.value) })}
              disabled={disabled}
            />
            <span className="mcts-slider-val">{Math.round(m.cutoffThreshold * 100)}%</span>
          </label>
        </div>
      )}
    </div>
  );
}
