// ModelConfig — the single "Model Configuration" block. A shared "Move variety"
// slider (it applies to both modes) leads, then a One-Shot vs MCTS mode toggle.
// MCTS exposes one tunable, "Max simulations"; the other PUCT knobs (c_puct and
// the early-cutoff threshold) just use their defaults. "Move variety" is
// value-adaptive and bounded (sharpens when losing, relaxes when winning, never
// plays an unreasonable move).

import {
  MCTS_MAX_SIMS,
  MCTS_MIN_SIMS,
  estimateElo,
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
  const elo = estimateElo(mctsOn, m.sims, state.variety);

  return (
    <div className="model-config">
      <div
        className="model-config-title"
        title={
          'The ~Elo is this model’s approximate playing strength for the current settings, calibrated by playing it ' +
          'against Stockfish 18. It updates with the mode, the max-simulations slider, and Move variety (more variety ' +
          'plays a little weaker). Anchored to Stockfish’s own Elo scale, which only roughly matches FIDE / online ratings.'
        }
      >
        Model Configuration <span className="model-config-elo">(~{elo.elo} Elo)</span>
      </div>

      {/* Move variety — shared by both modes, so it leads. */}
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
          ? 'At 0 the model always plays its single top move (fully predictable). Slide up to let it sometimes choose other strong moves, so games don’t all feel the same.'
          : 'How willing the model is to play a move other than its top pick. It’s value-aware: it sharpens toward its best move when it judges it’s losing, and allows more variety when it’s comfortably ahead, but it never plays a move a strong player would reject.'}
      </div>

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
          ? 'MCTS (Monte Carlo Tree Search): instead of answering at a glance, the model “thinks ahead”: it plays out many short what-if lines from this position, spending more of them on the moves that look most promising, then plays the move those trials backed up best. The search is steered only by the model’s own move hunches and position scores. No outside chess knowledge is added. It runs up to the simulation cap, but stops early once one move is the clear favorite.'
          : 'One-Shot: the model looks at the position once and names its move: a single pass through the network, with no looking ahead. Fast, and how the model plays by default.'}
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
          <div className="model-config-hint mcts-slider-help">
            How many what-if trials the search may run before it has to move. More trials = stronger play, but it takes longer to move.
          </div>
        </div>
      )}

    </div>
  );
}
