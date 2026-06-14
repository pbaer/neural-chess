// SearchControls — the MCTS ("think harder") on/off toggle + live-adjustable
// search settings (simulation budget, time budget, c_puct, root exploration
// temperature). Off by default: when off, the model plays its one-shot argmax
// move exactly as before. Changing a setting takes effect on the next move.

import type { GameState, GameStore } from '../../../core/index.ts';

export interface SearchControlsProps {
  store: GameStore;
  state: GameState;
  disabled: boolean;
}

export function SearchControls({ store, state, disabled }: SearchControlsProps) {
  const m = state.mcts;
  return (
    <div className="mcts-controls">
      <label className="control-row mcts-toggle">
        <input
          type="checkbox"
          checked={m.enabled}
          onChange={(e) => store.getState().setMctsEnabled(e.target.checked)}
          disabled={disabled}
        />
        <span className="control-label">
          <strong>Think harder (MCTS)</strong> — let the model search a tree with its own policy &amp; value
        </span>
      </label>

      {m.enabled && (
        <div className="mcts-settings">
          <div className="mcts-hint">
            AlphaZero-style PUCT search. Uses <em>only</em> the model's move priors (P) and position value (V) — no
            chess heuristics. Stops at whichever budget (sims or time) is hit first.
          </div>

          <label className="mcts-slider-row">
            <span className="mcts-slider-label">Simulations</span>
            <input
              type="range"
              min={32}
              max={800}
              step={16}
              value={m.sims}
              onChange={(e) => store.getState().setMctsSettings({ sims: parseInt(e.target.value, 10) })}
              disabled={disabled}
            />
            <span className="mcts-slider-val">{m.sims}</span>
          </label>

          <label className="mcts-slider-row">
            <span className="mcts-slider-label">Time budget</span>
            <input
              type="range"
              min={250}
              max={3000}
              step={250}
              value={m.timeMs}
              onChange={(e) => store.getState().setMctsSettings({ timeMs: parseInt(e.target.value, 10) })}
              disabled={disabled}
            />
            <span className="mcts-slider-val">{(m.timeMs / 1000).toFixed(2)}s</span>
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
              onChange={(e) => store.getState().setMctsSettings({ cPuct: parseFloat(e.target.value) })}
              disabled={disabled}
            />
            <span className="mcts-slider-val">{m.cPuct.toFixed(1)}</span>
          </label>

          <label className="mcts-slider-row">
            <span className="mcts-slider-label" title="Root move-selection temperature (0 = strongest)">
              Exploration
            </span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={m.temperature}
              onChange={(e) => store.getState().setMctsSettings({ temperature: parseFloat(e.target.value) })}
              disabled={disabled}
            />
            <span className="mcts-slider-val">{m.temperature <= 0 ? 'best' : `T=${m.temperature.toFixed(2)}`}</span>
          </label>
        </div>
      )}
    </div>
  );
}
