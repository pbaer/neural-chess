// GameControls — new game (choose side), undo, and FEN setup. All actions go
// through the core game store; the engine auto-replies when it's the model's
// turn (e.g. play as Black → model moves first).

import { useState } from 'react';
import type { Color, GameState, GameStore } from '../../../core/index.ts';

export interface GameControlsProps {
  store: GameStore;
  state: GameState;
  disabled: boolean;
}

export function GameControls({ store, state, disabled }: GameControlsProps) {
  const [fen, setFen] = useState('');
  const [fenError, setFenError] = useState<string | null>(null);

  const start = (color: Color) => store.getState().newGame(color);

  return (
    <div className="controls">
      <div className="control-row">
        <span className="control-label">New game as</span>
        <button
          className={'btn' + (state.humanColor === 'w' ? ' btn-active' : '')}
          onClick={() => start('w')}
          disabled={disabled}
        >
          White
        </button>
        <button
          className={'btn' + (state.humanColor === 'b' ? ' btn-active' : '')}
          onClick={() => start('b')}
          disabled={disabled}
        >
          Black
        </button>
        <button className="btn" onClick={() => store.getState().undo()} disabled={disabled}>
          Undo
        </button>
      </div>

      <label className="control-row pick-toggle">
        <input
          type="checkbox"
          checked={state.pickMode}
          onChange={(e) => store.getState().setPickMode(e.target.checked)}
          disabled={disabled}
        />
        <span className="control-label">Let me pick the model's move from its top suggestions</span>
      </label>

      <div className="control-row temp-row">
        <span className="control-label">Move variety</span>
        <input
          type="range"
          className="temp-slider"
          min={0}
          max={1.5}
          step={0.05}
          value={state.temperature}
          onChange={(e) => store.getState().setTemperature(parseFloat(e.target.value))}
          disabled={disabled || state.pickMode}
          aria-label="Move-selection temperature"
        />
        <span className="temp-val">{state.temperature <= 0 ? 'top move' : `T = ${state.temperature.toFixed(2)}`}</span>
      </div>
      <div className="temp-hint">
        {state.pickMode
          ? "You're choosing the model's move, so temperature is ignored."
          : state.temperature <= 0
            ? 'Deterministic: the model always plays its single best move.'
            : 'The model samples among its preferred moves — higher is more adventurous (and a bit weaker).'}
      </div>

      <div className="control-row">
        <input
          className="fen-input"
          placeholder="Paste FEN to set up a position…"
          value={fen}
          onChange={(e) => setFen(e.target.value)}
          spellCheck={false}
        />
        <button
          className="btn"
          disabled={disabled || !fen.trim()}
          onClick={() => {
            const ok = store.getState().loadFen(fen.trim());
            setFenError(ok ? null : (store.getState().error ?? 'Invalid FEN'));
            if (ok) setFen('');
          }}
        >
          Load
        </button>
      </div>
      {fenError && <div className="error-text">{fenError}</div>}
    </div>
  );
}
