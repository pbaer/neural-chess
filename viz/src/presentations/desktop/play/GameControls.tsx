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
  // FEN-setup mode: the "Paste FEN" row is only revealed once the user clicks the
  // FEN button. Starting a fresh game (White/Black) collapses it again.
  const [fenMode, setFenMode] = useState(false);

  const start = (color: Color) => {
    setFenMode(false);
    setFenError(null);
    store.getState().newGame(color);
  };

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
        <button
          className={'btn' + (fenMode ? ' btn-active' : '')}
          onClick={() => setFenMode((on) => !on)}
          disabled={disabled}
        >
          FEN
        </button>
      </div>

      <label className="control-row pick-toggle assist-toggle">
        <input
          type="checkbox"
          checked={state.assist}
          onChange={(e) => store.getState().setAssist(e.target.checked)}
          disabled={disabled}
        />
        <span className="control-label">Show model hints of its top moves for you</span>
      </label>

      {fenMode && (
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
              if (ok) {
                setFen('');
                setFenMode(false);
              }
            }}
          >
            Load
          </button>
        </div>
      )}
      {fenMode && fenError && <div className="error-text">{fenError}</div>}
    </div>
  );
}
