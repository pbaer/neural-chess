// Desktop presentation shell — the v1 play surface. Consumes the core game store
// via useGame(). The Model Inspector (M3 visualization) is permanently docked as
// a second column to the RIGHT of the board; the play controls live below it.

import { useEffect, useState, useSyncExternalStore } from 'react';
import type { EngineClient, EngineMeta, GameStore } from '../../core/index.ts';
import { Board } from './play/Board.tsx';
import { GameControls } from './play/GameControls.tsx';
import { MoveCandidates } from './play/MoveCandidates.tsx';
import { MoveHistory } from './play/MoveHistory.tsx';
import { ValueGauge } from './play/ValueGauge.tsx';
import { useGame } from './play/useGame.ts';
import { ModelInspector } from './viz/ModelInspector.tsx';
import { ThemeToggle } from './ThemeToggle.tsx';

export function App() {
  const { store, client, capsuleUrl, ready, meta, error } = useGame();
  return (
    <div className="app app-wide">
      <header className="app-header">
        <div className="app-header-text">
          <h1>Neural Chess</h1>
          <p className="subtitle">
            Play a tiny square-token chess transformer — every move is one in-browser forward pass.
          </p>
        </div>
        <ThemeToggle />
      </header>
      {store ? (
        <GameView store={store} client={client} capsuleUrl={capsuleUrl} ready={ready} meta={meta} error={error} />
      ) : (
        <div className="status">{error ?? 'Loading model…'}</div>
      )}
    </div>
  );
}

interface GameViewProps {
  store: GameStore;
  client: EngineClient | null;
  capsuleUrl: string;
  ready: boolean;
  meta: EngineMeta | null;
  error: string | null;
}

function GameView({ store, client, capsuleUrl, ready, meta, error }: GameViewProps) {
  const state = useSyncExternalStore(store.subscribe, store.getState, store.getState);
  const thinking = state.status === 'thinking';
  const choosing = state.status === 'choosing';
  const boardDisabled = !ready || thinking || choosing || state.status === 'over';

  // Which candidate move (by uci) the user is hovering in the picker — used to
  // accent the matching arrow on the board. Cleared when not choosing.
  const [hoverUci, setHoverUci] = useState<string | null>(null);
  useEffect(() => {
    if (!choosing) setHoverUci(null);
  }, [choosing]);

  const statusLine = (() => {
    if (error) return error;
    if (!ready) return 'Loading model…';
    if (state.error) return state.error;
    if (state.status === 'over') return state.resultText ?? 'Game over';
    if (choosing) return 'Pick a move for the model';
    if (thinking) return 'Model is thinking…';
    if (state.turn === state.humanColor) return `Your move (${state.humanColor === 'w' ? 'White' : 'Black'})`;
    return 'Model to move';
  })();

  return (
    <main className="layout layout-viz">
      <section className="board-col">
        <Board store={store} state={state} disabled={boardDisabled} hoverUci={hoverUci} />
        <div className={'status' + (state.inCheck && state.status !== 'over' ? ' status-check' : '')}>
          {statusLine}
          {state.inCheck && state.status !== 'over' && <span className="check-tag"> · check</span>}
        </div>

        {choosing && state.candidates && (
          <MoveCandidates store={store} candidates={state.candidates} onHover={setHoverUci} />
        )}

        <GameControls store={store} state={state} disabled={!ready} />

        <ValueGauge model={state.lastModelMove} modelColor={state.humanColor === 'w' ? 'b' : 'w'} />

        <div className="model-move">
          <div className="panel-title">Model move</div>
          {state.lastModelMove ? (
            <div className="model-move-body">
              <span className="model-san">{state.lastModelMove.san}</span>
              <span className="model-meta">
                {state.lastModelMove.uci} · p={(state.lastModelMove.policyProb * 100).toFixed(1)}%
              </span>
            </div>
          ) : (
            <div className="model-move-body model-move-empty">—</div>
          )}
        </div>

        <MoveHistory sanHistory={state.sanHistory} />

        {meta && (
          <div className="model-info">
            {meta.modelId} · arch {meta.arch} · {meta.paramCount.toLocaleString()} params
          </div>
        )}
      </section>

      <ModelInspector client={client} capsuleUrl={capsuleUrl} fen={state.fen} />
    </main>
  );
}
