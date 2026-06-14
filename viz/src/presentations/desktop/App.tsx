// Desktop presentation shell — the v1 play surface. Consumes the core game store
// via useGame(). The Model Inspector (M3 visualization) is permanently docked as
// a second column to the RIGHT of the board; the play controls live below it.

import { useEffect, useState, useSyncExternalStore } from 'react';
import type { EngineClient, EngineMeta, GameStore } from '../../core/index.ts';
import { Board } from './play/Board.tsx';
import { GameControls } from './play/GameControls.tsx';
import { MoveCandidates } from './play/MoveCandidates.tsx';
import { MoveHistory } from './play/MoveHistory.tsx';
import { SearchControls } from './play/SearchControls.tsx';
import { SearchPanel } from './play/SearchPanel.tsx';
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
      <Footer />
    </div>
  );
}

const REPO_URL = 'https://github.com/pbaer/neural-chess';

/** Small, muted footer: license, third-party attribution, and a repo link. */
function Footer() {
  return (
    <footer className="app-footer">
      <a href={`${REPO_URL}/blob/master/LICENSE`} target="_blank" rel="noopener noreferrer">
        MIT © Peter Baer
      </a>
      <span className="app-footer-sep" aria-hidden="true">·</span>
      <span>
        chess pieces:{' '}
        <a
          href="https://github.com/lichess-org/lila/tree/master/public/piece/cburnett"
          target="_blank"
          rel="noopener noreferrer"
        >
          cburnett
        </a>{' '}
        (BSD)
      </span>
      <span className="app-footer-sep" aria-hidden="true">·</span>
      <a href={REPO_URL} target="_blank" rel="noopener noreferrer">
        github.com/pbaer/neural-chess
      </a>
    </footer>
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
  const boardDisabled = !ready || thinking || state.status === 'over';

  // Move-assistant suggestions show only on the human's turn (assist on).
  const showSuggestions = state.turn === state.humanColor && !!state.suggestions;

  // Which suggested move (by uci) the user is hovering in the list — used to
  // accent the matching arrow on the board. Cleared when no suggestions show.
  const [hoverUci, setHoverUci] = useState<string | null>(null);
  useEffect(() => {
    if (!showSuggestions) setHoverUci(null);
  }, [showSuggestions]);

  // Which MCTS root move (by uci) the user is hovering in the stats table — used
  // to accent the matching visit-weighted arrow on the board.
  const [searchHoverUci, setSearchHoverUci] = useState<string | null>(null);
  // Hide the live visit-weighted search arrows during the post-search flash beat
  // so only the chosen-move highlight (gold/red, like a hovered pick) shows.
  const showSearchArrows = thinking && state.mcts.enabled && !state.flashMove;
  const modelColor: typeof state.humanColor = state.humanColor === 'w' ? 'b' : 'w';

  const statusLine = (() => {
    if (error) return error;
    if (!ready) return 'Loading model…';
    if (state.error) return state.error;
    if (state.status === 'over') return state.resultText ?? 'Game over';
    if (thinking) return 'Model is thinking…';
    if (state.turn === state.humanColor) return `Your move (${state.humanColor === 'w' ? 'White' : 'Black'})`;
    return 'Model to move';
  })();

  return (
    <main className="layout layout-viz">
      <section className="board-col">
        <Board
          store={store}
          state={state}
          disabled={boardDisabled}
          hoverUci={hoverUci}
          searchChildren={showSearchArrows ? state.search?.children ?? null : null}
          searchHoverUci={searchHoverUci}
        />
        <div className={'status' + (state.inCheck && state.status !== 'over' ? ' status-check' : '')}>
          {statusLine}
          {state.inCheck && state.status !== 'over' && <span className="check-tag"> · check</span>}
        </div>

        {showSuggestions && state.suggestions && (
          <MoveCandidates store={store} candidates={state.suggestions} onHover={setHoverUci} />
        )}

        <GameControls store={store} state={state} disabled={!ready} />

        <SearchControls store={store} state={state} disabled={!ready} />

        {state.search && (
          <SearchPanel
            search={state.search}
            thinking={thinking}
            modelColor={modelColor}
            onHover={setSearchHoverUci}
            hoverUci={searchHoverUci}
          />
        )}

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
