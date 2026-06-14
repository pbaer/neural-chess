// MoveCandidates — the move-assistant suggestion list. On the HUMAN's turn (assist
// mode), shows the model's top-K predicted moves for the human's OWN side, ranked
// by policy probability; the strongest is the most prominent and less likely ones
// fade out. These are HINTS — clicking a row plays it as the human's own move (a
// convenience; the human can equally drag/click a piece). The board draws matching
// arrows (see Board.tsx) so each suggestion is also spatial.

import type { GameStore, ModelCandidate } from '../../../core/index.ts';

export interface MoveCandidatesProps {
  store: GameStore;
  candidates: ModelCandidate[];
  /** Report the hovered/focused candidate's uci (or null) so the board can accent its arrow. */
  onHover: (uci: string | null) => void;
}

export function MoveCandidates({ store, candidates, onHover }: MoveCandidatesProps) {
  const top = candidates[0]?.prob || 1;
  return (
    <div className="candidates">
      <div className="panel-title">Model's suggestions for your move — click to play one</div>
      <ol className="cand-list" onMouseLeave={() => onHover(null)}>
        {candidates.map((c, i) => {
          const rel = c.prob / top; // 1 for the best, smaller for less likely
          // Most likely = full strength; less likely = increasingly faded.
          const opacity = 0.4 + 0.6 * rel;
          return (
            <li key={c.uci}>
              <button
                className={'cand-row' + (i === 0 ? ' cand-row-top' : '')}
                style={{ opacity }}
                onClick={() => store.getState().humanMove(c.fromIdx, c.toIdx, c.promotion)}
                onMouseEnter={() => onHover(c.uci)}
                onFocus={() => onHover(c.uci)}
                onBlur={() => onHover(null)}
                title={`${c.san} — ${(c.prob * 100).toFixed(1)}% · click to play this as your move`}
              >
                <span className="cand-rank">{i + 1}</span>
                <span className="cand-san">{c.san}</span>
                <span className="cand-bar" aria-hidden>
                  {/* bar length is relative to the top move (clear ranking); the % is the true probability */}
                  <span className="cand-bar-fill" style={{ width: `${Math.max(3, (c.prob / top) * 100)}%` }} />
                </span>
                <span className="cand-pct">{(c.prob * 100).toFixed(1)}%</span>
              </button>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
