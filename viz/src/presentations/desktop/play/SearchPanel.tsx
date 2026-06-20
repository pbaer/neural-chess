// SearchPanel — the MCTS visualization. Shows, while the model "thinks" (and as
// the final state after the move): a live progress indicator (sims / elapsed),
// the backed-up root evaluation, the principal variation, and a PUCT stats table
// for the top root moves (visits N, mean value Q, prior P, and the resulting
// PUCT score) — making the selection formula tangible. The on-board visit-
// weighted arrows are drawn by Board.tsx; this is the numeric companion.

import type { Color, SearchSnapshot } from '../../../core/index.ts';

export interface SearchPanelProps {
  search: SearchSnapshot | null;
  thinking: boolean;
  /** The model's color, to frame the root eval as White(+)/Black(−). */
  modelColor: Color;
  /** uci the user is hovering in the table → accent the matching board arrow. */
  onHover?: (uci: string | null) => void;
  hoverUci?: string | null;
}

export function SearchPanel({ search, thinking, modelColor, onHover, hoverUci }: SearchPanelProps) {
  if (!search) return null;
  const { children, pv, simsDone, totalSims, elapsedMs, rootEval } = search;
  // Re-frame the side-to-move root eval into a fixed White(+)/Black(−) value.
  const whiteEval = modelColor === 'w' ? rootEval : -rootEval;
  const pct = totalSims > 0 ? Math.min(100, (simsDone / totalSims) * 100) : 0;
  const maxN = children.length > 0 ? children[0].n : 1;

  return (
    <div className="search-panel">
      <div className="search-head">
        <span className="panel-title">
          Tree search {thinking ? <span className="search-spinner" aria-hidden /> : null}
          {thinking ? 'thinking…' : 'result'}
        </span>
        <span className="search-stats-line">
          {simsDone} sims · {(elapsedMs / 1000).toFixed(2)}s
        </span>
      </div>

      {thinking && (
        <div className="search-progress" aria-hidden>
          <div className="search-progress-fill" style={{ width: `${pct}%` }} />
        </div>
      )}

      <div className="search-eval">
        <span
          className="search-eval-label"
          title="Backed-up value: the search’s refined position score, averaged up the tree from all the trials (mover’s side: +1 winning, −1 losing)."
        >
          Score after search
        </span>
        <span className={'search-eval-num ' + (whiteEval >= 0 ? 'eval-w' : 'eval-b')}>
          {(whiteEval >= 0 ? '+' : '') + whiteEval.toFixed(2)}
        </span>
        <span className="search-eval-cap">{evalWord(whiteEval)}</span>
      </div>

      {pv.length > 0 && (
        <div className="search-pv" title="Principal variation — the move sequence the search currently thinks is best for both sides.">
          <span className="search-pv-label">Best line</span>
          <span className="search-pv-line">{pv.map((m) => m.san).join(' ')}</span>
        </div>
      )}

      <table className="search-table">
        <thead>
          <tr>
            <th className="st-move">move</th>
            <th className="st-n" title="N — how many what-if trials explored this move">N</th>
            <th className="st-bar" />
            <th className="st-q" title="Q — how those trials turned out, mover’s side (+1 best … −1 worst)">Q</th>
            <th className="st-p" title="P — the model’s first-glance probability for this move">P</th>
            <th className="st-puct" title="PUCT selection score = Q + c·P·√ΣN/(1+N) — balances exploit (Q) vs. explore (P, low N)">PUCT</th>
          </tr>
        </thead>
        <tbody onMouseLeave={() => onHover?.(null)}>
          {children.map((c, i) => (
            <tr
              key={c.uci}
              className={(i === 0 ? 'st-best ' : '') + (c.uci === hoverUci ? 'st-hover' : '')}
              onMouseEnter={() => onHover?.(c.uci)}
            >
              <td className="st-move">{c.san}</td>
              <td className="st-n">{c.n}</td>
              <td className="st-bar">
                <span className="st-bar-fill" style={{ width: `${(c.n / maxN) * 100}%` }} />
              </td>
              <td className="st-q">{(c.q >= 0 ? '+' : '') + c.q.toFixed(2)}</td>
              <td className="st-p">{(c.p * 100).toFixed(1)}%</td>
              <td className="st-puct">{c.puct.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <p className="search-legend">
        Each row is a candidate move. <b>N</b> = how many what-if trials explored it (the move with the most trials is the one played);{' '}
        <b>Q</b> = how those trials turned out, +1 great to −1 bad, from the mover’s side; <b>P</b> = the model’s first-glance hunch for the move;{' '}
        <b>PUCT</b> = the score that balances trying moves that already scored well (Q) against promising or rarely-tried ones (P, low N), deciding what to explore next.
      </p>
    </div>
  );
}

function evalWord(whiteEval: number): string {
  const m = Math.abs(whiteEval);
  const who = whiteEval > 0.05 ? 'White' : whiteEval < -0.05 ? 'Black' : null;
  if (!who) return 'roughly even';
  const s = m > 0.6 ? 'winning' : m > 0.25 ? 'clearly better' : 'slightly better';
  return `${who} ${s}`;
}
