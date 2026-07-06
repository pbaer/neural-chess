// MoveHistory — SAN move list in numbered pairs (chess.js history), each move
// annotated with the model's value-head reading at that ply (when available) so
// you can watch how the judged White-vs-Black advantage shifted over the game.

import type { ValueSample } from '../../../core/index.ts';

/** Compact per-move value badge: a centered White(+)/Black(−) bar + signed number. */
function MoveValue({ whiteValue }: { whiteValue: number }) {
  const mag = Math.min(1, Math.abs(whiteValue));
  const halfPct = mag * 50; // each side fills up to half the track from the center
  const num = (whiteValue >= 0 ? '+' : '') + whiteValue.toFixed(2);
  const favored = whiteValue > 0.05 ? 'White better' : whiteValue < -0.05 ? 'Black better' : 'roughly even';
  return (
    <span className="move-value" title={`Value head: ${num} (${favored})`}>
      <span className="move-value-bar" aria-hidden>
        {whiteValue >= 0 ? (
          <span className="value-fill-w" style={{ left: '50%', width: `${halfPct}%` }} />
        ) : (
          <span className="value-fill-b" style={{ left: `${50 - halfPct}%`, width: `${halfPct}%` }} />
        )}
        <span className="value-mid" />
      </span>
      <span className="move-value-num">{num}</span>
    </span>
  );
}

/** One SAN cell (may be empty) with its value badge underneath when the model has one. */
function MoveCell({ san, whiteValue }: { san: string; whiteValue?: number }) {
  if (!san) return <span className="move-cell" />;
  return (
    <span className="move-cell">
      <span className="move-san">{san}</span>
      {whiteValue !== undefined && <MoveValue whiteValue={whiteValue} />}
    </span>
  );
}

export function MoveHistory({
  sanHistory,
  valueHistory = [],
}: {
  sanHistory: string[];
  valueHistory?: ValueSample[];
}) {
  // ply (1-based half-move index) → White-framed value head reading.
  const valueByPly = new Map(valueHistory.map((v) => [v.ply, v.whiteValue]));
  const rows: Array<{ n: number; w: string; b: string; wPly: number; bPly: number }> = [];
  for (let i = 0; i < sanHistory.length; i += 2) {
    rows.push({ n: i / 2 + 1, w: sanHistory[i], b: sanHistory[i + 1] ?? '', wPly: i + 1, bPly: i + 2 });
  }
  return (
    <div className="move-history">
      <div className="panel-title">Moves</div>
      <ol className="move-list">
        {rows.length === 0 && <li className="move-empty">No moves yet.</li>}
        {rows.map((row) => (
          <li key={row.n} className="move-row">
            <span className="move-num">{row.n}.</span>
            <MoveCell san={row.w} whiteValue={valueByPly.get(row.wPly)} />
            <MoveCell san={row.b} whiteValue={valueByPly.get(row.bPly)} />
          </li>
        ))}
      </ol>
    </div>
  );
}
