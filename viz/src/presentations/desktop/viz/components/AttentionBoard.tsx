// AttentionBoard — the marquee "the 64 tokens ARE the 64 squares" view. Pick a
// query square; the board colors how much that query attends to every other
// square (the real probs[head][query][*] from the trace), with piece glyphs in
// the engine frame (moving side at the bottom). Hovering a key square reaches the
// exact scalar and decomposes the score into q·k/√d + geometry-bias.

import { useMemo, useState } from 'react';
import { relIndex, type Color } from '../../../../core/index.ts';
import { parseBoard } from '../../parseBoard.ts';
import { Piece } from '../../play/pieces.tsx';
import { niceSeq, rangeOf, sequential, rgbCss } from '../render/colormap.ts';
import { Legend } from '../render/Heatmap.tsx';
import type { ScalarView } from '../scalar/ScalarInspector.tsx';

// The engine's (64,64)→[0,224] relative-position table (query i, key j).
const REL_IDX = relIndex();

/** Engine-frame token s → algebraic label, accounting for the black mirror. */
function tokenLabel(s: number, turn: Color): string {
  const real = turn === 'w' ? s : s ^ 56;
  return String.fromCharCode(97 + (real & 7)) + String.fromCharCode(49 + (real >> 3));
}

export interface AttentionBoardProps {
  /** probs[heads,64,64] from the trace. */
  probs: Float32Array;
  /** scores[heads,64,64] from the trace (for the breakdown). */
  scores?: Float32Array;
  /** rel_bias[heads,225] from the capsule (optional geometry bias). */
  relBias?: Float32Array;
  heads: number;
  fen: string;
  turn: Color;
  onScalar: (v: ScalarView | null) => void;
}

const LIGHT = '#3a4150';
const DARK = '#2b313c';

export function AttentionBoard({ probs, scores, relBias, heads, fen, turn, onScalar }: AttentionBoardProps) {
  const [head, setHead] = useState(0); // -1 == average
  // Query follows the hovered square (transient); a click pins it so it persists
  // when the mouse leaves — the same hover/pin pattern as the other grids.
  const [hoverQ, setHoverQ] = useState<number | null>(null);
  const [pinQ, setPinQ] = useState(36); // e5-ish default
  const query = hoverQ ?? pinQ;
  const board = useMemo(() => parseBoard(fen), [fen]);

  // Attention row for query q at the current head: probs[head][q][*] (avg over heads when head<0).
  const rowFor = (q: number): Float32Array => {
    const out = new Float32Array(64);
    if (head >= 0) {
      const base = head * 64 * 64 + q * 64;
      for (let j = 0; j < 64; j++) out[j] = probs[base + j];
    } else {
      for (let h = 0; h < heads; h++) {
        const base = h * 64 * 64 + q * 64;
        for (let j = 0; j < 64; j++) out[j] += probs[base + j];
      }
      for (let j = 0; j < 64; j++) out[j] /= heads;
    }
    return out;
  };
  const row = useMemo(() => rowFor(query), [probs, head, query, heads]);
  // Color scale spans this query's actual attention values (its peak rarely
  // reaches 1, so a fixed 0..1 would wash everything out).
  const range = useMemo(() => niceSeq(...rangeOf(row)), [row]);

  // Scalar for a hovered square AS the query: its strongest attention target,
  // with the score decomposition (q·k/√d + geometry bias) for that target.
  function queryScalar(q: number): ScalarView {
    const hLabel = head >= 0 ? `h=${head}` : 'avg';
    const r = rowFor(q);
    let kmax = 0;
    for (let j = 1; j < 64; j++) if (r[j] > r[kmax]) kmax = j;
    const p = r[kmax];
    const terms: ScalarView['terms'] = [{ label: `→ ${tokenLabel(kmax, turn)} (strongest)`, value: p }];
    let formula: string | undefined;
    if (head >= 0 && scores) {
      const s = scores[head * 64 * 64 + q * 64 + kmax];
      terms.push({ label: 'score (pre-softmax)', value: s });
      if (relBias) {
        const bias = relBias[head * 225 + REL_IDX[q * 64 + kmax]];
        terms.push({ label: 'geometry bias', value: bias });
        terms.push({ label: 'q·k / √d  (= score − bias)', value: s - bias });
        formula = 'score = q·k/√d + bias ;  prob = softmax(score)';
      } else {
        terms.push({ label: 'q·k / √d  (= score)', value: s });
        formula = 'prob = softmax(score)';
      }
    }
    return {
      name: `attn[${hLabel}][${tokenLabel(q, turn)} → *]`,
      value: p,
      description: `Square ${tokenLabel(q, turn)} attends most strongly to ${tokenLabel(kmax, turn)}; the board colors show the full spread. Hover another square to change the source.`,
      formula,
      terms,
    };
  }

  const qFile = query & 7;
  const qDispRow = 7 - (query >> 3); // engine-frame rank → display row (mover at bottom)

  const rows = [0, 1, 2, 3, 4, 5, 6, 7];
  return (
    <div className="attn-board">
      <div className="attn-head-row">
        <span className="control-label">Head</span>
        {Array.from({ length: heads }, (_, h) => (
          <button key={h} className={'btn btn-mini' + (head === h ? ' btn-active' : '')} onClick={() => setHead(h)}>
            {h}
          </button>
        ))}
        <button className={'btn btn-mini' + (head === -1 ? ' btn-active' : '')} onClick={() => setHead(-1)}>
          avg
        </button>
      </div>

      <div className="attn-svg-wrap" onMouseLeave={() => { setHoverQ(null); onScalar(null); }}>
        <svg viewBox="0 0 8 8" className="attn-svg" role="grid" aria-label="Attention from the query square">
          {rows.map((dr) =>
            rows.map((f) => {
              const h = 7 - dr; // engine-frame rank (moving side at bottom)
              const s = h * 8 + f;
              const base = (f + h) % 2 === 1 ? LIGHT : DARK;
              const w = row[s];
              const fill = w > 0.001 ? rgbCss(sequential(w, range[0], range[1])) : base;
              const piece = board[turn === 'w' ? s : s ^ 56];
              return (
                <g
                  key={s}
                  onMouseEnter={() => { setHoverQ(s); onScalar(queryScalar(s)); }}
                  onClick={() => setPinQ(s)}
                  style={{ cursor: 'pointer' }}
                >
                  <rect x={f} y={dr} width={1} height={1} fill={fill} />
                  {piece && (
                    <Piece type={piece.type} color={piece.color} cx={f + 0.5} cy={dr + 0.5} size={0.92} halo />
                  )}
                </g>
              );
            }),
          )}
        </svg>
        {/* Crisp CSS-bordered outline marking the current query square. */}
        <div className="cell-outline" style={{ left: `${qFile * 12.5}%`, top: `${qDispRow * 12.5}%` }} />
      </div>
      <Legend mode="sequential" range={range} />
      <p className="attn-hint">
        Hover a square to make it the <b>query</b> (gold outline); click to pin it. Colors show how much that square
        attends to each other square{head === -1 ? ' (averaged over heads)' : ` in head ${head}`}. Board is in the
        network’s frame: the side to move is always at the bottom.
      </p>
    </div>
  );
}
