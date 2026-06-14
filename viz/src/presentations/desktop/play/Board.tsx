// SVG chessboard — click-to-move, dependency-free (Unicode glyphs). Orientation
// follows humanColor (human side at the bottom). The model only ranks among
// chess.js's legal moves, so all legality/turn gating comes from the store.

import { useMemo, useState } from 'react';
import type { Color, GameState, GameStore, PieceType, PromotionPiece, RootChildStat } from '../../../core/index.ts';
import { Piece, PieceGlyph } from './pieces.tsx';

interface Cell {
  color: Color;
  type: PieceType;
}

/** Parse a FEN placement field into a 64-array indexed by python-chess square. */
function parseBoard(fen: string): (Cell | null)[] {
  const arr: (Cell | null)[] = new Array(64).fill(null);
  const rows = fen.split(' ')[0].split('/'); // rows[0] = rank 8
  for (let r = 0; r < 8; r++) {
    const rankIdx = 7 - r;
    let file = 0;
    for (const ch of rows[r]) {
      if (ch >= '1' && ch <= '8') {
        file += ch.charCodeAt(0) - 48;
      } else {
        const color: Color = ch === ch.toUpperCase() ? 'w' : 'b';
        arr[rankIdx * 8 + file] = { color, type: ch.toLowerCase() as PieceType };
        file++;
      }
    }
  }
  return arr;
}

export interface BoardProps {
  store: GameStore;
  state: GameState;
  disabled: boolean;
  /** uci of the candidate move the user is hovering in the picker (pick-mode). */
  hoverUci?: string | null;
  /** Root child stats to draw as visit-weighted MCTS arrows (when searching). */
  searchChildren?: RootChildStat[] | null;
  /** uci of the search row the user is hovering (accent that arrow). */
  searchHoverUci?: string | null;
}

const LIGHT = '#ebd9b4';
const DARK = '#9d7b4f';
const ARROW = '#7ec4ff';
const ARROW_HOVER = '#ff4d4d';
const SEARCH_ARROW = '#54d6a0';
const SEARCH_ARROW_HOVER = '#ffd24d';
/** How many top root moves to draw as arrows (keeps the board readable). */
const MAX_SEARCH_ARROWS = 8;

export function Board({ store, state, disabled, hoverUci, searchChildren, searchHoverUci }: BoardProps) {
  const board = useMemo(() => parseBoard(state.fen), [state.fen]);
  const [selected, setSelected] = useState<number | null>(null);
  const [pendingPromo, setPendingPromo] = useState<{ from: number; to: number } | null>(null);
  const orientation = state.humanColor;

  const targets = useMemo<number[]>(
    () => (selected != null ? store.getState().legalTargets(selected) : []),
    [selected, store, state.fen],
  );
  const targetSet = useMemo(() => new Set(targets), [targets]);

  // displayed cell (row r, col c top-left origin) → python-chess square index
  const cellToIdx = (r: number, c: number): number =>
    orientation === 'w' ? (7 - r) * 8 + c : r * 8 + (7 - c);

  // inverse: python-chess square index → displayed cell (top-left origin)
  const idxToCell = (idx: number): { r: number; c: number } => {
    const file = idx & 7;
    const rank = idx >> 3;
    return orientation === 'w' ? { r: 7 - rank, c: file } : { r: rank, c: 7 - file };
  };

  // Pick-mode candidates drive the picker arrows + hover accent. When MCTS (or
  // non-MCTS auto-play) has just chosen a move it publishes `flashMove`; we reuse
  // the EXACT same arrow + square-tint highlight by treating it as a one-element
  // candidate list that is "hovered" — so the chosen move flashes identically
  // before it's played. And non-MCTS auto-play first publishes `previewMoves` (its
  // top policy moves), rendered through the same prob-weighted candidate arrows so
  // the move distribution shows for a beat before the flash.
  const pickCandidates = state.status === 'choosing' ? state.candidates : null;
  const candidates = pickCandidates ?? (state.flashMove ? [state.flashMove] : state.previewMoves);
  const activeHoverUci = state.flashMove ? state.flashMove.uci : hoverUci;
  const hoverCand = candidates && activeHoverUci ? candidates.find((c) => c.uci === activeHoverUci) ?? null : null;

  const kingInCheck = useMemo(() => {
    if (!state.inCheck) return -1;
    const idx = board.findIndex((p) => p && p.type === 'k' && p.color === state.turn);
    return idx;
  }, [board, state.inCheck, state.turn]);

  function isPromotion(from: number, to: number): boolean {
    const p = board[from];
    if (!p || p.type !== 'p') return false;
    const toRank = to >> 3;
    return toRank === 7 || toRank === 0;
  }

  function clickSquare(idx: number): void {
    if (disabled || pendingPromo) return;
    const piece = board[idx];
    if (selected == null) {
      if (piece && piece.color === state.humanColor && state.turn === state.humanColor) setSelected(idx);
      return;
    }
    if (idx === selected) {
      setSelected(null);
      return;
    }
    if (targetSet.has(idx)) {
      if (isPromotion(selected, idx)) {
        setPendingPromo({ from: selected, to: idx });
      } else {
        store.getState().humanMove(selected, idx);
        setSelected(null);
      }
      return;
    }
    // Reselect another own piece, else clear.
    if (piece && piece.color === state.humanColor && state.turn === state.humanColor) setSelected(idx);
    else setSelected(null);
  }

  function choosePromo(piece: PromotionPiece): void {
    if (!pendingPromo) return;
    store.getState().humanMove(pendingPromo.from, pendingPromo.to, piece);
    setPendingPromo(null);
    setSelected(null);
  }

  const rows = [0, 1, 2, 3, 4, 5, 6, 7];
  return (
    <div className="board-wrap">
      <svg viewBox="0 0 8 8" className="board" role="grid" aria-label="Chess board">
        {rows.map((r) =>
          rows.map((c) => {
            const idx = cellToIdx(r, c);
            const fileIdx = idx & 7;
            const rankIdx = idx >> 3;
            const isLight = (fileIdx + rankIdx) % 2 === 1;
            const piece = board[idx];
            const isSel = idx === selected;
            const isLast = state.lastMove && (state.lastMove.fromIdx === idx || state.lastMove.toIdx === idx);
            const isTarget = targetSet.has(idx);
            const isCheck = idx === kingInCheck;
            const isHoverSq = !!hoverCand && (idx === hoverCand.fromIdx || idx === hoverCand.toIdx);
            return (
              <g key={idx} onClick={() => clickSquare(idx)} style={{ cursor: disabled ? 'default' : 'pointer' }}>
                <rect x={c} y={r} width={1} height={1} fill={isLight ? LIGHT : DARK} />
                {isHoverSq && <rect x={c} y={r} width={1} height={1} fill={ARROW_HOVER} opacity={0.42} />}
                {isLast && <rect x={c} y={r} width={1} height={1} fill="#f4e07a" opacity={0.45} />}
                {isSel && <rect x={c} y={r} width={1} height={1} fill="#7ec4ff" opacity={0.55} />}
                {isCheck && <rect x={c} y={r} width={1} height={1} fill="#ff5d5d" opacity={0.5} />}
                {/* file/rank coordinate ticks in board corners */}
                {c === 0 && (
                  <text x={c + 0.04} y={r + 0.22} fontSize={0.17} fill={isLight ? DARK : LIGHT} opacity={0.8}>
                    {rankIdx + 1}
                  </text>
                )}
                {r === 7 && (
                  <text x={c + 0.78} y={r + 0.96} fontSize={0.17} fill={isLight ? DARK : LIGHT} opacity={0.8}>
                    {String.fromCharCode(97 + fileIdx)}
                  </text>
                )}
                {piece && (
                  <Piece
                    type={piece.type}
                    cx={c + 0.5}
                    cy={r + 0.5}
                    size={0.96}
                    fill={piece.color === 'w' ? '#f7f7f5' : '#1a1a1a'}
                    stroke={piece.color === 'w' ? '#33312c' : '#1a1a1a'}
                    strokeWidth={3.5}
                  />
                )}
                {isTarget && !piece && <circle cx={c + 0.5} cy={r + 0.5} r={0.16} fill="#2c2c2c" opacity={0.35} />}
                {isTarget && piece && (
                  <rect x={c} y={r} width={1} height={1} fill="none" stroke="#2c2c2c" strokeWidth={0.07} opacity={0.5} />
                )}
              </g>
            );
          }),
        )}

        {candidates && candidates.length > 0 && (
          <g className="cand-arrows" pointerEvents="none">
            <defs>
              <marker
                id="cand-arrowhead"
                viewBox="0 0 10 10"
                refX={7}
                refY={5}
                markerWidth={0.4}
                markerHeight={0.4}
                markerUnits="userSpaceOnUse"
                orient="auto"
              >
                <path d="M0,1 L9,5 L0,9 z" fill={ARROW} />
              </marker>
              <marker
                id="cand-arrowhead-hover"
                viewBox="0 0 10 10"
                refX={7}
                refY={5}
                markerWidth={0.46}
                markerHeight={0.46}
                markerUnits="userSpaceOnUse"
                orient="auto"
              >
                <path d="M0,1 L9,5 L0,9 z" fill={ARROW_HOVER} />
              </marker>
            </defs>
            {candidates
              .slice()
              // draw the hovered arrow last so it sits on top of the others
              .sort((a, b) => Number(a.uci === activeHoverUci) - Number(b.uci === activeHoverUci))
              .map((cand) => {
              const top = candidates[0].prob || 1;
              const rel = cand.prob / top; // 1 = best move, smaller = less likely
              const hovered = cand.uci === activeHoverUci;
              const from = idxToCell(cand.fromIdx);
              const to = idxToCell(cand.toIdx);
              const x1c = from.c + 0.5;
              const y1c = from.r + 0.5;
              const x2c = to.c + 0.5;
              const y2c = to.r + 0.5;
              const len = Math.hypot(x2c - x1c, y2c - y1c) || 1;
              const ux = (x2c - x1c) / len;
              const uy = (y2c - y1c) / len;
              return (
                <line
                  key={cand.uci}
                  x1={x1c + ux * 0.3}
                  y1={y1c + uy * 0.3}
                  x2={x2c - ux * 0.34}
                  y2={y2c - uy * 0.34}
                  stroke={hovered ? ARROW_HOVER : ARROW}
                  strokeWidth={hovered ? 0.16 : 0.05 + 0.09 * rel}
                  strokeLinecap="round"
                  opacity={hovered ? 1 : 0.3 + 0.6 * rel}
                  markerEnd={`url(#cand-arrowhead${hovered ? '-hover' : ''})`}
                />
              );
            })}
          </g>
        )}

        {searchChildren && searchChildren.length > 0 && (
          <g className="search-arrows" pointerEvents="none">
            <defs>
              <marker
                id="search-arrowhead"
                viewBox="0 0 10 10"
                refX={7}
                refY={5}
                markerWidth={0.4}
                markerHeight={0.4}
                markerUnits="userSpaceOnUse"
                orient="auto"
              >
                <path d="M0,1 L9,5 L0,9 z" fill={SEARCH_ARROW} />
              </marker>
              <marker
                id="search-arrowhead-hover"
                viewBox="0 0 10 10"
                refX={7}
                refY={5}
                markerWidth={0.46}
                markerHeight={0.46}
                markerUnits="userSpaceOnUse"
                orient="auto"
              >
                <path d="M0,1 L9,5 L0,9 z" fill={SEARCH_ARROW_HOVER} />
              </marker>
            </defs>
            {searchChildren
              .slice(0, MAX_SEARCH_ARROWS)
              .slice()
              // draw the hovered (and most-visited) arrow last → on top
              .sort((a, b) => Number(a.uci === searchHoverUci) - Number(b.uci === searchHoverUci) || a.n - b.n)
              .map((c) => {
                const topN = searchChildren[0].n || 1;
                const rel = c.n / topN; // 1 = most-visited (the chosen move)
                const hovered = c.uci === searchHoverUci;
                const from = idxToCell(c.fromIdx);
                const to = idxToCell(c.toIdx);
                const x1c = from.c + 0.5;
                const y1c = from.r + 0.5;
                const x2c = to.c + 0.5;
                const y2c = to.r + 0.5;
                const len = Math.hypot(x2c - x1c, y2c - y1c) || 1;
                const ux = (x2c - x1c) / len;
                const uy = (y2c - y1c) / len;
                return (
                  <line
                    key={c.uci}
                    x1={x1c + ux * 0.3}
                    y1={y1c + uy * 0.3}
                    x2={x2c - ux * 0.34}
                    y2={y2c - uy * 0.34}
                    stroke={hovered ? SEARCH_ARROW_HOVER : SEARCH_ARROW}
                    strokeWidth={hovered ? 0.16 : 0.04 + 0.12 * rel}
                    strokeLinecap="round"
                    opacity={hovered ? 1 : 0.28 + 0.6 * rel}
                    markerEnd={`url(#search-arrowhead${hovered ? '-hover' : ''})`}
                  />
                );
              })}
          </g>
        )}
      </svg>

      {pendingPromo && (
        <div className="promo-overlay" role="dialog" aria-label="Choose promotion piece">
          <span className="promo-label">Promote to:</span>
          {(['q', 'r', 'b', 'n'] as PromotionPiece[]).map((p) => (
            <button key={p} className="promo-btn" onClick={() => choosePromo(p)}>
              <PieceGlyph color={state.humanColor} type={p as PieceType} />
            </button>
          ))}
          <button className="promo-cancel" onClick={() => setPendingPromo(null)}>
            ✕
          </button>
        </div>
      )}
    </div>
  );
}
