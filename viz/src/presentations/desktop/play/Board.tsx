// SVG chessboard — click-to-move, dependency-free (Unicode glyphs). Orientation
// follows humanColor (human side at the bottom). The model only ranks among
// chess.js's legal moves, so all legality/turn gating comes from the store.

import { useMemo, useState } from 'react';
import type { Color, GameState, GameStore, PieceType, PromotionPiece, RootChildStat } from '../../../core/index.ts';
import { Piece, PieceGlyph } from './pieces.tsx';
import { CANDIDATE_ARROW_STYLE, MoveArrows, SEARCH_ARROW_STYLE, type ArrowSpec } from './MoveArrows.tsx';

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
const ARROW_HOVER = '#2ecc71'; // square-tint accent under a hovered/selected candidate (green = "this is the move")
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

  // The blue prob-weighted arrows serve three purposes, in priority order:
  //  • move-assistant SUGGESTIONS — on the HUMAN's turn (assist mode), the model's
  //    top moves for the human's own side, drawn as hints (the human still moves).
  //  • flashMove — after MCTS / auto-play chooses, the picked move flashes (a
  //    one-element "hovered" list) with the same arrow + square tint before it lands.
  //  • previewMoves — non-MCTS auto-play first shows its top policy distribution.
  // Suggestions only appear on the human's turn; flash/preview only on the model's,
  // so these are mutually exclusive in time.
  const suggestions = state.turn === state.humanColor ? state.suggestions : null;
  const candidates = suggestions ?? (state.flashMove ? [state.flashMove] : state.previewMoves);
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
                  <Piece type={piece.type} color={piece.color} cx={c + 0.5} cy={r + 0.5} size={0.95} />
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
          <MoveArrows
            className="cand-arrows"
            idPrefix="cand"
            style={CANDIDATE_ARROW_STYLE}
            arrows={candidates.map((cand): ArrowSpec => ({
              key: cand.uci,
              from: idxToCell(cand.fromIdx),
              to: idxToCell(cand.toIdx),
              rel: cand.prob / (candidates[0].prob || 1), // 1 = best move
              hovered: cand.uci === activeHoverUci,
            }))}
          />
        )}

        {searchChildren && searchChildren.length > 0 && (
          <MoveArrows
            className="search-arrows"
            idPrefix="search"
            style={SEARCH_ARROW_STYLE}
            arrows={searchChildren.slice(0, MAX_SEARCH_ARROWS).map((c): ArrowSpec => ({
              key: c.uci,
              from: idxToCell(c.fromIdx),
              to: idxToCell(c.toIdx),
              rel: c.n / (searchChildren[0].n || 1), // 1 = most-visited (chosen)
              hovered: c.uci === searchHoverUci,
            }))}
          />
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
