// StageDetail — drilling into one stage (the component → heatmap → scalar levels).
// Version-neutral: it switches on the node KIND and the trace-field naming from
// the core traceIndex, never on a specific architecture. It shows the stage's
// explanation card, a flow of its real intermediates (activations) + its weights,
// and renders whichever field is selected as a heatmap whose cells reach a single
// scalar. Attention fields get the board-tied AttentionBoard.

import { useEffect, useMemo, useState } from 'react';
import {
  traceFieldsFor,
  traceStore,
  type Capsule,
  type Color,
  type GraphNode,
  type PieceType,
  type TraceField,
} from '../../../../core/index.ts';
import { Heatmap, HEATMAP_GUTTER, Legend } from '../render/Heatmap.tsx';
import { colorScaleOf, diverging, niceSeq, rangeOf, rgbCss, sequential } from '../render/colormap.ts';
import { ContentCard } from '../ContentCard.tsx';
import { AttentionBoard } from './AttentionBoard.tsx';
import { Piece } from '../../play/pieces.tsx';
import { ScalarInspector, type ScalarView } from '../scalar/ScalarInspector.tsx';

const PLANE_NAMES = [
  'own pawns', 'own rooks', 'own knights', 'own bishops', 'own queens', 'own king',
  'opp pawns', 'opp rooks', 'opp knights', 'opp bishops', 'opp queens', 'opp king',
  'own O-O', 'own O-O-O', 'opp O-O', 'opp O-O-O',
  'en passant', 'halfmove/100', 'fullmove/100', 'side to move', 'repetition',
];

interface PieceCell {
  color: Color;
  type: PieceType;
}

/** Parse a FEN placement field into a 64-array indexed by python-chess square. */
function parseBoard(fen: string): (PieceCell | null)[] {
  const arr: (PieceCell | null)[] = new Array(64).fill(null);
  const rows = (fen.split(' ')[0] ?? '').split('/');
  for (let r = 0; r < 8; r++) {
    const rankIdx = 7 - r;
    let file = 0;
    for (const ch of rows[r] ?? '') {
      if (ch >= '1' && ch <= '8') file += ch.charCodeAt(0) - 48;
      else {
        arr[rankIdx * 8 + file] = { color: ch === ch.toUpperCase() ? 'w' : 'b', type: ch.toLowerCase() as PieceType };
        file++;
      }
    }
  }
  return arr;
}

/** Piece glyphs for an 8×8 board (viewBox 0 0 8 8), network frame: side to move at
 *  the bottom. A contrasting halo (paint-order stroke) keeps both colors legible
 *  over any heatmap cell. Non-interactive overlay. */
function BoardGlyphs({ board, turn }: { board: (PieceCell | null)[]; turn: Color }) {
  const rows = [0, 1, 2, 3, 4, 5, 6, 7];
  return (
    <g style={{ pointerEvents: 'none' }}>
      {rows.map((dr) =>
        rows.map((f) => {
          const s = (7 - dr) * 8 + f; // engine-frame square (rank 0 at the bottom)
          const piece = board[turn === 'w' ? s : s ^ 56];
          if (!piece) return null;
          return <Piece key={s} type={piece.type} color={piece.color} cx={f + 0.5} cy={dr + 0.5} size={0.92} halo />;
        }),
      )}
    </g>
  );
}

type FieldSel =
  | { kind: 'act'; key: string; label: string; opKind?: string }
  | { kind: 'weight'; name: string; label: string };

export interface StageDetailProps {
  node: GraphNode;
  capsule: Capsule;
  /** traceStore version — forces a re-read of activations when the trace updates. */
  traceVersion: number;
}

function squareLabel(s: number, turn: Color): string {
  const real = turn === 'w' ? s : s ^ 56;
  return String.fromCharCode(97 + (real & 7)) + String.fromCharCode(49 + (real >> 3));
}

function shortWeightLabel(name: string): string {
  const parts = name.split('.');
  // Drop a leading "blocks.<i>." so the chip reads "attn.qkv.weight".
  if (parts[0] === 'blocks') return parts.slice(2).join('.');
  return name;
}

/** Chip label: short name without a trailing ".weight" (the bias rides along). */
function chipLabel(name: string): string {
  const s = shortWeightLabel(name);
  return s.endsWith('.weight') ? s.slice(0, -7) : s;
}

/** Sibling bias tensor name for a "*.weight", or null. */
function biasNameFor(weightName: string): string | null {
  return weightName.endsWith('.weight') ? weightName.slice(0, -7) + '.bias' : null;
}

/** Map a weight tensor name to the op-kind whose explanation card best fits it. */
function weightOpKind(name: string, fallback: string): string {
  if (name.includes('rel_bias') || name.includes('qkv') || name.includes('attn.proj')) return 'attention';
  if (name.includes('ffn')) return 'ffn';
  if (name.includes('ln') || name.includes('norm')) return 'layernorm';
  if (name.includes('pos_emb')) return 'tokenize';
  if (name.includes('policy')) return 'policy_head';
  if (name.includes('value')) return 'value_head';
  return fallback;
}

export function StageDetail({ node, capsule, traceVersion }: StageDetailProps) {
  // This stage owns the scalar it most recently surfaced, shown in its own rail
  // beside the description — so the hovered number stays tied to this block.
  const [scalar, setScalar] = useState<ScalarView | null>(null);
  const snap = traceStore.snapshot();
  const meta = snap?.meta ?? null;

  // Available activation fields = those recorded in the current trace.
  const actFields: TraceField[] = useMemo(() => {
    void traceVersion;
    return traceFieldsFor(node).filter((f) => traceStore.has(f.key));
  }, [node, traceVersion]);

  const weightFields = node.weightNames;
  // A bias is shown alongside its weight, so it gets no chip of its own.
  const primaryWeights = useMemo(
    () =>
      weightFields.filter((w) => {
        if (w.endsWith('.bias') && weightFields.includes(w.slice(0, -5) + '.weight')) return false;
        return true;
      }),
    [weightFields],
  );

  // Default selection: prefer attention probs for a block (the marquee), else the
  // node's first available activation, else its first weight.
  const defaultSel: FieldSel | null = useMemo(() => {
    const probs = actFields.find((f) => f.key.endsWith('.attn.probs'));
    if (probs) return { kind: 'act', key: probs.key, label: probs.label, opKind: probs.opKind };
    if (actFields[0]) return { kind: 'act', key: actFields[0].key, label: actFields[0].label, opKind: actFields[0].opKind };
    if (primaryWeights[0]) return { kind: 'weight', name: primaryWeights[0], label: chipLabel(primaryWeights[0]) };
    return null;
  }, [actFields, primaryWeights]);

  const [sel, setSel] = useState<FieldSel | null>(defaultSel);
  const active = sel ?? defaultSel;

  const select = (s: FieldSel) => {
    setSel(s);
    setScalar(null); // a new field's scalar context starts empty
  };

  const dims = `${node.label} · ${node.kind} · ${node.params.toLocaleString()} params`;
  const opKindForCard =
    active == null
      ? node.kind
      : active.kind === 'act'
        ? active.opKind ?? node.kind
        : weightOpKind(active.name, node.kind);

  return (
    <div className="stage-grid">
      <div className="stage-left">
        {actFields.length > 0 && (
          <div className="field-chips">
            <span className="field-group-label">activations</span>
            {actFields.map((f) => (
              <button
                key={f.key}
                className={'chip' + (active?.kind === 'act' && active.key === f.key ? ' chip-active' : '')}
                onClick={() => select({ kind: 'act', key: f.key, label: f.label, opKind: f.opKind })}
                title={f.blurb}
              >
                {f.label}
              </button>
            ))}
          </div>
        )}
        {primaryWeights.length > 0 && (
          <div className="field-chips">
            <span className="field-group-label">weights</span>
            {primaryWeights.map((w) => (
              <button
                key={w}
                className={'chip chip-weight' + (active?.kind === 'weight' && active.name === w ? ' chip-active' : '')}
                onClick={() => select({ kind: 'weight', name: w, label: chipLabel(w) })}
                title={w}
              >
                {chipLabel(w)}
              </button>
            ))}
          </div>
        )}

        <div className="field-view">
          {active ? (
            <FieldView node={node} capsule={capsule} sel={active} meta={meta} onScalar={setScalar} />
          ) : (
            <div className="scalar-empty">No data for this stage in the current trace.</div>
          )}
        </div>
      </div>

      <aside className="stage-rail">
        <ContentCard kind={opKindForCard} dims={dims} />
        <ScalarInspector view={scalar} />
      </aside>
    </div>
  );
}

// The value head is a single scalar; surface it into the rail via an effect
// (never call onScalar during render — that updates a parent mid-render).
function ValueView({ value, onScalar }: { value: number; onScalar: (v: ScalarView | null) => void }) {
  useEffect(() => {
    onScalar({
      name: 'value',
      value,
      description: 'Position evaluation after tanh, from the side-to-move’s perspective: +1 winning, −1 losing, 0 equal.',
    });
    return () => onScalar(null);
  }, [value, onScalar]);
  return <div className="value-scalar">value = {value.toFixed(6)}</div>;
}

interface FieldViewProps {
  node: GraphNode;
  capsule: Capsule;
  sel: FieldSel;
  meta: { fen: string; turn: Color } | null;
  onScalar: (v: ScalarView | null) => void;
}

function FieldView({ node, capsule, sel, meta, onScalar }: FieldViewProps) {
  if (sel.kind === 'act') {
    const entry = traceStore.entry(sel.key);
    if (!entry) return <div className="scalar-empty">Run a position to populate this activation.</div>;

    // Attention (board-tied) view.
    if (sel.key.endsWith('.attn.probs') || sel.key.endsWith('.attn.scores')) {
      const probs = traceStore.entry(`${node.id}.attn.probs`)?.data;
      const scores = traceStore.entry(`${node.id}.attn.scores`)?.data;
      const relName = node.weightNames.find((n) => n.endsWith('attn.rel_bias'));
      const relBias = relName ? capsule.data(relName) : undefined;
      const heads = Number(node.dims.heads);
      if (probs && meta) {
        return (
          <AttentionBoard
            probs={probs}
            scores={scores}
            relBias={relBias}
            heads={heads}
            fen={meta.fen}
            turn={meta.turn}
            onScalar={onScalar}
          />
        );
      }
    }

    if (sel.key === 'planes') return <PlanesView data={entry.data} turn={meta?.turn ?? 'w'} fen={meta?.fen} onScalar={onScalar} />;
    if (sel.key === 'value') {
      return <ValueView value={entry.data[0]} onScalar={onScalar} />;
    }
    if (sel.key === 'policy_logits') {
      return <MatrixView data={entry.data} rows={64} cols={73} label="policy logits" rowKind="square" turn={meta?.turn ?? 'w'} fen={meta?.fen} colName="move-type" onScalar={onScalar} />;
    }
    // Conv-style (C × 8 × 8) activation, e.g. the per-square embed: treat each
    // of the 64 squares as a row and its C channels as the columns (board frame).
    if (entry.shape.length === 3 && entry.shape[1] * entry.shape[2] === 64) {
      return <ConvActView data={entry.data} channels={entry.shape[0]} label={sel.label} turn={meta?.turn ?? 'w'} fen={meta?.fen} onScalar={onScalar} />;
    }
    // Generic (64 × d) activation.
    if (entry.shape.length === 2) {
      return <MatrixView data={entry.data} rows={entry.shape[0]} cols={entry.shape[1]} label={sel.label} rowKind="square" turn={meta?.turn ?? 'w'} fen={meta?.fen} colName="feature" onScalar={onScalar} />;
    }
    return <div className="scalar-empty">Shape {entry.shape.join('×')} — no viewer.</div>;
  }

  // Weight field.
  const t = capsule.tensor(sel.name);
  if (sel.name.endsWith('attn.rel_bias')) {
    return <GeometryBiasView data={t.data} heads={t.shape[0]} onScalar={onScalar} />;
  }
  // The position embedding is per-square (1,64,d) → render in the board frame.
  if (sel.name.includes('pos_emb')) {
    const d = t.shape[t.shape.length - 1];
    return <MatrixView data={t.data} rows={64} cols={d} label={sel.label} rowKind="square" colName="feature" turn={meta?.turn ?? 'w'} fen={meta?.fen} onScalar={onScalar} />;
  }
  // Pair the weight with its bias (rendered as an aligned strip beside it).
  const biasName = biasNameFor(sel.name);
  const bias = biasName && node.weightNames.includes(biasName) ? capsule.tensor(biasName).data : null;
  // Collapse trailing singleton conv dims (out,in,1,1) → (out,in).
  const dims2 = t.shape.filter((d) => d > 1);
  if (dims2.length === 2) {
    return <MatrixView data={t.data} rows={dims2[0]} cols={dims2[1]} label={sel.label} rowKind="out" colName="in" bias={bias} onScalar={onScalar} />;
  }
  if (dims2.length <= 1) {
    const n = t.data.length;
    return <MatrixView data={t.data} rows={1} cols={n} label={sel.label} rowKind="out" colName="index" bias={bias} onScalar={onScalar} />;
  }
  return <div className="scalar-empty">Shape {t.shape.join('×')} — no viewer.</div>;
}

// ---- a conv-style (C × 8 × 8) activation rendered as a 64 × C square-matrix ----
// Transpose channel-major (c*64+s) → row-major square tokens (s*C+c) once, then
// reuse the generic square matrix (which carries its own board companion).

function ConvActView({ data, channels, label, turn, fen, onScalar }: { data: Float32Array; channels: number; label: string; turn: Color; fen?: string; onScalar: (v: ScalarView | null) => void }) {
  const transposed = useMemo(() => {
    const t = new Float32Array(64 * channels);
    for (let c = 0; c < channels; c++) for (let s = 0; s < 64; s++) t[s * channels + c] = data[c * 64 + s];
    return t;
  }, [data, channels]);
  return <MatrixView data={transposed} rows={64} cols={channels} label={label} rowKind="square" colName="feature" turn={turn} fen={fen} onScalar={onScalar} />;
}

// ---- generic matrix view (heatmap + legend + hover→scalar) ----
// Two layouts off `rowKind`:
//  • 'square' — the 64 board squares. Displayed ROTATED so squares are COLUMNS
//    and the feature vector runs down the ROWS (better use of width). A companion
//    8×8 board grid (same color scale) sits on the left; hovering/selecting a
//    square points a gutter arrow at its column, and vice-versa.
//  • 'out' — a learned weight; its bias (if any) renders as an aligned strip
//    just outside the matching edge, on the same shared color scale.

const MAX_PX = 360;
const cellPxFor = (rows: number, cols: number) => Math.max(3, Math.floor(Math.min(MAX_PX, Math.max(rows, cols) * 8) / Math.max(rows, cols)));

interface MatrixViewProps {
  data: Float32Array;
  rows: number;
  cols: number;
  label: string;
  rowKind: 'square' | 'out';
  colName: string;
  turn?: Color;
  /** Current position FEN — draws the live pieces on the companion board grid. */
  fen?: string;
  /** Bias vector to render aligned with the matrix (length must match a side). */
  bias?: Float32Array | null;
  onScalar: (v: ScalarView | null) => void;
}

function MatrixView({ data, rows, cols, label, rowKind, colName, turn = 'w', fen, bias = null, onScalar }: MatrixViewProps) {
  const showBoard = rowKind === 'square';
  const board = useMemo(() => (showBoard && fen ? parseBoard(fen) : null), [showBoard, fen]);

  // Shared color scale over the matrix AND its bias, so every panel that sits
  // side-by-side (board grid, heatmap, bias strip) reads on one scale.
  const scaleData = useMemo(() => {
    if (!bias) return data;
    const m = new Float32Array(data.length + bias.length);
    m.set(data);
    m.set(bias, data.length);
    return m;
  }, [data, bias]);
  // Color scale spans exactly the actual values present (across matrix ∪ bias).
  const { mode, range } = useMemo(() => colorScaleOf(scaleData), [scaleData]);

  // Per-square average (board frame), and the rotated display matrix.
  const avg = useMemo(() => {
    if (!showBoard) return null;
    const a = new Float32Array(64);
    for (let s = 0; s < 64; s++) {
      let sum = 0;
      for (let c = 0; c < cols; c++) sum += data[s * cols + c];
      a[s] = sum / cols;
    }
    return a;
  }, [data, cols, showBoard]);
  const disp = useMemo(() => {
    if (!showBoard) return data;
    const t = new Float32Array(cols * 64); // [feature, square]
    for (let s = 0; s < 64; s++) for (let c = 0; c < cols; c++) t[c * 64 + s] = data[s * cols + c];
    return t;
  }, [data, cols, showBoard]);

  // pinned = a clicked square (sticky); hoverSq = transient board/heatmap hover.
  const [pinned, setPinned] = useState<number | null>(null);
  const [hoverSq, setHoverSq] = useState<number | null>(null);
  const activeSq = hoverSq ?? pinned;

  if (showBoard) {
    const heatRows = cols; // feature index
    const heatCols = 64; // board square
    // The heatmap's grid region is heatRows·cell tall; size the companion board
    // to exactly that so the two line up (top-aligned below the arrow gutter).
    const hc = cellPxFor(heatRows, heatCols);
    const gridH = heatRows * hc;
    const squareScalar = (s: number): ScalarView => ({
      name: `${label}[${squareLabel(s, turn)}] · avg`,
      value: avg ? avg[s] : 0,
      description: `Mean of the ${cols} ${colName}s at square ${squareLabel(s, turn)} (network frame).`,
    });
    return (
      <div className="matrix-view">
        <div className="matrix-with-board">
          {avg && (
            <BoardSquares
              avg={avg}
              board={board}
              turn={turn}
              sizePx={gridH}
              gutter={HEATMAP_GUTTER}
              mode={mode}
              range={range}
              highlight={activeSq}
              onHoverSquare={(s) => {
                setHoverSq(s);
                onScalar(s == null ? null : squareScalar(s));
              }}
              onPick={(s) => setPinned((p) => (p === s ? null : s))}
            />
          )}
          <div className="matrix-heat">
            <Heatmap
              data={disp}
              rows={heatRows}
              cols={heatCols}
              mode={mode}
              range={range}
              sizePx={Math.min(MAX_PX, Math.max(heatRows, heatCols) * 8)}
              arrowAxis="col"
              arrowIndex={activeSq}
              ariaLabel={`${label} ${heatRows}×${heatCols}`}
              onHover={(cell) => {
                setHoverSq(cell ? cell.c : null);
                onScalar(
                  cell
                    ? {
                        name: `${label}[${squareLabel(cell.c, turn)}][${colName} ${cell.r}]`,
                        value: cell.v,
                        description: `Square ${squareLabel(cell.c, turn)}, ${colName} ${cell.r}.`,
                      }
                    : null,
                );
              }}
            />
            <Legend mode={mode} range={range} />
          </div>
        </div>
        <p className="matrix-caption">
          Columns = 64 board squares (network frame, side to move at the bottom); rows = {cols} {colName}s. Hover the
          board or a heatmap cell.
        </p>
      </div>
    );
  }

  // Weight matrix (+ optional aligned bias strip).
  const biasAxis: 'rows' | 'cols' | null = bias ? (bias.length === rows ? 'rows' : bias.length === cols ? 'cols' : null) : null;
  const wCell = cellPxFor(rows, cols);
  const heat = (
    <Heatmap
      data={data}
      rows={rows}
      cols={cols}
      mode={mode}
      range={range}
      sizePx={Math.min(MAX_PX, Math.max(rows, cols) * 8)}
      ariaLabel={`${label} ${rows}×${cols}`}
      onHover={(cell) =>
        onScalar(
          cell
            ? { name: `${label}[${cell.r}][${colName} ${cell.c}]`, value: cell.v, description: `Output ${cell.r}, ${colName} ${cell.c}.` }
            : null,
        )
      }
    />
  );
  const strip = bias && biasAxis ? (
    <BiasStrip bias={bias} axis={biasAxis} cellPx={wCell} mode={mode} range={range} label={label} onScalar={onScalar} />
  ) : null;

  return (
    <div className="matrix-view">
      <div className={biasAxis === 'rows' ? 'weight-bias-row' : 'weight-bias-col'}>
        {heat}
        {strip}
      </div>
      <Legend mode={mode} range={range} />
      <p className="matrix-caption">
        {rows} {rows === 1 ? 'row' : 'outputs'} × {cols} {colName}s.
        {biasAxis ? ' The separated strip is the bias (one per output).' : ''} Hover a cell for its exact value.
      </p>
    </div>
  );
}

// ---- bias vector rendered as a 1-wide/1-tall strip, aligned to the weight ----

function BiasStrip({ bias, axis, cellPx, mode, range, label, onScalar }: {
  bias: Float32Array;
  axis: 'rows' | 'cols';
  cellPx: number;
  mode: 'diverging' | 'sequential';
  range: [number, number];
  label: string;
  onScalar: (v: ScalarView | null) => void;
}) {
  const n = bias.length;
  // sizePx chosen so the strip's cell size matches the weight's exactly.
  const sizePx = cellPx * n;
  return (
    <div className="bias-strip">
      <Heatmap
        data={bias}
        rows={axis === 'rows' ? n : 1}
        cols={axis === 'rows' ? 1 : n}
        mode={mode}
        range={range}
        sizePx={sizePx}
        ariaLabel={`${label} bias`}
        onHover={(cell) =>
          onScalar(
            cell
              ? { name: `${label} bias[${axis === 'rows' ? cell.r : cell.c}]`, value: cell.v, description: `Bias added to output ${axis === 'rows' ? cell.r : cell.c}.` }
              : null,
          )
        }
      />
      <span className="bias-tag">bias</span>
    </div>
  );
}

// ---- companion 8×8 board: one cell per square, colored by that square's row
//      average on the SAME scale as the heatmap; hovering/selecting drives the
//      heatmap column arrow (network frame, side to move at the bottom). ----

function BoardSquares({ avg, board, turn, sizePx, gutter, mode, range, highlight, onHoverSquare, onPick }: {
  avg: Float32Array;
  board: (PieceCell | null)[] | null;
  turn: Color;
  /** Side length in px — set to the heatmap's grid height so the two line up. */
  sizePx: number;
  /** Top spacer matching the heatmap's arrow gutter, so the grids top-align. */
  gutter: number;
  mode: 'diverging' | 'sequential';
  range: [number, number];
  highlight: number | null;
  onHoverSquare: (s: number | null) => void;
  onPick: (s: number) => void;
}) {
  const rows = [0, 1, 2, 3, 4, 5, 6, 7];
  return (
    <div className="square-grid">
      <div className="square-grid-head" style={{ height: gutter }}>avg / square</div>
      <div className="square-board" style={{ width: sizePx, height: sizePx }} onMouseLeave={() => onHoverSquare(null)}>
        <svg viewBox="0 0 8 8" className="square-svg" role="grid" aria-label="Per-square row average">
          {rows.map((dr) =>
            rows.map((f) => {
              const h = 7 - dr; // engine-frame rank (side to move at the bottom)
              const s = h * 8 + f;
              const v = avg[s];
              const fill = rgbCss(mode === 'diverging' ? diverging(v, range[0], range[1]) : sequential(v, range[0], range[1]));
              return (
                <rect key={s} x={f} y={dr} width={1} height={1} fill={fill} onMouseEnter={() => onHoverSquare(s)} onClick={() => onPick(s)} style={{ cursor: 'pointer' }} />
              );
            }),
          )}
          {board && <BoardGlyphs board={board} turn={turn} />}
        </svg>
        {/* Crisp CSS-bordered outline (even on all sides, never clipped). */}
        {highlight != null && (
          <div className="cell-outline" style={{ left: `${(highlight & 7) * 12.5}%`, top: `${(7 - (highlight >> 3)) * 12.5}%` }} />
        )}
      </div>
    </div>
  );
}

// ---- geometry bias (the principle-clean showpiece): per-head 15×15 (Δrank,Δfile) ----

function GeometryBiasView({ data, heads, onScalar }: { data: Float32Array; heads: number; onScalar: (v: ScalarView | null) => void }) {
  const [head, setHead] = useState(0);
  const grid = useMemo(() => data.subarray(head * 225, head * 225 + 225), [data, head]);
  const { mode, range } = useMemo(() => colorScaleOf(grid), [grid]);
  return (
    <div className="geo-bias">
      <div className="attn-head-row">
        <span className="control-label">Head</span>
        {Array.from({ length: heads }, (_, h) => (
          <button key={h} className={'btn btn-mini' + (head === h ? ' btn-active' : '')} onClick={() => setHead(h)}>
            {h}
          </button>
        ))}
      </div>
      <Heatmap
        data={grid}
        rows={15}
        cols={15}
        mode={mode}
        range={range}
        sizePx={240}
        ariaLabel="Geometry bias 15×15"
        onHover={(cell) =>
          onScalar(
            cell
              ? {
                  name: `geo_bias[h=${head}][Δrank=${cell.r - 7}][Δfile=${cell.c - 7}]`,
                  value: cell.v,
                  description: 'Learned nudge added to attention scores for this relative board offset.',
                }
              : null,
          )
        }
      />
      <Legend mode={mode} range={range} />
      <p className="matrix-caption">
        Rows = Δrank (−7…+7), cols = Δfile (−7…+7); center = same square. The model learned which board offsets matter —
        nothing about chess geometry was injected.
      </p>
    </div>
  );
}

// ---- input planes (21 × 8×8), board-tied ----

function PlanesView({ data, turn, fen, onScalar }: { data: Float32Array; turn: Color; fen?: string; onScalar: (v: ScalarView | null) => void }) {
  const [plane, setPlane] = useState(0);
  const grid = useMemo(() => data.subarray(plane * 64, plane * 64 + 64), [data, plane]);
  // Color scale spans this plane's actual values (binary planes land on [0,1];
  // fractional ones — clocks, repetition — use their own narrower range).
  const range = useMemo(() => niceSeq(...rangeOf(grid)), [grid]);
  const board = useMemo(() => (fen ? parseBoard(fen) : null), [fen]);
  const PX = 240;
  return (
    <div className="planes-view">
      <div className="plane-picker">
        {PLANE_NAMES.map((nm, p) => (
          <button key={p} className={'chip chip-plane' + (plane === p ? ' chip-active' : '')} onClick={() => setPlane(p)} title={`plane ${p}: ${nm}`}>
            {p}
          </button>
        ))}
      </div>
      <div className="plane-name">plane {plane} — {PLANE_NAMES[plane]}</div>
      {/* Heatmap canvas with the live pieces overlaid as a non-interactive SVG. */}
      <div className="plane-board" style={{ width: PX, height: PX }}>
        <Heatmap
          data={grid}
          rows={8}
          cols={8}
          mode="sequential"
          range={range}
          sizePx={PX}
          flipY
          ariaLabel={`Input plane ${plane}`}
          onHover={(cell) =>
            onScalar(
              cell
                ? {
                    name: `plane[${plane}][${squareLabel(cell.r * 8 + cell.c, turn)}]`,
                    value: cell.v,
                    description: `${PLANE_NAMES[plane]} at ${squareLabel(cell.r * 8 + cell.c, turn)} (network frame).`,
                  }
                : null,
            )
          }
        />
        {board && (
          <svg viewBox="0 0 8 8" className="plane-glyphs" aria-hidden>
            <BoardGlyphs board={board} turn={turn} />
          </svg>
        )}
      </div>
      <Legend mode="sequential" range={range} />
      <p className="matrix-caption">8×8, network frame (side to move at the bottom). Hover a square for its value.</p>
    </div>
  );
}
