// Heatmap — a hand-rolled <canvas> matrix renderer (no charting lib). Paints one
// cell per matrix entry with the chosen colormap and reports hover cells (so a
// caller can reach a single scalar). Crisp at any DPR.

import { useEffect, useMemo, useRef } from 'react';
import { autoMode, diverging, rangeOf, rgbCss, sequential, type RGB } from './colormap.ts';

export interface Cell {
  r: number;
  c: number;
  v: number;
}

export interface HeatmapProps {
  data: ArrayLike<number>;
  rows: number;
  cols: number;
  mode?: 'diverging' | 'sequential';
  /** Actual [lo, hi] color range (else computed from data). Diverging is a
   *  two-slope scale over this range with 0 fixed at the neutral ZERO color. */
  range?: [number, number];
  /** Target on-screen size of the longer axis, in px (default 280). */
  sizePx?: number;
  /** Reserve a gutter outside the grid for an arrow that points at one row/col
   *  (e.g. the board square a companion grid is hovering). 'col' = top gutter,
   *  'row' = left gutter. The gutter is reserved whenever this is set, so the
   *  grid never shifts when the arrow appears/disappears. */
  arrowAxis?: 'row' | 'col';
  /** Which row/col index the arrow points at (null = none drawn this frame). */
  arrowIndex?: number | null;
  onHover?: (cell: Cell | null) => void;
  /** Flip the row axis so row 0 paints at the bottom (board-style). */
  flipY?: boolean;
  ariaLabel?: string;
}

function colorFor(v: number, mode: 'diverging' | 'sequential', range: [number, number]): RGB {
  return mode === 'diverging' ? diverging(v, range[0], range[1]) : sequential(v, range[0], range[1]);
}

/** px reserved outside the grid for the row/col arrow (also used by companion
 *  panels that want to top-align with the grid region, not the gutter). */
export const HEATMAP_GUTTER = 14;
const GUTTER = HEATMAP_GUTTER;

export function Heatmap(props: HeatmapProps) {
  const { data, rows, cols, arrowAxis, arrowIndex, onHover, flipY = false, ariaLabel } = props;
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const mode = props.mode ?? autoMode(data);
  const range = props.range ?? rangeOf(data);
  const sizePx = props.sizePx ?? 280;

  const cell = Math.max(3, Math.floor(sizePx / Math.max(rows, cols)));
  // Gutter is reserved on the relevant edge so the grid never shifts on hover.
  const padTop = arrowAxis === 'col' ? GUTTER : 0;
  const padLeft = arrowAxis === 'row' ? GUTTER : 0;
  const gridW = cols * cell;
  const gridH = rows * cell;
  const wPx = gridW + padLeft;
  const hPx = gridH + padTop;

  // Row index → painted y-row (top origin). flipY puts r=0 at the bottom.
  const paintRow = useMemo(() => (r: number) => (flipY ? rows - 1 - r : r), [flipY, rows]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = (typeof window !== 'undefined' && window.devicePixelRatio) || 1;
    canvas.width = Math.round(wPx * dpr);
    canvas.height = Math.round(hPx * dpr);
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, wPx, hPx);
    for (let r = 0; r < rows; r++) {
      const py = padTop + paintRow(r) * cell;
      for (let c = 0; c < cols; c++) {
        const v = data[r * cols + c];
        ctx.fillStyle = rgbCss(colorFor(v, mode, range));
        ctx.fillRect(padLeft + c * cell, py, cell, cell);
      }
    }
    // Grid lines when cells are big enough to read.
    if (cell >= 12) {
      ctx.strokeStyle = 'rgba(0,0,0,0.12)';
      ctx.lineWidth = 1;
      for (let r = 0; r <= rows; r++) {
        ctx.beginPath();
        ctx.moveTo(padLeft, padTop + r * cell + 0.5);
        ctx.lineTo(padLeft + gridW, padTop + r * cell + 0.5);
        ctx.stroke();
      }
      for (let c = 0; c <= cols; c++) {
        ctx.beginPath();
        ctx.moveTo(padLeft + c * cell + 0.5, padTop);
        ctx.lineTo(padLeft + c * cell + 0.5, padTop + gridH);
        ctx.stroke();
      }
    }
    // Arrow in the reserved gutter, pointing at a whole row/col from outside the
    // grid (so a thin line isn't swamped by an on-cell outline).
    if (arrowAxis && arrowIndex != null && arrowIndex >= 0 && arrowIndex < (arrowAxis === 'col' ? cols : rows)) {
      ctx.fillStyle = '#ffd166';
      const half = 5;
      if (arrowAxis === 'col') {
        const xc = Math.min(wPx - half - 1, Math.max(padLeft + half + 1, padLeft + arrowIndex * cell + cell / 2));
        ctx.beginPath();
        ctx.moveTo(xc - half, 2);
        ctx.lineTo(xc + half, 2);
        ctx.lineTo(xc, padTop - 2);
        ctx.closePath();
        ctx.fill();
      } else {
        const yc = Math.min(hPx - half - 1, Math.max(padTop + half + 1, padTop + paintRow(arrowIndex) * cell + cell / 2));
        ctx.beginPath();
        ctx.moveTo(2, yc - half);
        ctx.lineTo(2, yc + half);
        ctx.lineTo(padLeft - 2, yc);
        ctx.closePath();
        ctx.fill();
      }
    }
  }, [data, rows, cols, cell, wPx, hPx, padTop, padLeft, gridW, gridH, mode, range, arrowAxis, arrowIndex, paintRow]);

  function cellAt(e: React.MouseEvent): Cell | null {
    const c = Math.floor((e.nativeEvent.offsetX - padLeft) / cell);
    const pr = Math.floor((e.nativeEvent.offsetY - padTop) / cell);
    if (c < 0 || c >= cols || pr < 0 || pr >= rows) return null;
    const r = flipY ? rows - 1 - pr : pr;
    return { r, c, v: data[r * cols + c] };
  }

  return (
    <canvas
      ref={canvasRef}
      role="img"
      aria-label={ariaLabel ?? `${rows}×${cols} heatmap`}
      style={{ width: wPx, height: hPx, display: 'block', imageRendering: 'pixelated', borderRadius: 4 }}
      onMouseMove={onHover ? (e) => onHover(cellAt(e)) : undefined}
      onMouseLeave={onHover ? () => onHover(null) : undefined}
    />
  );
}

export interface LegendProps {
  mode: 'diverging' | 'sequential';
  range?: [number, number];
}

/** A compact horizontal scale-bar matching a heatmap's colormap, labelled with
 *  the actual [lo, hi] range it spans. */
export function Legend({ mode, range = [0, 1] }: LegendProps) {
  const [lo, hi] = range;
  const stops = 24;
  const cells = [];
  for (let i = 0; i < stops; i++) {
    const val = lo + (i / (stops - 1)) * (hi - lo);
    const v: RGB = mode === 'diverging' ? diverging(val, lo, hi) : sequential(val, lo, hi);
    cells.push(<span key={i} style={{ flex: 1, background: rgbCss(v) }} />);
  }
  return (
    <div className="legend">
      <span className="legend-num">{lo.toFixed(2)}</span>
      <span className="legend-bar">{cells}</span>
      <span className="legend-num">{hi.toFixed(2)}</span>
    </div>
  );
}
