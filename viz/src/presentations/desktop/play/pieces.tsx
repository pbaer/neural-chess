// Chess pieces as vector SVG silhouettes (single path each) — self-contained,
// no image assets and no fonts. We deliberately do NOT use Unicode chess glyphs
// (♙♟♔ …): iOS Safari emoji-substitutes some of those code points (pawns), which
// ignores the SVG `fill` (a "white" pawn shows the dark emoji), uses different
// metrics (pawns too big, others too small) and mis-baselines them (sit too high).
// Vector paths render identically on every platform.
//
// Each path is authored in a 0..100 box, bbox-centered on (50,50), so a piece
// drops into any square by translating to its centre. Colour is conveyed purely
// by `fill` (white = light fill + dark outline, black = dark fill), with a stroke
// that doubles as a legibility halo over heatmap / attention tints. The original
// matched silhouette set (CC0 / no third-party assets) keeps the public site
// license-clean. Both colours use the same shape; the knight faces left.

import type { Color, PieceType } from '../../../core/index.ts';

export const PIECE_PATHS: Record<PieceType, string> = {
  p: 'M34 76L31 72C31 69 35 68 40 67L44 52C37 49 36 41 39 35C41 28 45 24 50 24C55 24 59 28 61 35C64 41 63 49 56 52L60 67C65 68 69 69 69 72L66 76Z',
  n: 'M79.5 84.5L79.5 79.5C78.5 71.5 75.5 64.5 69.5 57.5C65.5 50.5 64.5 42.5 67.5 34.5L70.5 24.5L65.5 17.5L61.5 25.5L56.5 15.5L54.5 27.5C51.5 31.5 46.5 33.5 40.5 34.5L31.5 36.5C25.5 39.5 21.5 43.5 20.5 48.5L24.5 53.5L31.5 54.5L38.5 52.5C42.5 54.5 45.5 58.5 45.5 63.5C45.5 70.5 43.5 76.5 40.5 79.5L37.5 84.5Z',
  b: 'M26 84L29 80C33 77 34 74 33 70C37 68 40 66 40 60C34 56 32 48 36 41C39 35 44 31 44 25C44 20 47 16 50 16C53 16 56 20 56 25C56 31 61 35 64 41C68 48 66 56 60 60C60 66 63 68 67 70C66 74 67 77 71 80L74 84Z',
  r: 'M24 81L27 77C29 73 30 69 30 63L33 61L33 39L28 35L28 19L34 19L34 25L42 25L42 19L50 19L58 19L58 25L66 25L66 19L72 19L72 35L67 39L67 61L70 63C70 69 71 73 73 77L76 81Z',
  q: 'M26 86L30 82C33 78 33 72 31 66L29 40L24 20L33 32L40 16L46 30L50 14L54 30L60 16L67 32L76 20L71 40L69 66C67 72 67 78 70 82L74 86Z',
  k: 'M26 90L30 86C33 82 33 76 31 70L29 46L27 32L36 40L44 30L44 22L38 22L38 16L44 16L44 10L50 10L56 10L56 16L62 16L62 22L56 22L56 30L64 40L73 32L71 46L69 70C67 76 67 82 70 86L74 90Z',
};

export interface PieceProps {
  type: PieceType;
  /** Centre of the target square, in the SVG's own coordinate units. */
  cx: number;
  cy: number;
  /** Piece extent (a full square ≈ 1 on a 0..8 board). The piece is centred. */
  size: number;
  fill: string;
  stroke: string;
  /** Stroke width in the path's 0..100 space, so it scales with the piece. */
  strokeWidth?: number;
}

/** One vector piece, centred at (cx,cy) and scaled to `size`. Non-interactive. */
export function Piece({ type, cx, cy, size, fill, stroke, strokeWidth = 4 }: PieceProps) {
  const s = size / 100;
  return (
    <path
      d={PIECE_PATHS[type]}
      transform={`translate(${cx - size / 2} ${cy - size / 2}) scale(${s})`}
      fill={fill}
      stroke={stroke}
      strokeWidth={strokeWidth}
      strokeLinejoin="round"
      paintOrder="stroke"
      style={{ pointerEvents: 'none', userSelect: 'none' }}
    />
  );
}

/** Standalone piece glyph for HTML contexts (e.g. promotion buttons): a small
 *  square SVG that fills the font box, coloured by side. */
export function PieceGlyph({ color, type }: { color: Color; type: PieceType }) {
  const white = color === 'w';
  return (
    <svg viewBox="0 0 100 100" width="1em" height="1em" aria-hidden focusable="false" style={{ display: 'block' }}>
      <Piece
        type={type}
        cx={50}
        cy={50}
        size={100}
        fill={white ? '#f3f3f0' : '#1a1a1a'}
        stroke={white ? '#333' : '#cfcfcf'}
        strokeWidth={4}
      />
    </svg>
  );
}
