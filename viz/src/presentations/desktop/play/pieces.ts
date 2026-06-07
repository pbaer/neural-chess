// Unicode chess glyphs — self-contained, no image assets. Keyed by color+type.
import type { Color, PieceType } from '../../../core/index.ts';

// Both colors use the SOLID (filled) glyph set; piece color is conveyed by the
// SVG fill (white = light fill + dark outline, black = dark fill). The outline
// white glyphs (♔♕…) render hollow against a light board, so we avoid them.
const GLYPH: Record<Color, Record<PieceType, string>> = {
  w: { k: '♚', q: '♛', r: '♜', b: '♝', n: '♞', p: '♟' },
  b: { k: '♚', q: '♛', r: '♜', b: '♝', n: '♞', p: '♟' },
};

export function pieceGlyph(color: Color, type: PieceType): string {
  return GLYPH[color][type];
}
