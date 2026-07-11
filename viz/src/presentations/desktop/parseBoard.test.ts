// parseBoard tests — FEN placement → 64-array in python-chess square indexing
// (a1=0 .. h8=63), shared by every board-drawing component.

import { describe, it, expect } from 'vitest';
import { parseBoard } from './parseBoard.ts';

const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

describe('parseBoard', () => {
  it('maps the start position into python-chess square indices', () => {
    const b = parseBoard(START_FEN);
    expect(b[0]).toEqual({ color: 'w', type: 'r' }); // a1
    expect(b[4]).toEqual({ color: 'w', type: 'k' }); // e1
    expect(b[12]).toEqual({ color: 'w', type: 'p' }); // e2
    expect(b[36]).toBeNull(); // e5 empty
    expect(b[60]).toEqual({ color: 'b', type: 'k' }); // e8
    expect(b[63]).toEqual({ color: 'b', type: 'r' }); // h8
    expect(b.filter(Boolean)).toHaveLength(32);
  });

  it('skips runs of empty squares by digit', () => {
    // Lone black king on d5, white king on a1.
    const b = parseBoard('8/8/8/3k4/8/8/8/K7 w - - 0 1');
    expect(b[35]).toEqual({ color: 'b', type: 'k' }); // d5
    expect(b[0]).toEqual({ color: 'w', type: 'k' }); // a1
    expect(b.filter(Boolean)).toHaveLength(2);
  });
});
