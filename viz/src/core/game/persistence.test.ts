import { describe, it, expect } from 'vitest';
import { parsePersisted, type RestoreDefaults } from './persistence.ts';

const D: RestoreDefaults = {
  humanColor: 'w',
  assist: false,
  variety: 0.5,
  mcts: { enabled: false, sims: 100, cPuct: 1.5, cutoffThreshold: 0.7 },
};

describe('parsePersisted (forward/backward compatibility contract)', () => {
  it('round-trips a full, current-schema blob', () => {
    const blob = {
      schema: 1,
      startFen: null,
      moves: [{ from: 'e2', to: 'e4' }, { from: 'e7', to: 'e8', promotion: 'q' }],
      humanColor: 'b',
      assist: true,
      variety: 0.25,
      mcts: { enabled: true, sims: 200, cPuct: 2.0, cutoffThreshold: 0.8 },
    };
    expect(parsePersisted(blob, D)).toEqual({
      startFen: null,
      moves: [{ from: 'e2', to: 'e4' }, { from: 'e7', to: 'e8', promotion: 'q' }],
      humanColor: 'b',
      assist: true,
      variety: 0.25,
      mcts: { enabled: true, sims: 200, cPuct: 2.0, cutoffThreshold: 0.8 },
    });
  });

  it('fills DEFAULTS for missing optional settings (older blob)', () => {
    // Only the CORE present — as an older version might have written.
    const r = parsePersisted({ moves: [{ from: 'd2', to: 'd4' }] }, D);
    expect(r).not.toBeNull();
    expect(r!.startFen).toBeNull();
    expect(r!.humanColor).toBe(D.humanColor);
    expect(r!.assist).toBe(D.assist);
    expect(r!.variety).toBe(D.variety);
    expect(r!.mcts).toEqual(D.mcts);
  });

  it('ignores unknown/extra fields (newer blob, same schema)', () => {
    const r = parsePersisted(
      { schema: 1, moves: [], futureFeature: { foo: 1 }, mcts: { sims: 150, somethingNew: true } },
      D,
    );
    expect(r).not.toBeNull();
    expect(r!.mcts.sims).toBe(150);
    expect(r!.mcts.cPuct).toBe(D.mcts.cPuct); // missing → default
    expect('futureFeature' in (r as object)).toBe(false);
  });

  it('discards a blob from a FUTURE breaking schema (returns null → fresh game)', () => {
    expect(parsePersisted({ schema: 2, moves: [] }, D)).toBeNull();
    expect(parsePersisted({ schema: 999, moves: [{ from: 'e2', to: 'e4' }] }, D)).toBeNull();
  });

  it('rejects malformed CORE (returns null, never throws)', () => {
    expect(parsePersisted(null, D)).toBeNull();
    expect(parsePersisted(42, D)).toBeNull();
    expect(parsePersisted({}, D)).toBeNull(); // no moves[]
    expect(parsePersisted({ moves: 'nope' }, D)).toBeNull();
    expect(parsePersisted({ moves: [{ from: 'e2' }] }, D)).toBeNull(); // move missing `to`
    expect(parsePersisted({ moves: [null] }, D)).toBeNull();
  });

  it('coerces an invalid humanColor / non-finite variety to defaults', () => {
    const r = parsePersisted({ moves: [], humanColor: 'x', variety: NaN }, D);
    expect(r!.humanColor).toBe(D.humanColor);
    expect(r!.variety).toBe(D.variety);
  });

  it('preserves a custom start FEN', () => {
    const fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
    const r = parsePersisted({ moves: [], startFen: fen }, D);
    expect(r!.startFen).toBe(fen);
  });
});
