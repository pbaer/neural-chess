// @vitest-environment jsdom
//
// The value-head readings — the CURRENT gauge value and the HISTORICAL per-move
// readings — must survive a page refresh and an undo (issue #3: they used to go
// blank). This drives the real store against a fake in-thread engine and asserts
// that a freshly-created store (a "reload", reading the persisted localStorage
// blob) restores `valueHistory`, and that undo keeps the surviving readings.
// jsdom gives us a working localStorage; fake timers drive the auto-play beats.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createGameStore } from '../../src/core/game/gameStore.ts';
import { fakeEngine, playFirstLegalHumanMove, playOutHumanFlash, playOutModelMove } from './helpers.ts';

/** The current gauge value = the most recent White-framed reading (or null). */
function currentValue(store: ReturnType<typeof createGameStore>): number | null {
  const h = store.getState().valueHistory;
  return h.length > 0 ? h[h.length - 1].whiteValue : null;
}

describe('value-head readings persist across reload + undo', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    localStorage.clear();
  });
  afterEach(() => {
    vi.useRealTimers();
    localStorage.clear();
  });

  it('restores the current + historical value readings after a reload', async () => {
    // Human is Black → the model plays White and moves first (ply 1).
    const store = createGameStore(fakeEngine(0.4), 'b');
    store.getState().setVariety(0);
    await playOutModelMove();
    expect(store.getState().valueHistory).toEqual([{ ply: 1, whiteValue: 0.4 }]);
    expect(currentValue(store)).toBeCloseTo(0.4, 6);

    // Simulate a page refresh: a brand-new store reads the persisted localStorage
    // blob. It must come back with the same readings (not blank).
    const reloaded = createGameStore(fakeEngine(0.4), 'b');
    expect(reloaded.getState().sanHistory).toHaveLength(1);
    expect(reloaded.getState().valueHistory).toEqual([{ ply: 1, whiteValue: 0.4 }]);
    expect(currentValue(reloaded)).toBeCloseTo(0.4, 6);
  });

  it('keeps the current reading after an undo (does not blank the gauge)', async () => {
    const store = createGameStore(fakeEngine(0.4), 'b');
    store.getState().setVariety(0);
    await playOutModelMove(); // model White → ply 1 reading (+0.40)
    playFirstLegalHumanMove(store); // human Black reply
    await playOutHumanFlash();
    await playOutModelMove(); // model White → ply 3 reading
    expect(store.getState().valueHistory).toHaveLength(2);

    // Undo rolls back to the human's turn; the ply-3 reading is dropped but the
    // ply-1 reading survives and becomes the current gauge value again.
    store.getState().undo();
    expect(store.getState().valueHistory).toEqual([{ ply: 1, whiteValue: 0.4 }]);
    expect(currentValue(store)).toBeCloseTo(0.4, 6);

    // And that surviving reading is itself persisted, so a reload after the undo
    // still shows it.
    const reloaded = createGameStore(fakeEngine(0.4), 'b');
    expect(reloaded.getState().valueHistory).toEqual([{ ply: 1, whiteValue: 0.4 }]);
    expect(currentValue(reloaded)).toBeCloseTo(0.4, 6);
  });
});
