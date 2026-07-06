// Value-head history in the game store: each model move appends one White-framed
// value reading to `valueHistory`; New Game resets it and Undo trims the readings
// for plies that no longer exist. Uses a fake in-thread engine (fixed value +
// uniform-over-legal policy) and fake timers to drive the auto-play beats — fully
// deterministic, no worker/network.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { NUM_MOVES } from '../../src/core/engine/index.ts';
import type { EngineClient, ForwardReply } from '../../src/core/game/engineClient.ts';
import {
  createGameStore,
  AUTOPLAY_PREVIEW_MS,
  MCTS_FLASH_MS,
} from '../../src/core/game/gameStore.ts';

/** Fake engine: constant value head + uniform probability over the legal moves. */
function fakeEngine(value: number): EngineClient {
  return {
    whenReady: async () => ({}) as never,
    async forward(_planes, legalMask): Promise<ForwardReply> {
      const policyProbs = new Float32Array(NUM_MOVES);
      let best = -1;
      for (let i = 0; i < NUM_MOVES; i++) {
        if (legalMask[i]) {
          policyProbs[i] = 1;
          if (best < 0) best = i;
        }
      }
      return { policyLogits: policyProbs.slice(), policyProbs, value, bestLegalIndex: best };
    },
    dispose() {},
  };
}

/** Advance past both auto-play beats (preview → flash) so a model move lands. */
async function playOutModelMove() {
  await vi.advanceTimersByTimeAsync(AUTOPLAY_PREVIEW_MS + MCTS_FLASH_MS + 1);
}

/** Advance past the human flash beat so a human move lands (and any model reply starts). */
async function playOutHumanFlash() {
  await vi.advanceTimersByTimeAsync(MCTS_FLASH_MS + 1);
}

/** Apply the first legal move for the side to move via the store's own API. */
function playFirstLegalHumanMove(store: ReturnType<typeof createGameStore>) {
  for (let fromIdx = 0; fromIdx < 64; fromIdx++) {
    const targets = store.getState().legalTargets(fromIdx);
    if (targets.length > 0) {
      store.getState().humanMove(fromIdx, targets[0]);
      return;
    }
  }
  throw new Error('no legal human move found');
}

describe('gameStore valueHistory', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('appends a White-framed value reading each time the model moves', async () => {
    // Human is Black → the model plays White and moves first (ply 1). variety 0
    // makes selection a deterministic argmax over the (uniform) legal policy.
    const store = createGameStore(fakeEngine(0.4), 'b');
    store.getState().setVariety(0);
    await playOutModelMove();

    const h1 = store.getState().valueHistory;
    expect(h1).toHaveLength(1);
    expect(h1[0].ply).toBe(1);
    // Model played White, so raw +0.40 (side-to-move) stays +0.40 in the White frame.
    expect(h1[0].whiteValue).toBeCloseTo(0.4, 6);
    expect(store.getState().sanHistory).toHaveLength(1);

    // Black (human) replies; the model (White) then makes its 2nd move at ply 3.
    playFirstLegalHumanMove(store);
    await playOutHumanFlash();
    await playOutModelMove();

    const h2 = store.getState().valueHistory;
    expect(h2).toHaveLength(2);
    expect(h2[1].ply).toBe(3);
  });

  it('re-frames the value to Black(−) when the model plays Black', async () => {
    // Human is White and moves first; the model plays Black. A raw +0.40 from
    // Black's perspective means Black is ahead → White frame is −0.40.
    const store = createGameStore(fakeEngine(0.4), 'w');
    store.getState().setVariety(0);
    playFirstLegalHumanMove(store);
    await playOutHumanFlash();
    await playOutModelMove(); // model (Black) replies at ply 2

    const h = store.getState().valueHistory;
    expect(h).toHaveLength(1);
    expect(h[0].ply).toBe(2);
    expect(h[0].whiteValue).toBeCloseTo(-0.4, 6);
  });

  it('New Game clears the value history', async () => {
    const store = createGameStore(fakeEngine(0.4), 'b');
    store.getState().setVariety(0);
    await playOutModelMove();
    expect(store.getState().valueHistory).toHaveLength(1);

    store.getState().newGame('b');
    expect(store.getState().valueHistory).toHaveLength(0);
  });

  it('Undo drops readings for plies that no longer exist', async () => {
    const store = createGameStore(fakeEngine(0.4), 'b');
    store.getState().setVariety(0);
    await playOutModelMove(); // model White move → ply 1 reading
    playFirstLegalHumanMove(store); // human Black reply
    await playOutHumanFlash();
    await playOutModelMove(); // model White move → ply 3 reading
    expect(store.getState().valueHistory).toHaveLength(2);

    // Undo rolls back to the human's turn (undoing the model reply too); the ply-3
    // reading is now beyond the history and must be dropped.
    store.getState().undo();
    const h = store.getState().valueHistory;
    expect(h.every((v) => v.ply <= store.getState().sanHistory.length)).toBe(true);
    expect(h.some((v) => v.ply === 3)).toBe(false);
  });
});
