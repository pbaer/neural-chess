// Shared play-test helpers: a deterministic fake in-thread engine and the
// fake-timer "beats" that drive the store's auto-play, used by the play-loop
// and persistence suites. No worker, no network.

import { vi } from 'vitest';
import { NUM_MOVES } from '../../src/core/engine/index.ts';
import type { EngineClient, ForwardReply } from '../../src/core/game/engineClient.ts';
import { AUTOPLAY_PREVIEW_MS, MCTS_FLASH_MS, type createGameStore } from '../../src/core/game/gameStore.ts';

/** Fake engine: constant value head + uniform probability over the legal moves. */
export function fakeEngine(value: number): EngineClient {
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
export async function playOutModelMove() {
  await vi.advanceTimersByTimeAsync(AUTOPLAY_PREVIEW_MS + MCTS_FLASH_MS + 1);
}

/** Advance past the human flash beat so a human move lands (and any model reply starts). */
export async function playOutHumanFlash() {
  await vi.advanceTimersByTimeAsync(MCTS_FLASH_MS + 1);
}

/** Apply the first legal move for the side to move via the store's own API. */
export function playFirstLegalHumanMove(store: ReturnType<typeof createGameStore>) {
  for (let fromIdx = 0; fromIdx < 64; fromIdx++) {
    const targets = store.getState().legalTargets(fromIdx);
    if (targets.length > 0) {
      store.getState().humanMove(fromIdx, targets[0]);
      return;
    }
  }
  throw new Error('no legal human move found');
}
