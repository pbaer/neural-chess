// EngineClient — main-thread transport to the inference worker (Comlink).
//
// Spawns the module worker, proxies init()/forward(), and exposes a small
// promise-based API the game store consumes. Presentation-agnostic: it knows
// nothing about React or the DOM beyond `Worker`. A null-worker fallback (direct
// in-thread engine) is provided for non-Worker runtimes / tests.

import * as Comlink from 'comlink';
import type { WorkerApi, EngineMeta, ForwardReply } from '../engine/worker.ts';
import type { SearchOptions, SearchResult, SearchSnapshot } from '../search/types.ts';

export type { EngineMeta, ForwardReply } from '../engine/worker.ts';

export interface EngineClient {
  whenReady(): Promise<EngineMeta>;
  forward(planes: Float32Array, legalMask: Uint8Array, wantTrace?: boolean): Promise<ForwardReply>;
  /** Optional MCTS search (off the main thread). Present on the worker client. */
  search?(
    fen: string,
    options: SearchOptions,
    onProgress?: (snap: SearchSnapshot) => void,
  ): Promise<SearchResult>;
  /** Cancel any in-flight search. */
  cancelSearch?(): void;
  dispose(): void;
}

/** Worker-backed client (browser). Weights load once, off the main thread. */
export function createWorkerEngineClient(capsuleUrl: string): EngineClient {
  const worker = new Worker(new URL('../engine/worker.ts', import.meta.url), {
    type: 'module',
    name: 'neural-chess-inference',
  });
  const api = Comlink.wrap<WorkerApi>(worker);
  const ready = api.init(capsuleUrl);

  return {
    whenReady: () => ready,
    async forward(planes, legalMask, wantTrace = false) {
      await ready;
      return api.forward(planes, legalMask, wantTrace);
    },
    async search(fen, options, onProgress) {
      await ready;
      // Proxy the progress callback so the worker can invoke it across threads.
      const cb = onProgress ? Comlink.proxy(onProgress) : undefined;
      return api.search(fen, options, cb);
    },
    cancelSearch() {
      void api.cancelSearch();
    },
    dispose() {
      api[Comlink.releaseProxy]();
      worker.terminate();
    },
  };
}
