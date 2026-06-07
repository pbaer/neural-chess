// Inference Web Worker — a thin Comlink wrapper around the (finished, parity-
// verified) engine kernel. It owns NO chess logic: it receives {planes,
// legalMask} and returns {policyLogits, policyProbs, value, bestLegalIndex,
// trace?}. Weights load once via init(); result buffers are transferred
// zero-copy back to the main thread.

import * as Comlink from 'comlink';
import { createEngine, type Engine } from './index.ts';
import type { CapsuleConfig } from './capsule.ts';

export interface EngineMeta {
  modelId: string;
  arch: string;
  paramCount: number;
  config: CapsuleConfig;
}

export interface SerializedTrace {
  entries: Array<{ name: string; data: Float32Array; shape: number[] }>;
  bytes: number;
  granularity: string;
  capped: boolean;
}

export interface ForwardReply {
  policyLogits: Float32Array;
  policyProbs: Float32Array;
  value: number;
  bestLegalIndex: number;
  /** Present only when requested (M3 visualization layer). */
  trace?: SerializedTrace;
}

let enginePromise: Promise<Engine> | null = null;

const api = {
  /** Fetch + parse the capsule and build the engine. Returns model metadata. */
  async init(capsuleUrl: string): Promise<EngineMeta> {
    enginePromise = createEngine(capsuleUrl);
    const engine = await enginePromise;
    const m = engine.capsule.manifest;
    return { modelId: m.model_id, arch: m.arch, paramCount: m.param_count, config: engine.config };
  },

  /** One-shot forward pass. `wantTrace` captures every intermediate (M3). */
  async forward(planes: Float32Array, legalMask: Uint8Array, wantTrace = false): Promise<ForwardReply> {
    if (!enginePromise) throw new Error('Worker not initialized — call init(capsuleUrl) first.');
    const engine = await enginePromise;
    const res = engine.forward(planes, { legalMask, trace: wantTrace });

    const transfers: Transferable[] = [res.policyLogits.buffer, res.policyProbs.buffer];
    let trace: SerializedTrace | undefined;
    if (res.trace) {
      const entries: SerializedTrace['entries'] = [];
      for (const [name, e] of res.trace.entries) {
        entries.push({ name, data: e.data, shape: e.shape });
        transfers.push(e.data.buffer);
      }
      trace = { entries, bytes: res.trace.bytes, granularity: res.trace.granularity, capped: res.trace.capped };
    }

    const reply: ForwardReply = {
      policyLogits: res.policyLogits,
      policyProbs: res.policyProbs,
      value: res.value,
      bestLegalIndex: res.bestLegalIndex,
      trace,
    };
    return Comlink.transfer(reply, transfers);
  },
};

export type WorkerApi = typeof api;

Comlink.expose(api);
