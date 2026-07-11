// Engine contract — the public API the rest of the app imports.
//
// The worker is a pure math kernel: it consumes {planes, legalMask} and returns
// {policyLogits, policyProbs, value, bestLegalIndex, trace?}. Featurize / move
// decode / un-rotation are the game layer's job (main thread, chess.js). For M1
// the parity tests drive `forward` directly.

import { Capsule, fetchCapsule, loadCapsuleFromBytes, type CapsuleManifest } from './capsule.ts';
import { Model } from './model.ts';
import { NUM_MOVES } from './moves.ts';
import { TraceRecorder, type Trace, type TraceOptions } from './trace.ts';

export interface ForwardOptions {
  /** (4672) 1=legal, 0=illegal. When given, policyProbs is masked+renormalized. */
  legalMask?: Uint8Array;
  /** Pass a recorder (or trace options) to capture every intermediate. */
  trace?: TraceRecorder | TraceOptions | boolean;
}

export interface EngineResult {
  /** Raw policy logits (4672), pre-mask. */
  policyLogits: Float32Array;
  /** softmax over all 4672, then masked + renormalized (matches inference.py). */
  policyProbs: Float32Array;
  /** Value in [-1,1], side-to-move perspective. */
  value: number;
  /** argmax over legal logits (or over all when no mask); -1 if mask is all-zero. */
  bestLegalIndex: number;
  trace?: Trace;
}

export interface Engine {
  readonly capsule: Capsule;
  readonly config: Capsule['config'];
  readonly graph: Capsule['graph'];
  forward(planes: Float32Array, opts?: ForwardOptions): EngineResult;
}

function resolveRecorder(t: ForwardOptions['trace']): TraceRecorder | undefined {
  if (!t) return undefined;
  if (t instanceof TraceRecorder) return t;
  if (t === true) return new TraceRecorder();
  return new TraceRecorder(t);
}

/** Full-domain softmax (over all 4672), f64 accumulation. Returns new array. */
function softmaxAll(logits: Float32Array): Float32Array {
  let mx = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > mx) mx = logits[i];
  const out = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] - mx);
    out[i] = e;
    sum += e;
  }
  const inv = 1 / sum;
  for (let i = 0; i < out.length; i++) out[i] *= inv;
  return out;
}

class EngineImpl implements Engine {
  readonly capsule: Capsule;
  private model: Model;
  constructor(capsule: Capsule) {
    this.capsule = capsule;
    this.model = new Model(capsule);
  }
  get config() {
    return this.capsule.config;
  }
  get graph() {
    return this.capsule.graph;
  }

  forward(planes: Float32Array, opts: ForwardOptions = {}): EngineResult {
    const recorder = resolveRecorder(opts.trace);
    const { policyLogits, value } = this.model.forward(planes, { trace: recorder });

    const probs = softmaxAll(policyLogits);
    let bestLegalIndex = -1;
    const mask = opts.legalMask;
    if (mask) {
      // Mask + renormalize probs; argmax over legal logits.
      let total = 0;
      let bestLogit = -Infinity;
      for (let i = 0; i < NUM_MOVES; i++) {
        if (mask[i]) {
          total += probs[i];
          if (policyLogits[i] > bestLogit) {
            bestLogit = policyLogits[i];
            bestLegalIndex = i;
          }
        } else {
          probs[i] = 0;
        }
      }
      if (total > 0) {
        const inv = 1 / total;
        for (let i = 0; i < NUM_MOVES; i++) probs[i] *= inv;
      }
    } else {
      let bestLogit = -Infinity;
      for (let i = 0; i < NUM_MOVES; i++) {
        if (policyLogits[i] > bestLogit) {
          bestLogit = policyLogits[i];
          bestLegalIndex = i;
        }
      }
    }

    return {
      policyLogits,
      policyProbs: probs,
      value,
      bestLegalIndex,
      trace: recorder?.finalize(),
    };
  }
}

/** Build an engine from in-memory capsule bytes (no IO; used by tests). */
export function createEngineFromBytes(manifest: CapsuleManifest, weights: ArrayBuffer): Engine {
  return new EngineImpl(loadCapsuleFromBytes(manifest, weights));
}

/** Build an engine by fetching capsule.json (+weights.bin) from a URL. */
export async function createEngine(capsuleUrl: string): Promise<Engine> {
  return new EngineImpl(await fetchCapsule(capsuleUrl));
}

// Re-exports — the engine's surface for the rest of the app.
export { Capsule, fetchCapsule, loadCapsuleFromBytes } from './capsule.ts';
export type { CapsuleManifest, CapsuleConfig, GraphStage, TensorIndexEntry } from './capsule.ts';
export { featurize, applyInt8Truncation, NUM_PLANES } from './featurize.ts';
export type { FeaturizeMode } from './featurize.ts';
export {
  encodeMove,
  decodeMove,
  legalMask,
  legalPolicySoftmax,
  pieceToMoveProbs,
  rotateSquare,
  NUM_MOVES,
  NUM_MOVE_TYPES,
} from './moves.ts';
export type { Move } from './moves.ts';
export type { BoardState, PieceInfo, Color, PieceType } from './boardState.ts';
export { TraceRecorder } from './trace.ts';
export type { Trace, TraceEntry, TraceOptions, TraceGranularity } from './trace.ts';
export { Model } from './model.ts';
export type { ForwardResult } from './model.ts';
