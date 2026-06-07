// GeoSelfAttention — multi-head self-attention over the 64 square-tokens with a
// learned 2D relative-position (geometry) bias. Faithful port of
// src/v3/model.py::GeoSelfAttention + the SDPA semantics it relies on.
//
// CRITICAL parity detail: PyTorch passes the geometry bias to
// F.scaled_dot_product_attention as an additive `attn_mask`, so the bias is
// added AFTER the 1/√head_dim scale:
//
//     score[h,i,j] = (q[h,i] · k[h,j]) * head_dim^-0.5 + bias[h,i,j]
//     bias[h,i,j]  = rel_bias[h][ relIdx[i][j] ]
//
// qkv layout: nn.Linear(d, 3d) → reshape(N,3,n_heads,head_dim). So within the
// 3d output the order is [q|k|v] each (n_heads·head_dim), head-major.

import { linear, softmaxRowInplace } from './ops.ts';
import { relIndex } from './relIndex.ts';
import type { TraceRecorder } from './trace.ts';

const REL_IDX = relIndex(); // (64*64) — model-independent, compute once.

export interface AttnWeights {
  qkvW: Float32Array; // (3d × d)
  qkvB: Float32Array; // (3d)
  projW: Float32Array; // (d × d)
  projB: Float32Array; // (d)
  relBias?: Float32Array; // (n_heads × 225) or undefined when geometry_bias=false
}

/**
 * @param x        (N=64 × d) pre-normed tokens
 * @param d        model dim
 * @param nHeads   number of heads
 * @returns        (N × d) attention output (post output-projection)
 *
 * When `trace`/`tracePrefix` are supplied, records per-head scores and probs
 * (each n_heads×64×64) for parity + visualization.
 */
export function geoSelfAttention(
  x: Float32Array,
  d: number,
  nHeads: number,
  w: AttnWeights,
  trace?: TraceRecorder,
  tracePrefix?: string,
): Float32Array {
  const N = 64;
  const headDim = d / nHeads;
  const scale = 1 / Math.sqrt(headDim);

  // qkv: (N × 3d). Within 3d: [q(d) | k(d) | v(d)], each head-major.
  const qkv = linear(x, N, d, w.qkvW, 3 * d, w.qkvB);

  const scores = new Float32Array(nHeads * N * N);
  const qOff = 0;
  const kOff = d;
  const vOff = 2 * d;

  // scores[h,i,j] = (q·k)*scale + bias
  for (let h = 0; h < nHeads; h++) {
    const headBase = h * headDim;
    const sH = h * N * N;
    const relBiasH = w.relBias ? h * 225 : 0;
    for (let i = 0; i < N; i++) {
      const qi = i * 3 * d + qOff + headBase;
      const sRow = sH + i * N;
      const relRow = i * N;
      for (let j = 0; j < N; j++) {
        const kj = j * 3 * d + kOff + headBase;
        let dot = 0;
        for (let t = 0; t < headDim; t++) dot += qkv[qi + t] * qkv[kj + t];
        let s = dot * scale;
        if (w.relBias) s += w.relBias[relBiasH + REL_IDX[relRow + j]];
        scores[sRow + j] = s;
      }
    }
  }

  if (trace && tracePrefix) trace.record(`${tracePrefix}.scores`, scores, [nHeads, N, N]);

  // softmax over keys (last dim) per (head,row)
  const probs = Float32Array.from(scores);
  softmaxRowInplace(probs, nHeads * N, N);

  if (trace && tracePrefix) trace.record(`${tracePrefix}.probs`, probs, [nHeads, N, N]);

  // out[h,i,:] = Σ_j probs[h,i,j] · v[h,j,:] ; merge heads → (N × d), head-major
  const merged = new Float32Array(N * d);
  for (let h = 0; h < nHeads; h++) {
    const headBase = h * headDim;
    const pH = h * N * N;
    for (let i = 0; i < N; i++) {
      const pRow = pH + i * N;
      const outBase = i * d + headBase;
      for (let j = 0; j < N; j++) {
        const p = probs[pRow + j];
        if (p === 0) continue;
        const vj = j * 3 * d + vOff + headBase;
        for (let t = 0; t < headDim; t++) merged[outBase + t] += p * qkv[vj + t];
      }
    }
  }

  // output projection
  return linear(merged, N, d, w.projW, d, w.projB);
}
