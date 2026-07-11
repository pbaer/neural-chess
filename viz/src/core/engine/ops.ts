// Hand-written tensor ops — the load-bearing arithmetic of the forward pass.
//
// Parity rules (must match PyTorch exactly enough to pass the golden test):
//  - Linear weights are stored PyTorch (out,in) row-major: y[o] = Σ_i x[i]·W[o*in+i] + b[o].
//  - LayerNorm is over the last dim with eps=1e-5; mean/variance accumulated in
//    f64 (JS number). PyTorch uses the BIASED variance (divide by N).
//  - GELU is exact-erf: 0.5·x·(1 + erf(x/√2)) — NOT the tanh approximation.
//  - softmax / mean-pool reductions accumulate in f64.
//  - conv2d is the folded conv-with-bias (BatchNorm folded at export time), so
//    there is no BatchNorm op here at all.

import { erf } from './erf.ts';

const SQRT1_2 = Math.SQRT1_2; // 1/√2

/**
 * Linear / affine: x (rows × inDim) → out (rows × outDim).
 * W is (outDim × inDim) row-major (PyTorch layout); b is (outDim) or undefined.
 */
export function linear(
  x: Float32Array,
  rows: number,
  inDim: number,
  W: Float32Array,
  outDim: number,
  b?: Float32Array,
): Float32Array {
  const out = new Float32Array(rows * outDim);
  for (let r = 0; r < rows; r++) {
    const xBase = r * inDim;
    const oBase = r * outDim;
    for (let o = 0; o < outDim; o++) {
      const wBase = o * inDim;
      let acc = b ? b[o] : 0;
      for (let i = 0; i < inDim; i++) acc += x[xBase + i] * W[wBase + i];
      out[oBase + o] = acc;
    }
  }
  return out;
}

/**
 * conv2d over an 8×8 board (the only spatial size in this model), stride 1.
 * Input x: (cIn, H, W) row-major. Weight: (cOut, cIn, k, k). bias: (cOut).
 * Padding = floor(k/2) (SAME). Returns (cOut, H, W).
 *
 * For k=1 this is a per-square linear over channels (the v3.1 "embed").
 */
export function conv2d(
  x: Float32Array,
  cIn: number,
  H: number,
  Wd: number,
  weight: Float32Array,
  cOut: number,
  k: number,
  bias?: Float32Array,
): Float32Array {
  const pad = k >> 1;
  const out = new Float32Array(cOut * H * Wd);
  for (let oc = 0; oc < cOut; oc++) {
    const wOcBase = oc * cIn * k * k;
    const bias_oc = bias ? bias[oc] : 0;
    for (let oh = 0; oh < H; oh++) {
      for (let ow = 0; ow < Wd; ow++) {
        let acc = bias_oc;
        for (let ic = 0; ic < cIn; ic++) {
          const xIcBase = ic * H * Wd;
          const wIcBase = wOcBase + ic * k * k;
          for (let kh = 0; kh < k; kh++) {
            const ih = oh + kh - pad;
            if (ih < 0 || ih >= H) continue;
            for (let kw = 0; kw < k; kw++) {
              const iw = ow + kw - pad;
              if (iw < 0 || iw >= Wd) continue;
              acc += x[xIcBase + ih * Wd + iw] * weight[wIcBase + kh * k + kw];
            }
          }
        }
        out[oc * H * Wd + oh * Wd + ow] = acc;
      }
    }
  }
  return out;
}

/** In-place ReLU. */
export function reluInplace(x: Float32Array): Float32Array {
  for (let i = 0; i < x.length; i++) if (x[i] < 0) x[i] = 0;
  return x;
}

/** Exact-erf GELU, element-wise (returns a new array). */
export function gelu(x: Float32Array): Float32Array {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const v = x[i];
    out[i] = 0.5 * v * (1 + erf(v * SQRT1_2));
  }
  return out;
}

/**
 * LayerNorm over the last dim. x: (rows × dim). weight/bias: (dim).
 * eps=1e-5; biased variance; f64 accumulation. Returns a new array.
 */
export function layerNorm(
  x: Float32Array,
  rows: number,
  dim: number,
  weight: Float32Array,
  bias: Float32Array,
  eps = 1e-5,
): Float32Array {
  const out = new Float32Array(rows * dim);
  for (let r = 0; r < rows; r++) {
    const base = r * dim;
    let mean = 0;
    for (let i = 0; i < dim; i++) mean += x[base + i];
    mean /= dim;
    let varAcc = 0;
    for (let i = 0; i < dim; i++) {
      const d = x[base + i] - mean;
      varAcc += d * d;
    }
    varAcc /= dim;
    const invStd = 1 / Math.sqrt(varAcc + eps);
    for (let i = 0; i < dim; i++) {
      out[base + i] = (x[base + i] - mean) * invStd * weight[i] + bias[i];
    }
  }
  return out;
}

/**
 * Row-wise softmax, in place over a (rows × cols) matrix.
 * Numerically stabilized (subtract row max); f64 accumulation.
 */
export function softmaxRowInplace(x: Float32Array, rows: number, cols: number): Float32Array {
  for (let r = 0; r < rows; r++) {
    const base = r * cols;
    let mx = -Infinity;
    for (let c = 0; c < cols; c++) if (x[base + c] > mx) mx = x[base + c];
    let sum = 0;
    for (let c = 0; c < cols; c++) {
      const e = Math.exp(x[base + c] - mx);
      x[base + c] = e;
      sum += e;
    }
    const inv = 1 / sum;
    for (let c = 0; c < cols; c++) x[base + c] *= inv;
  }
  return x;
}

/**
 * Mean over the row axis: x (rows × dim) → (dim). f64 accumulation.
 */
export function meanPool(x: Float32Array, rows: number, dim: number): Float32Array {
  const acc = new Float64Array(dim);
  for (let r = 0; r < rows; r++) {
    const base = r * dim;
    for (let i = 0; i < dim; i++) acc[i] += x[base + i];
  }
  const out = new Float32Array(dim);
  for (let i = 0; i < dim; i++) out[i] = acc[i] / rows;
  return out;
}
