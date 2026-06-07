// NdArray: a minimal row-major tensor view used throughout the engine.
//
// We keep this deliberately tiny. The model is small enough that clarity beats
// cleverness; every op in ops.ts works on plain Float32Arrays with explicit
// shapes so the arithmetic stays legible and matches PyTorch op-for-op.

export interface NdArray {
  /** Row-major (C-order) data. */
  data: Float32Array;
  /** Logical shape; product(shape) === data.length. */
  shape: number[];
}

export function nd(data: Float32Array, shape: number[]): NdArray {
  return { data, shape };
}

export function numel(shape: number[]): number {
  let n = 1;
  for (const s of shape) n *= s;
  return n;
}

/** Row-major strides for a shape. */
export function strides(shape: number[]): number[] {
  const st = new Array<number>(shape.length);
  let acc = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    st[i] = acc;
    acc *= shape[i];
  }
  return st;
}

/** Allocate a zero-filled NdArray of the given shape. */
export function zeros(shape: number[]): NdArray {
  return { data: new Float32Array(numel(shape)), shape };
}

/** Owned copy of an NdArray (data is duplicated). */
export function cloneNd(a: NdArray): NdArray {
  return { data: Float32Array.from(a.data), shape: a.shape.slice() };
}
