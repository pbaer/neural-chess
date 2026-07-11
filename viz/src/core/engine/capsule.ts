// Capsule loader — the tool's ONLY coupling to a model.
//
// A Model Capsule is architecture-version-NEUTRAL: a self-describing `graph`
// of typed stages + a `tensors` index into a concatenated little-endian f32
// `weights.bin` (BatchNorm folded at export). The tool validates the FORMAT
// version (`capsule_version`), tolerates unknown optional fields, and drives
// the forward pass off `graph[].kind` — never hardcoding any architecture.
//
// See viz/scripts/export/export_model.py for the producer.

import { nd, numel, type NdArray } from './tensor.ts';

/** Format versions this reader understands. */
export const SUPPORTED_CAPSULE_VERSIONS = [1] as const;

export interface TensorIndexEntry {
  name: string;
  shape: number[];
  dtype: string;
  /** Offset in FLOATS (not bytes) into weights.bin. */
  offset: number;
  /** Length in floats. */
  length: number;
}

export interface GraphStage {
  id: string;
  kind: string;
  dims: Record<string, number | boolean>;
  weights: string[];
  reads: string[];
}

export interface CapsuleConfig {
  d_model: number;
  n_heads: number;
  n_blocks: number;
  ffn_mult: number;
  stem_kernel: number;
  stem_blocks: number;
  value_hidden: number;
  geometry_bias: boolean;
  input_planes: number;
  [k: string]: number | boolean;
}

export interface CapsuleManifest {
  capsule_version: number;
  arch: string;
  model_id: string;
  param_count: number;
  config: CapsuleConfig;
  weights_file: string;
  weights_bytes: number;
  weights_sha256: string;
  tensors: TensorIndexEntry[];
  graph: GraphStage[];
  [k: string]: unknown;
}

export class Capsule {
  readonly manifest: CapsuleManifest;
  readonly config: CapsuleConfig;
  readonly graph: GraphStage[];
  private tensors = new Map<string, NdArray>();

  constructor(manifest: CapsuleManifest, weights: ArrayBuffer) {
    validateManifest(manifest);
    this.manifest = manifest;
    this.config = manifest.config;
    this.graph = manifest.graph;

    const floats = new Float32Array(weights);
    for (const t of manifest.tensors) {
      if (t.dtype !== 'f32') {
        throw new Error(`Capsule tensor "${t.name}" has unsupported dtype "${t.dtype}" (expected f32).`);
      }
      const end = t.offset + t.length;
      if (end > floats.length) {
        throw new Error(
          `Capsule tensor "${t.name}" runs past weights.bin (offset ${t.offset} + length ${t.length} > ${floats.length}).`,
        );
      }
      if (t.length !== numel(t.shape)) {
        throw new Error(`Capsule tensor "${t.name}" length ${t.length} != product(shape ${JSON.stringify(t.shape)}).`);
      }
      // Zero-copy view into the weights buffer.
      const view = new Float32Array(weights, t.offset * 4, t.length);
      this.tensors.set(t.name, nd(view, t.shape));
    }
  }

  hasTensor(name: string): boolean {
    return this.tensors.has(name);
  }

  /** Fetch a tensor by name; throws if absent. */
  tensor(name: string): NdArray {
    const t = this.tensors.get(name);
    if (!t) throw new Error(`Capsule missing tensor "${name}".`);
    return t;
  }

  /** Raw data of a tensor by name. */
  data(name: string): Float32Array {
    return this.tensor(name).data;
  }
}

function validateManifest(m: CapsuleManifest): void {
  if (typeof m.capsule_version !== 'number') {
    throw new Error('Capsule manifest missing numeric "capsule_version".');
  }
  if (!SUPPORTED_CAPSULE_VERSIONS.includes(m.capsule_version as 1)) {
    throw new Error(
      `Unsupported capsule_version ${m.capsule_version}. ` +
        `This build understands: ${SUPPORTED_CAPSULE_VERSIONS.join(', ')}.`,
    );
  }
  if (!Array.isArray(m.tensors) || !Array.isArray(m.graph)) {
    throw new Error('Capsule manifest must contain "tensors" and "graph" arrays.');
  }
  if (!m.config || typeof m.config.d_model !== 'number') {
    throw new Error('Capsule manifest missing "config.d_model".');
  }
}

/** Build a Capsule from already-loaded manifest JSON + weights bytes (no IO). */
export function loadCapsuleFromBytes(manifest: CapsuleManifest, weights: ArrayBuffer): Capsule {
  return new Capsule(manifest, weights);
}

/**
 * Fetch + parse a capsule from a URL pointing at capsule.json. weights.bin is
 * resolved relative to it. (Browser / fetch-capable runtimes.)
 */
export async function fetchCapsule(capsuleUrl: string): Promise<Capsule> {
  const manifestRes = await fetch(capsuleUrl);
  if (!manifestRes.ok) {
    throw new Error(`Failed to fetch capsule manifest ${capsuleUrl}: HTTP ${manifestRes.status}.`);
  }
  const manifest = (await manifestRes.json()) as CapsuleManifest;
  const weightsUrl = new URL(manifest.weights_file ?? 'weights.bin', capsuleUrl).href;
  const weightsRes = await fetch(weightsUrl);
  if (!weightsRes.ok) {
    throw new Error(`Failed to fetch capsule weights ${weightsUrl}: HTTP ${weightsRes.status}.`);
  }
  const weights = await weightsRes.arrayBuffer();
  return new Capsule(manifest, weights);
}
