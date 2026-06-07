// Forward pass — driven off the capsule `graph` via a kind→compute registry, so
// it is architecture-version-NEUTRAL (no "v3" anywhere). Each stage reads its
// upstream activations + its own tensors and writes one output. Adding a new op
// kind is a localized registry entry, never a rewrite.
//
// Confirmed against src/v3/model.py::ChessModelV3.forward. Reductions accumulate
// in f64 inside ops.ts; BatchNorm is folded at export so there is no BN op.

import { Capsule, type GraphStage } from './capsule.ts';
import { conv2d, gelu, layerNorm, linear, meanPool, reluInplace } from './ops.ts';
import { geoSelfAttention, type AttnWeights } from './attention.ts';
import { TraceRecorder } from './trace.ts';
import { NUM_MOVE_TYPES, NUM_MOVES } from './moves.ts';

interface Activation {
  data: Float32Array;
  shape: number[];
}

export interface ForwardResult {
  /** Raw policy logits (4672), pre-mask. */
  policyLogits: Float32Array;
  /** Value scalar in [-1,1] (side-to-move perspective). */
  value: number;
  trace?: TraceRecorder;
}

interface ForwardCtx {
  acts: Map<string, Activation>;
  trace?: TraceRecorder;
  policyLogits?: Float32Array;
  value?: number;
}

type ComputeFn = (m: Model, stage: GraphStage, inputs: Activation[], ctx: ForwardCtx) => Activation | null;

/** Find a stage's tensor whose name ends with `suffix` (graph-driven lookup). */
function wname(stage: GraphStage, suffix: string): string | undefined {
  return stage.weights.find((n) => n === suffix || n.endsWith('.' + suffix));
}

const REGISTRY: Record<string, ComputeFn> = {
  input_planes(_m, _stage, _inputs, ctx) {
    // io.planes output is seeded directly into ctx.acts before the walk.
    return ctx.acts.get('io.planes')!;
  },

  embed: convStem,
  stem_conv: convStem,

  tokenize(m, stage, inputs, ctx) {
    const embed = inputs[0]; // (C,8,8)
    const C = embed.shape[0];
    const posName = wname(stage, 'pos_emb')!;
    const pos = m.capsule.data(posName); // (1,64,C) flat = 64*C
    const t = new Float32Array(64 * C);
    for (let s = 0; s < 64; s++) {
      for (let c = 0; c < C; c++) {
        // embed[c][h][w] with h=s>>3, w=s&7 → flat c*64 + s
        t[s * C + c] = embed.data[c * 64 + s] + pos[s * C + c];
      }
    }
    ctx.trace?.record('tokenize', t, [64, C]);
    return { data: t, shape: [64, C] };
  },

  block(m, stage, inputs, ctx) {
    const x = inputs[0].data; // (64,C)
    const d = Number(stage.dims.d);
    const heads = Number(stage.dims.heads);
    const id = stage.id;

    const ln1 = layerNorm(x, 64, d, m.w(stage, 'ln1.weight'), m.w(stage, 'ln1.bias'));
    ctx.trace?.record(`${id}.ln1`, ln1, [64, d]);

    const relName = wname(stage, 'attn.rel_bias');
    const attnW: AttnWeights = {
      qkvW: m.w(stage, 'attn.qkv.weight'),
      qkvB: m.w(stage, 'attn.qkv.bias'),
      projW: m.w(stage, 'attn.proj.weight'),
      projB: m.w(stage, 'attn.proj.bias'),
      relBias: relName ? m.capsule.data(relName) : undefined,
    };
    const attn = geoSelfAttention(ln1, d, heads, attnW, ctx.trace, `${id}.attn`);
    ctx.trace?.record(`${id}.attn`, attn, [64, d]);

    const postAttn = new Float32Array(64 * d);
    for (let i = 0; i < postAttn.length; i++) postAttn[i] = x[i] + attn[i];
    ctx.trace?.record(`${id}.postAttn`, postAttn, [64, d]);

    const ln2 = layerNorm(postAttn, 64, d, m.w(stage, 'ln2.weight'), m.w(stage, 'ln2.bias'));
    ctx.trace?.record(`${id}.ln2`, ln2, [64, d]);

    const ffn0W = m.w(stage, 'ffn.0.weight');
    const ffn0B = m.w(stage, 'ffn.0.bias');
    const hidden = ffn0B.length;
    const h1 = linear(ln2, 64, d, ffn0W, hidden, ffn0B);
    const a1 = gelu(h1);
    const ffnOut = linear(a1, 64, hidden, m.w(stage, 'ffn.2.weight'), d, m.w(stage, 'ffn.2.bias'));
    ctx.trace?.record(`${id}.ffn`, ffnOut, [64, d]);

    const out = new Float32Array(64 * d);
    for (let i = 0; i < out.length; i++) out[i] = postAttn[i] + ffnOut[i];
    ctx.trace?.record(`${id}.postFfn`, out, [64, d]);
    return { data: out, shape: [64, d] };
  },

  layernorm(m, stage, inputs, ctx) {
    const x = inputs[0].data;
    const d = Number(stage.dims.d);
    const y = layerNorm(x, 64, d, m.w(stage, 'weight'), m.w(stage, 'bias'));
    ctx.trace?.record(stage.id, y, [64, d]);
    return { data: y, shape: [64, d] };
  },

  policy_head(m, stage, inputs, ctx) {
    const t = inputs[0].data; // (64,C)
    const C = inputs[0].shape[1];
    const W = m.w(stage, 'policy_head.weight'); // (73,C)
    const b = m.w(stage, 'policy_head.bias'); // (73)
    const logits = linear(t, 64, C, W, NUM_MOVE_TYPES, b); // (64,73) row-major == flat[s*73+mt]
    const out = logits.length === NUM_MOVES ? logits : logits.subarray(0, NUM_MOVES);
    ctx.policyLogits = out;
    ctx.trace?.record('policy_logits', out, [NUM_MOVES]);
    return { data: out, shape: [NUM_MOVES] };
  },

  value_head(m, stage, inputs, ctx) {
    const t = inputs[0].data; // (64,C)
    const C = inputs[0].shape[1];
    const pooled = meanPool(t, 64, C); // (C)
    const h1 = linear(pooled, 1, C, m.w(stage, 'value_fc1.weight'), Number(stage.dims.hidden), m.w(stage, 'value_fc1.bias'));
    reluInplace(h1);
    const pre = linear(h1, 1, h1.length, m.w(stage, 'value_fc2.weight'), 1, m.w(stage, 'value_fc2.bias'));
    const value = Math.tanh(pre[0]);
    ctx.value = value;
    ctx.trace?.record('value', new Float32Array([value]), [1]);
    return { data: new Float32Array([value]), shape: [1] };
  },
};

function convStem(m: Model, stage: GraphStage, inputs: Activation[], ctx: ForwardCtx): Activation {
  const x = inputs[0].data; // (cIn,8,8)
  const wName = stage.weights.find((n) => n.endsWith('weight'))!;
  const bName = stage.weights.find((n) => n.endsWith('bias'));
  const Wt = m.capsule.tensor(wName); // (cOut,cIn,k,k)
  const [cOut, cIn, k] = [Wt.shape[0], Wt.shape[1], Wt.shape[Wt.shape.length - 1]];
  const b = bName ? m.capsule.data(bName) : undefined;
  const out = conv2d(x, cIn, 8, 8, Wt.data, cOut, k, b);
  reluInplace(out); // input_bn folded into conv; ReLU follows (matches model.py)
  ctx.trace?.record('embed', out, [cOut, 8, 8]);
  return { data: out, shape: [cOut, 8, 8] };
}

export class Model {
  readonly capsule: Capsule;
  constructor(capsule: Capsule) {
    this.capsule = capsule;
  }

  /** Stage-scoped tensor lookup by suffix. */
  w(stage: GraphStage, suffix: string): Float32Array {
    const name = wname(stage, suffix);
    if (!name) throw new Error(`Stage "${stage.id}" has no weight ending in "${suffix}".`);
    return this.capsule.data(name);
  }

  /**
   * Run the forward pass over `planes` (Float32Array of input_planes·64,
   * plane-major, position h*8+w). Returns raw policy logits + value, plus the
   * trace if a recorder is supplied.
   */
  forward(planes: Float32Array, opts: { trace?: TraceRecorder } = {}): ForwardResult {
    const ctx: ForwardCtx = { acts: new Map(), trace: opts.trace };
    const planesStage = this.capsule.graph.find((s) => s.kind === 'input_planes');
    const planeId = planesStage ? planesStage.id : 'io.planes';
    const inPlanes = this.capsule.config.input_planes;
    ctx.trace?.record('planes', planes, [inPlanes, 8, 8]);
    ctx.acts.set(planeId, { data: planes, shape: [inPlanes, 8, 8] });

    for (const stage of this.capsule.graph) {
      const fn = REGISTRY[stage.kind];
      if (!fn) throw new Error(`No compute registered for stage kind "${stage.kind}" (stage ${stage.id}).`);
      const inputs = stage.reads.map((r) => {
        const a = ctx.acts.get(r);
        if (!a) throw new Error(`Stage "${stage.id}" reads "${r}" which has no activation.`);
        return a;
      });
      const out = fn(this, stage, inputs, ctx);
      if (out) ctx.acts.set(stage.id, out);
    }

    if (!ctx.policyLogits || ctx.value === undefined) {
      throw new Error('Forward pass did not produce both policy and value heads.');
    }
    return { policyLogits: ctx.policyLogits, value: ctx.value, trace: opts.trace };
  }
}
