// traceIndex — the version-neutral join between a graph node and the trace entry
// names the engine records for it. Switches on `kind` (never on a hardcoded
// architecture) and composes block-internal keys from the node id, mirroring how
// model.ts / attention.ts name their recorded intermediates.

import type { GraphNode } from './graph.ts';

export interface TraceField {
  /** Trace entry name (key into traceStore.entry). */
  key: string;
  /** Short label for the field chip. */
  label: string;
  /** One-line description of what this intermediate is. */
  blurb: string;
  /** The stage's single output (vs an internal step). */
  role: 'output' | 'internal';
  /** Optional op kind for the contextual explanation lens. */
  opKind?: string;
}

/**
 * The trace fields a node exposes, in forward order. Mirrors the record() calls
 * in model.ts and attention.ts:
 *   input_planes → "planes"
 *   embed/stem_conv → "embed"
 *   tokenize → "tokenize"
 *   block.<i> → ln1, attn.scores, attn.probs, attn, postAttn, ln2, ffn, postFfn
 *   layernorm (final) → "<id>"
 *   policy_head → "policy_logits"
 *   value_head → "value"
 */
export function traceFieldsFor(node: GraphNode): TraceField[] {
  switch (node.kind) {
    case 'input_planes':
      return [{ key: 'planes', label: 'planes', blurb: 'The raw input board planes.', role: 'output', opKind: 'input_planes' }];

    case 'embed':
    case 'stem_conv':
      return [{ key: 'embed', label: 'embed', blurb: 'Per-square features after the stem + ReLU.', role: 'output', opKind: node.kind }];

    case 'tokenize':
      return [{ key: 'tokenize', label: 'tokens', blurb: '64 square-tokens after adding the position embedding.', role: 'output', opKind: 'tokenize' }];

    case 'block': {
      const id = node.id;
      return [
        { key: `${id}.ln1`, label: 'LN1', blurb: 'Pre-attention LayerNorm of the residual stream.', role: 'internal', opKind: 'layernorm' },
        { key: `${id}.attn.scores`, label: 'scores', blurb: 'Per-head raw attention scores q·k/√d + geometry bias.', role: 'internal', opKind: 'attention' },
        { key: `${id}.attn.probs`, label: 'probs', blurb: 'Per-head attention weights (softmax over keys).', role: 'internal', opKind: 'softmax' },
        { key: `${id}.attn`, label: 'attn out', blurb: 'Attention output after merging heads + output projection.', role: 'internal', opKind: 'attention' },
        { key: `${id}.postAttn`, label: '+attn', blurb: 'Residual stream after adding the attention output.', role: 'internal', opKind: 'residual' },
        { key: `${id}.ln2`, label: 'LN2', blurb: 'Pre-FFN LayerNorm.', role: 'internal', opKind: 'layernorm' },
        { key: `${id}.ffn`, label: 'FFN', blurb: 'Position-wise feed-forward output (Linear→GELU→Linear).', role: 'internal', opKind: 'ffn' },
        { key: `${id}.postFfn`, label: '+ffn', blurb: 'Residual stream after adding the FFN output (block output).', role: 'output', opKind: 'residual' },
      ];
    }

    case 'layernorm':
      return [{ key: node.id, label: 'normed', blurb: 'The normalized token stream.', role: 'output', opKind: 'layernorm' }];

    case 'policy_head':
      return [{ key: 'policy_logits', label: 'logits', blurb: 'Flat 4672 policy logits (64 squares × 73 move types).', role: 'output', opKind: 'policy_head' }];

    case 'value_head':
      return [{ key: 'value', label: 'value', blurb: 'Scalar position evaluation in [-1,1] after tanh.', role: 'output', opKind: 'value_head' }];

    default:
      return [];
  }
}

/** The single trace key representing a node's OUTPUT activation, or null. */
export function outputTraceKey(node: GraphNode): string | null {
  const fields = traceFieldsFor(node);
  const out = fields.find((f) => f.role === 'output');
  return out ? out.key : null;
}
