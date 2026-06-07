// model-graph — builds the presentation-facing architecture graph FROM the
// capsule `graph[]`, with NO hardcoded architecture. Every node's structure,
// connectivity, dims and parameter count come from the capsule; a different
// capsule (more/fewer blocks, a conv stem, geometry-bias off, wider heads)
// renders with zero code changes.

import type { GraphStage, TensorIndexEntry } from '../engine/capsule.ts';

export interface GraphNode {
  id: string;
  kind: string;
  /** Human label derived generically from kind (+ numeric suffix of id). */
  label: string;
  dims: Record<string, number | boolean>;
  /** Tensor names owned by this stage. */
  weightNames: string[];
  /** Total learned parameters in this stage (sum of its tensors' lengths). */
  params: number;
  /** Upstream stage ids this node reads. */
  reads: string[];
}

export interface ModelGraph {
  nodes: GraphNode[];
  byId: Map<string, GraphNode>;
  totalParams: number;
}

/** Base display names per known kind. Unknown kinds fall back to the raw kind. */
const KIND_LABELS: Record<string, string> = {
  input_planes: 'Input planes',
  embed: 'Embed',
  stem_conv: 'Conv stem',
  stem_block: 'Stem block',
  tokenize: 'Tokenize + position',
  block: 'Block',
  layernorm: 'LayerNorm',
  policy_head: 'Policy head',
  value_head: 'Value head',
};

/** Trailing integer of an id like "block.7" → 7, else null. */
export function idIndex(id: string): number | null {
  const m = /(\d+)$/.exec(id);
  return m ? Number(m[1]) : null;
}

function labelFor(stage: GraphStage): string {
  const base = KIND_LABELS[stage.kind] ?? stage.kind;
  const n = idIndex(stage.id);
  // Repeatable stages (blocks/stem blocks) read better numbered.
  if (n !== null && (stage.kind === 'block' || stage.kind === 'stem_block')) return `${base} ${n}`;
  return base;
}

export interface GraphSource {
  graph: GraphStage[];
  tensors: TensorIndexEntry[];
}

/** Build the architecture graph from a capsule manifest (graph + tensor index). */
export function buildModelGraph(src: GraphSource): ModelGraph {
  const lenByName = new Map<string, number>();
  for (const t of src.tensors) lenByName.set(t.name, t.length);

  const nodes: GraphNode[] = src.graph.map((stage) => {
    let params = 0;
    for (const w of stage.weights) params += lenByName.get(w) ?? 0;
    return {
      id: stage.id,
      kind: stage.kind,
      label: labelFor(stage),
      dims: stage.dims,
      weightNames: stage.weights.slice(),
      params,
      reads: stage.reads.slice(),
    };
  });

  const byId = new Map<string, GraphNode>();
  for (const n of nodes) byId.set(n.id, n);
  const totalParams = nodes.reduce((s, n) => s + n.params, 0);
  return { nodes, byId, totalParams };
}

/** A compact one-line shape summary for a node (for the DAG box). */
export function shapeSummary(node: GraphNode): string {
  const d = node.dims;
  switch (node.kind) {
    case 'input_planes':
      return `${d.planes}×${d.h}×${d.w}`;
    case 'embed':
    case 'stem_conv':
      return `${d.in}→${d.out}`;
    case 'tokenize':
      return `${d.tokens}×${d.d}`;
    case 'block':
      return `d${d.d} · ${d.heads}h · ffn${d.ffn}`;
    case 'layernorm':
      return `d${d.d}`;
    case 'policy_head':
      return `${d.in}→${d.move_types} · ${d.moves}`;
    case 'value_head':
      return `${d.in}→${d.hidden}→1`;
    default:
      return Object.entries(d)
        .map(([k, v]) => `${k}=${v}`)
        .join(' · ');
  }
}
