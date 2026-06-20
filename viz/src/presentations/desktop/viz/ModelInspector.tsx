// ModelInspector — the model visualization, laid out as a vertical column of
// independently expand/collapsible stage sections (input planes, each transformer
// block, the policy head, the value head, …). Any number of sections can be open
// at once, each expanded/collapsed on its own. Each open section carries its OWN
// scalar inspector beside its heatmap, so a hovered cell and its exact number stay
// together within the same architecture block. Driven entirely off the capsule
// graph + kind registries, so a different model renders without code changes.
// Renders over the SAME live game position via the trace.

import { useState } from 'react';
import { shapeSummary } from '../../../core/index.ts';
import type { Capsule, EngineClient, GraphNode } from '../../../core/index.ts';
import { StageDetail } from './components/StageDetail.tsx';
import { useTelescope, useTraceVersion } from './useTelescope.ts';

export interface ModelInspectorProps {
  client: EngineClient | null;
  capsuleUrl: string;
  /** Current game FEN — drives the live trace. */
  fen: string;
}

function paramStr(p: number): string {
  if (p === 0) return '';
  if (p >= 1000) return `${(p / 1000).toFixed(p >= 10000 ? 0 : 1)}k`;
  return String(p);
}

export function ModelInspector({ client, capsuleUrl, fen }: ModelInspectorProps) {
  const { capsule, graph, ready, error } = useTelescope(client, capsuleUrl, fen, true);
  const traceVersion = useTraceVersion();
  const [open, setOpen] = useState<Set<string>>(() => new Set());

  if (error) return <section className="inspector"><div className="status error-text">{error}</div></section>;
  if (!ready || !graph || !capsule)
    return <section className="inspector"><div className="status">Loading model capsule…</div></section>;

  const toggle = (id: string) =>
    setOpen((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  const allOpen = open.size === graph.nodes.length;
  const expandAll = () => setOpen(new Set(graph.nodes.map((n) => n.id)));
  const collapseAll = () => setOpen(new Set());

  return (
    <section className="inspector" aria-label="Model Inspector">
      <div className="inspector-bar">
        <h2 className="inspector-title">Model Inspector</h2>
        <span className="inspector-meta">
          {graph.totalParams.toLocaleString()} params · {graph.nodes.length} stages
        </span>
      </div>

      <div className="inspector-actions">
        <button className="btn btn-mini" onClick={allOpen ? collapseAll : expandAll}>
          {allOpen ? 'Collapse all' : 'Expand all'}
        </button>
        <span className="inspector-hint">
          A look inside the model for the exact position on the board. Each stage below is one step the network runs,
          top to bottom — from the raw board to its move scores and a “who’s winning” number. Expand any stage to see
          the real numbers flowing through it (its “activations”) and the learned values it trained to use (its “weights”).
        </span>
      </div>

      <div className="inspector-sections">
        {graph.nodes.map((node) => (
          <InspectorSection
            key={node.id}
            node={node}
            capsule={capsule}
            traceVersion={traceVersion}
            open={open.has(node.id)}
            onToggle={() => toggle(node.id)}
          />
        ))}
      </div>
    </section>
  );
}

interface InspectorSectionProps {
  node: GraphNode;
  capsule: Capsule;
  traceVersion: number;
  open: boolean;
  onToggle: () => void;
}

function InspectorSection({ node, capsule, traceVersion, open, onToggle }: InspectorSectionProps) {
  return (
    <div className={'insp-section dag-kind-' + node.kind + (open ? ' insp-open' : '')}>
      <button className="insp-head" onClick={onToggle} aria-expanded={open}>
        <span className="insp-chevron" aria-hidden>{open ? '▾' : '▸'}</span>
        <span className="insp-label">{node.label}</span>
        <span className="insp-shape">{shapeSummary(node)}</span>
        {node.params > 0 && <span className="insp-params">{paramStr(node.params)}</span>}
      </button>
      {open && (
        <div className="insp-body">
          <StageDetail node={node} capsule={capsule} traceVersion={traceVersion} />
        </div>
      )}
    </div>
  );
}
