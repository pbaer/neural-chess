// ScalarInspector — the bottom of the telescope. Shows ONE number exactly: its
// math name, its full-precision value, and (when the caller supplies one) the
// expanded arithmetic that produced it. Purely presentational — callers build
// the math name + term breakdown from the trace/weights.

export interface ScalarTerm {
  label: string;
  value: number;
}

export interface ScalarView {
  /** Math name, e.g. "probs[h=0][e4→d5]" or "W_embed[c=3, plane=0]". */
  name: string;
  value: number;
  description?: string;
  /** Optional formula line, e.g. "score = q·k / √d + bias". */
  formula?: string;
  /** Optional arithmetic breakdown shown as label = value rows. */
  terms?: ScalarTerm[];
}

function fmt(v: number): string {
  if (!Number.isFinite(v)) return String(v);
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e4 || a < 1e-3) return v.toExponential(4);
  return v.toPrecision(7);
}

export function ScalarInspector({ view }: { view: ScalarView | null }) {
  return (
    <div className="scalar-inspector">
      <div className="panel-title">Scalar</div>
      {!view ? (
        <div className="scalar-empty">Hover or click a cell to inspect a single number.</div>
      ) : (
        <>
          <div className="scalar-name">{view.name}</div>
          <div className="scalar-value">{fmt(view.value)}</div>
          {view.description && <div className="scalar-desc">{view.description}</div>}
          {view.formula && <div className="scalar-formula">{view.formula}</div>}
          {view.terms && view.terms.length > 0 && (
            <table className="scalar-terms">
              <tbody>
                {view.terms.map((t, i) => (
                  <tr key={i}>
                    <td className="scalar-term-label">{t.label}</td>
                    <td className="scalar-term-val">{fmt(t.value)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </>
      )}
    </div>
  );
}
