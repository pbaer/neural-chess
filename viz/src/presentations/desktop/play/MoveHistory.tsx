// MoveHistory — SAN move list in numbered pairs (chess.js history).

export function MoveHistory({ sanHistory }: { sanHistory: string[] }) {
  const rows: Array<{ n: number; w: string; b: string }> = [];
  for (let i = 0; i < sanHistory.length; i += 2) {
    rows.push({ n: i / 2 + 1, w: sanHistory[i], b: sanHistory[i + 1] ?? '' });
  }
  return (
    <div className="move-history">
      <div className="panel-title">Moves</div>
      <ol className="move-list">
        {rows.length === 0 && <li className="move-empty">No moves yet.</li>}
        {rows.map((row) => (
          <li key={row.n} className="move-row">
            <span className="move-num">{row.n}.</span>
            <span className="move-san">{row.w}</span>
            <span className="move-san">{row.b}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}
