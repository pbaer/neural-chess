// ContentCard — renders a kind-keyed explanation card (what / how / why /
// principle) plus a generic dims line the caller supplies. Version-neutral: it
// only knows op/stage KINDS, never an architecture. Falls back gracefully for a
// kind with no authored card.

import { content } from '../../../core/index.ts';

export interface ContentCardProps {
  kind: string;
  /** Generic shape/where-it-sits line, built by the caller from the graph. */
  dims?: string;
  compact?: boolean;
}

export function ContentCard({ kind, dims, compact }: ContentCardProps) {
  const card = content(kind);
  if (!card) {
    return (
      <div className="content-card">
        <div className="content-title">{kind}</div>
        <div className="content-what">No explanation authored for this kind yet.</div>
        {dims && <div className="content-dims">{dims}</div>}
      </div>
    );
  }
  return (
    <div className="content-card">
      <div className="content-title">{card.title}</div>
      <p className="content-what">{card.what}</p>
      {!compact && (
        <>
          <Slot label="How" text={card.how} />
          <Slot label="Why" text={card.why} />
          {card.principle && <Slot label="Principle" text={card.principle} principle />}
        </>
      )}
      {dims && <div className="content-dims">{dims}</div>}
    </div>
  );
}

function Slot({ label, text, principle }: { label: string; text: string; principle?: boolean }) {
  return (
    <div className={'content-slot' + (principle ? ' content-principle' : '')}>
      <span className="content-slot-label">{label}</span>
      <span className="content-slot-text">{text}</span>
    </div>
  );
}
