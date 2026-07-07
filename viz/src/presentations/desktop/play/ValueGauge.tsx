// ValueGauge — renders the model's value-head output as a White-vs-Black balance.
// The network emits tanh ∈ [−1,1] from the SIDE-TO-MOVE's perspective at the
// moment it moved; the store re-expresses it in a fixed White(+)/Black(−) frame
// (see ValueSample) and hands us that `whiteValue` directly — so the reading
// survives a refresh / undo just like the move-list history does. The bar is
// centered at 0 (dead even) and grows toward whichever side is ahead — right for
// White, left for Black. `null` = the model hasn't moved yet.

import type { Color } from '../../../core/index.ts';

function strength(mag: number): string {
  return mag > 0.6 ? 'winning' : mag > 0.25 ? 'clearly better' : 'slightly better';
}

export function ValueGauge({ whiteValue }: { whiteValue: number | null }) {
  const has = whiteValue !== null;
  const whiteV = whiteValue ?? 0;
  const mag = Math.min(1, Math.abs(whiteV));
  const halfPct = mag * 50; // each side fills up to half the track from the center
  const favored: Color | null = whiteV > 0.05 ? 'w' : whiteV < -0.05 ? 'b' : null;

  return (
    <div className="value-gauge">
      <div className="value-head">
        <span className="value-title">Value head: White vs Black</span>
        <span className="value-num">{has ? (whiteV >= 0 ? '+' : '') + whiteV.toFixed(2) : '—'}</span>
      </div>
      <div className="value-bar value-bar-split" aria-hidden>
        {whiteV >= 0 ? (
          <div className="value-fill-w" style={{ left: '50%', width: `${halfPct}%` }} />
        ) : (
          <div className="value-fill-b" style={{ left: `${50 - halfPct}%`, width: `${halfPct}%` }} />
        )}
        <div className="value-mid" />
      </div>
      <div className="value-ends">
        <span>← Black</span>
        <span>White →</span>
      </div>
      <div className="value-caption">
        {has ? (
          favored ? (
            <>
              the model thinks <strong>{favored === 'w' ? 'White' : 'Black'}</strong> is {strength(mag)}
            </>
          ) : (
            'the model judges the position roughly even'
          )
        ) : (
          'no model move yet'
        )}
      </div>
    </div>
  );
}
