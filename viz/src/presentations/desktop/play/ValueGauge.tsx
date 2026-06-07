// ValueGauge — renders the model's value-head output as a White-vs-Black balance.
// The network emits tanh ∈ [−1,1] from the SIDE-TO-MOVE's perspective at the
// moment it moved; here we re-express it in a fixed frame: + favors White, −
// favors Black. The bar is centered at 0 (dead even) and grows toward whichever
// side is ahead — right for White, left for Black.

import type { Color, ModelMoveInfo } from '../../../core/index.ts';

function strength(mag: number): string {
  return mag > 0.6 ? 'winning' : mag > 0.25 ? 'clearly better' : 'slightly better';
}

export function ValueGauge({ model, modelColor }: { model: ModelMoveInfo | null; modelColor: Color }) {
  const v = model?.value ?? 0;
  // Re-frame side-to-move value into White(+)/Black(−). The model moved as
  // `modelColor`, so its own +1 means White ahead iff it was White.
  const whiteV = modelColor === 'w' ? v : -v;
  const mag = Math.min(1, Math.abs(whiteV));
  const halfPct = mag * 50; // each side fills up to half the track from the center
  const favored: Color | null = whiteV > 0.05 ? 'w' : whiteV < -0.05 ? 'b' : null;

  return (
    <div className="value-gauge">
      <div className="value-head">
        <span className="value-title">Value head — White vs Black</span>
        <span className="value-num">{model ? (whiteV >= 0 ? '+' : '') + whiteV.toFixed(2) : '—'}</span>
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
        {model ? (
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
