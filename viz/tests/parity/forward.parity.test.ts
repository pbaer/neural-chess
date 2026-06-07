// Forward / op parity: feed the SAME golden planes into the TS engine and
// compare EVERY intermediate stage against the PyTorch fp32-eval golden.
// This isolates arithmetic from featurization.
//
// Tolerances (per IMPLEMENTATION_PLAN §3):
//   per-op stage max-abs   < 1e-4
//   final policy logits     max-abs < 1e-3 AND cosine > 0.99999
//   value                   abs < 1e-4
//   argmax-over-legal index IDENTICAL

import { describe, it, expect } from 'vitest';
import { TraceRecorder, NUM_MOVES } from '../../src/core/engine/index.ts';
import { loadEngine, loadGolden, maxAbsDiff, cosine } from './fixtures.ts';

const STAGE_TOL = 1e-4;
const LOGIT_MAXABS_TOL = 1e-3;
const LOGIT_COSINE_TOL = 0.99999;
const VALUE_TOL = 1e-4;

const engine = loadEngine();
const { golden, tensor } = loadGolden();

// Stage tensor keys we compare op-by-op (everything in golden.tensors except the
// featurize-only 'planesTrained' and the policy head, which gets its own gate).
function stageKeys(caseTensors: Record<string, unknown>): string[] {
  return Object.keys(caseTensors).filter((k) => k !== 'planesTrained' && k !== 'policy_logits');
}

describe('forward parity (TS engine vs PyTorch golden)', () => {
  for (const c of golden.cases) {
    describe(c.name, () => {
      const planes = tensor(c.tensors.planes);
      const legalMask = new Uint8Array(NUM_MOVES);
      for (const idx of c.legalIndices) legalMask[idx] = 1;

      const recorder = new TraceRecorder({ granularity: 'full' });
      const res = engine.forward(planes, { legalMask, trace: recorder });
      const trace = res.trace!;

      it('captures all expected stages', () => {
        for (const key of stageKeys(c.tensors)) {
          expect(trace.entries.has(key), `missing trace stage "${key}"`).toBe(true);
        }
      });

      for (const key of stageKeys(golden.cases[0].tensors)) {
        it(`stage ${key} max-abs < ${STAGE_TOL}`, () => {
          const g = tensor(c.tensors[key]);
          const t = trace.entries.get(key)!;
          expect(t.data.length).toBe(g.length);
          const d = maxAbsDiff(t.data, g);
          expect(d, `stage ${key} max-abs ${d}`).toBeLessThan(STAGE_TOL);
        });
      }

      it('policy logits: max-abs < 1e-3 and cosine > 0.99999', () => {
        const g = tensor(c.tensors.policy_logits);
        const d = maxAbsDiff(res.policyLogits, g);
        const cos = cosine(res.policyLogits, g);
        expect(d, `policy max-abs ${d}`).toBeLessThan(LOGIT_MAXABS_TOL);
        expect(cos, `policy cosine ${cos}`).toBeGreaterThan(LOGIT_COSINE_TOL);
      });

      it('value: abs < 1e-4', () => {
        const d = Math.abs(res.value - c.value);
        expect(d, `value abs ${d}`).toBeLessThan(VALUE_TOL);
      });

      it('argmax-over-legal move index is IDENTICAL', () => {
        expect(res.bestLegalIndex).toBe(c.bestLegalIndex);
      });
    });
  }
});
