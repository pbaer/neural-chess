// traceIndex tests — the trace keys a node exposes must match the names the
// engine actually records (model.ts / attention.ts). We assert against a live
// trace captured from the real engine so the join can never silently drift.

import { describe, it, expect } from 'vitest';
import { traceFieldsFor, outputTraceKey } from './traceIndex.ts';
import { buildModelGraph } from './graph.ts';
import { loadEngine } from '../../../tests/parity/fixtures.ts';
import { featurize, TraceRecorder } from '../engine/index.ts';
import type { CapsuleManifest } from '../engine/capsule.ts';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { Chess } from 'chess.js';
import { chessToBoardState } from '../game/chessAdapter.ts';

const manifest = JSON.parse(
  readFileSync(fileURLToPath(new URL('../../../public/weights/v3.1-nano/capsule.json', import.meta.url)), 'utf-8'),
) as CapsuleManifest;

function captureTraceKeys(): Set<string> {
  const engine = loadEngine();
  const chess = new Chess();
  const board = chessToBoardState(chess, { epSquare: null, isRepetition: () => false });
  const planes = featurize(board, 'float');
  const rec = new TraceRecorder({ granularity: 'full' });
  engine.forward(planes, { trace: rec });
  return new Set(rec.finalize().entries.keys());
}

describe('traceFieldsFor', () => {
  const g = buildModelGraph(manifest);
  const recorded = captureTraceKeys();

  it('every declared trace field exists in a real engine trace', () => {
    for (const node of g.nodes) {
      for (const f of traceFieldsFor(node)) {
        expect(recorded.has(f.key), `missing trace key ${f.key} for ${node.id}`).toBe(true);
      }
    }
  });

  it('a block exposes attention scores + probs internals', () => {
    const block = g.nodes.find((n) => n.kind === 'block')!;
    const keys = traceFieldsFor(block).map((f) => f.key);
    expect(keys).toContain(`${block.id}.attn.scores`);
    expect(keys).toContain(`${block.id}.attn.probs`);
  });

  it('each stage has exactly one output field, and it is recorded', () => {
    const recordedKeys = captureTraceKeys();
    for (const node of g.nodes) {
      const out = outputTraceKey(node);
      if (traceFieldsFor(node).length === 0) {
        expect(out).toBeNull();
        continue;
      }
      expect(out, `no output for ${node.id}`).not.toBeNull();
      expect(recordedKeys.has(out!)).toBe(true);
    }
  });
});
