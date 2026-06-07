// traceStore tests — version bumps, O(1) indexed lookup, subscribe/clear.

import { describe, it, expect, beforeEach } from 'vitest';
import { traceStore, type TracePayload, type TraceMeta } from './traceStore.ts';

const meta: TraceMeta = { fen: 'startpos', turn: 'w', value: 0.1, bestLegalIndex: 5 };

function payload(): TracePayload {
  return {
    entries: [
      { name: 'planes', data: new Float32Array([1, 2, 3]), shape: [3] },
      { name: 'block.0.attn.probs', data: new Float32Array([0.25, 0.75]), shape: [2] },
    ],
  };
}

describe('traceStore', () => {
  beforeEach(() => traceStore.clear());

  it('bumps version and exposes the snapshot on set', () => {
    const v0 = traceStore.version();
    traceStore.set(payload(), meta);
    expect(traceStore.version()).toBe(v0 + 1);
    expect(traceStore.snapshot()?.meta.fen).toBe('startpos');
  });

  it('indexes entries by name for O(1) lookup', () => {
    traceStore.set(payload(), meta);
    expect(traceStore.has('block.0.attn.probs')).toBe(true);
    expect(traceStore.entry('block.0.attn.probs')?.data[1]).toBeCloseTo(0.75);
    expect(traceStore.entry('missing')).toBeUndefined();
  });

  it('notifies subscribers and supports unsubscribe', () => {
    let hits = 0;
    const off = traceStore.subscribe(() => hits++);
    traceStore.set(payload(), meta);
    traceStore.set(payload(), meta);
    expect(hits).toBe(2);
    off();
    traceStore.set(payload(), meta);
    expect(hits).toBe(2);
  });

  it('clear drops the trace and bumps version', () => {
    traceStore.set(payload(), meta);
    const v = traceStore.version();
    traceStore.clear();
    expect(traceStore.snapshot()).toBeNull();
    expect(traceStore.has('planes')).toBe(false);
    expect(traceStore.version()).toBe(v + 1);
  });

  it('clear is a no-op (no version bump) when already empty', () => {
    const v = traceStore.version();
    traceStore.clear();
    expect(traceStore.version()).toBe(v);
  });
});
