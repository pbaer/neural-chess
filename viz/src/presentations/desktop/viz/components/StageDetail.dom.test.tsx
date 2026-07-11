// @vitest-environment jsdom
//
// StageDetail selection behavior: the default field selection must stay LIVE
// until the user explicitly picks a chip. A stage rendered before the first
// trace lands can only offer weights; once a trace arrives, the marquee
// activation (attn.probs for a block) must take over — while an explicit user
// pick must survive later trace updates.

import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, beforeEach } from 'vitest';
import { Chess } from 'chess.js';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { StageDetail } from './StageDetail.tsx';
import { buildModelGraph, featurize, traceStore } from '../../../../core/index.ts';
import { createEngineFromBytes, TraceRecorder } from '../../../../core/engine/index.ts';
import { loadCapsuleFromBytes, type CapsuleManifest } from '../../../../core/engine/capsule.ts';
import { chessToBoardState } from '../../../../core/game/chessAdapter.ts';

// import.meta.url is not a file: URL under jsdom, so resolve from the vitest
// root (viz/) instead.
const WEIGHTS_DIR = join(process.cwd(), 'public/weights/v3.1-nano');
const manifest = JSON.parse(readFileSync(join(WEIGHTS_DIR, 'capsule.json'), 'utf-8')) as CapsuleManifest;

function weightsBytes(): ArrayBuffer {
  const buf = readFileSync(join(WEIGHTS_DIR, 'weights.bin'));
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
}

/** Run one real traced forward on the start position and publish it. */
function publishTrace(): void {
  const engine = createEngineFromBytes(manifest, weightsBytes());
  const chess = new Chess();
  const board = chessToBoardState(chess, { epSquare: null, isRepetition: () => false });
  const planes = featurize(board, 'float');
  const rec = new TraceRecorder({ granularity: 'full' });
  const res = engine.forward(planes, { trace: rec });
  const trace = rec.finalize();
  const entries = [...trace.entries].map(([name, e]) => ({ name, data: e.data, shape: e.shape }));
  traceStore.set({ entries }, { fen: chess.fen(), turn: 'w', value: res.value, bestLegalIndex: res.bestLegalIndex ?? -1 });
}

const capsule = loadCapsuleFromBytes(manifest, weightsBytes());
const graph = buildModelGraph(manifest);
const block = graph.nodes.find((n) => n.kind === 'block')!;

const activeChip = () => document.querySelector('.chip-active');

describe('StageDetail default selection', () => {
  beforeEach(() => traceStore.clear());

  it('upgrades a pre-trace weight default to the marquee activation when the trace lands', () => {
    const { rerender } = render(<StageDetail node={block} capsule={capsule} traceVersion={traceStore.version()} />);
    // No trace yet: only weight chips exist, so the default selection is a weight.
    expect(screen.queryByText('activations')).not.toBeInTheDocument();
    expect(activeChip()).toHaveClass('chip-weight');

    publishTrace();
    rerender(<StageDetail node={block} capsule={capsule} traceVersion={traceStore.version()} />);
    // The default must upgrade to the block's marquee activation (attn.probs).
    expect(screen.getByText('activations')).toBeInTheDocument();
    expect(activeChip()).toHaveTextContent('probs');
  });

  it('keeps an explicit user pick across later trace updates', () => {
    publishTrace();
    const { rerender } = render(<StageDetail node={block} capsule={capsule} traceVersion={traceStore.version()} />);
    expect(activeChip()).toHaveTextContent('probs');

    // The user explicitly picks a weight chip...
    const weightChip = document.querySelector('.chip-weight') as HTMLButtonElement;
    fireEvent.click(weightChip);
    expect(activeChip()).toBe(weightChip);

    // ...and a fresh trace must NOT steal the selection back to the default.
    publishTrace();
    rerender(<StageDetail node={block} capsule={capsule} traceVersion={traceStore.version()} />);
    expect(activeChip()).toBe(document.querySelector('.chip-weight'));
    expect(activeChip()).toHaveClass('chip-weight');
  });
});
