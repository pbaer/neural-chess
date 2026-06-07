// model-graph walk tests — driven off the REAL hero capsule (version-neutral:
// the test asserts structure derived from the capsule, not hardcoded counts).

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { buildModelGraph, shapeSummary, idIndex } from './graph.ts';
import type { CapsuleManifest } from '../engine/capsule.ts';

const manifest = JSON.parse(
  readFileSync(fileURLToPath(new URL('../../../public/weights/v3.1-nano/capsule.json', import.meta.url)), 'utf-8'),
) as CapsuleManifest;

describe('buildModelGraph', () => {
  const g = buildModelGraph(manifest);

  it('produces one node per graph stage, indexed by id', () => {
    expect(g.nodes.length).toBe(manifest.graph.length);
    for (const stage of manifest.graph) expect(g.byId.get(stage.id)?.kind).toBe(stage.kind);
  });

  it('sums each stage’s parameters from the tensor index', () => {
    const lenByName = new Map(manifest.tensors.map((t) => [t.name, t.length] as const));
    for (const node of g.nodes) {
      const expected = node.weightNames.reduce((s, w) => s + (lenByName.get(w) ?? 0), 0);
      expect(node.params).toBe(expected);
    }
  });

  it('totalParams equals the sum of all stage params', () => {
    expect(g.totalParams).toBe(g.nodes.reduce((s, n) => s + n.params, 0));
    // And matches the concatenated weight count (stored_floats), since every
    // tensor belongs to exactly one stage in this capsule.
    const allTensorFloats = manifest.tensors.reduce((s, t) => s + t.length, 0);
    expect(g.totalParams).toBe(allTensorFloats);
  });

  it('numbers repeated block stages in their label', () => {
    const block0 = g.nodes.find((n) => n.kind === 'block');
    expect(block0).toBeDefined();
    expect(block0!.label).toMatch(/Block \d+/);
  });

  it('preserves connectivity (reads) from the capsule', () => {
    for (const stage of manifest.graph) {
      expect(g.byId.get(stage.id)!.reads).toEqual(stage.reads);
    }
  });
});

describe('helpers', () => {
  it('idIndex extracts a trailing integer', () => {
    expect(idIndex('block.7')).toBe(7);
    expect(idIndex('tokenize')).toBeNull();
    expect(idIndex('head.policy')).toBeNull();
  });

  it('shapeSummary renders a kind-appropriate line', () => {
    const g = buildModelGraph(manifest);
    const planes = g.nodes.find((n) => n.kind === 'input_planes')!;
    expect(shapeSummary(planes)).toContain('×');
    const block = g.nodes.find((n) => n.kind === 'block')!;
    expect(shapeSummary(block)).toContain('h');
  });
});
