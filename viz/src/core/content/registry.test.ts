// Content registry tests — every stage kind present in the hero capsule must
// have an authored explanation card (so the telescope never shows a blank).

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { content, CONTENT } from './registry.ts';
import type { CapsuleManifest } from '../engine/capsule.ts';

const manifest = JSON.parse(
  readFileSync(fileURLToPath(new URL('../../../public/weights/v3.1-nano/capsule.json', import.meta.url)), 'utf-8'),
) as CapsuleManifest;

describe('content registry', () => {
  it('covers every stage kind in the capsule graph', () => {
    const kinds = new Set(manifest.graph.map((s) => s.kind));
    for (const kind of kinds) {
      expect(content(kind), `no content card for kind "${kind}"`).toBeDefined();
    }
  });

  it('covers the op-level kinds used by the math lens', () => {
    for (const op of ['attention', 'softmax', 'ffn', 'residual']) {
      expect(content(op)).toBeDefined();
    }
  });

  it('every card fills the required slots', () => {
    for (const [kind, card] of Object.entries(CONTENT)) {
      expect(card.title, `${kind}.title`).toBeTruthy();
      expect(card.what, `${kind}.what`).toBeTruthy();
      expect(card.how, `${kind}.how`).toBeTruthy();
      expect(card.why, `${kind}.why`).toBeTruthy();
    }
  });

  it('returns undefined for an unknown kind', () => {
    expect(content('nonexistent_kind')).toBeUndefined();
  });
});
