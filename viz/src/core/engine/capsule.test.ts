// fetchCapsule tests — HTTP failures must surface as clear errors naming the
// URL and status (a raw 404 body would otherwise die as a confusing JSON parse
// error), and a good response must load into a working Capsule.

import { describe, it, expect, vi, afterEach } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { fetchCapsule } from './capsule.ts';

const CAPSULE_URL = 'https://example.test/weights/v3.1-nano/capsule.json';
const WEIGHTS_URL = 'https://example.test/weights/v3.1-nano/weights.bin';

const fixture = (name: string) =>
  readFileSync(fileURLToPath(new URL(`../../../public/weights/v3.1-nano/${name}`, import.meta.url)));

afterEach(() => vi.unstubAllGlobals());

describe('fetchCapsule', () => {
  it('throws a clear error naming URL and status when the manifest fetch fails', async () => {
    vi.stubGlobal('fetch', async () => new Response('Not Found', { status: 404 }));
    await expect(fetchCapsule(CAPSULE_URL)).rejects.toThrow(`Failed to fetch capsule manifest ${CAPSULE_URL}: HTTP 404.`);
  });

  it('throws a clear error naming URL and status when the weights fetch fails', async () => {
    vi.stubGlobal('fetch', async (url: string | URL) =>
      String(url) === CAPSULE_URL
        ? new Response(fixture('capsule.json'), { status: 200 })
        : new Response('Server Error', { status: 500 }),
    );
    await expect(fetchCapsule(CAPSULE_URL)).rejects.toThrow(`Failed to fetch capsule weights ${WEIGHTS_URL}: HTTP 500.`);
  });

  it('loads a Capsule (weights.bin resolved relative to the manifest) on success', async () => {
    const fetched: string[] = [];
    vi.stubGlobal('fetch', async (url: string | URL) => {
      fetched.push(String(url));
      return String(url) === CAPSULE_URL
        ? new Response(fixture('capsule.json'), { status: 200 })
        : new Response(fixture('weights.bin'), { status: 200 });
    });
    const capsule = await fetchCapsule(CAPSULE_URL);
    expect(fetched).toEqual([CAPSULE_URL, WEIGHTS_URL]);
    expect(capsule.config.d_model).toBeGreaterThan(0);
    expect(capsule.graph.length).toBeGreaterThan(0);
  });
});
