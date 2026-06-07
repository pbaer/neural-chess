// React bindings over the presentation-agnostic core: build the worker-backed
// engine client + game store inside an effect (so StrictMode's mount/cleanup/
// mount correctly disposes the first worker and uses the second), then expose a
// possibly-null store + a ready flag + model metadata. The store itself lives in
// core (React-free); this hook is the desktop presentation's adapter to it.

import { useEffect, useState } from 'react';
import {
  createGameStore,
  createWorkerEngineClient,
  type EngineClient,
  type EngineMeta,
  type GameStore,
} from '../../../core/index.ts';

const MODEL_ID = 'v3.1-nano';

/** Absolute URL of a model's capsule.json (respects the GH-Pages deploy base). */
export function capsuleUrlFor(modelId: string = MODEL_ID): string {
  // public/weights/<id>/capsule.json, resolved to an ABSOLUTE url so the worker
  // (and capsule.ts's relative weights.bin lookup) can fetch it.
  const base = import.meta.env.BASE_URL ?? '/';
  return new URL(`${base}weights/${modelId}/capsule.json`, location.href).href;
}

export interface UseGame {
  store: GameStore | null;
  /** The same worker-backed client driving the store — reused by the telescope. */
  client: EngineClient | null;
  capsuleUrl: string;
  ready: boolean;
  meta: EngineMeta | null;
  error: string | null;
}

export function useGame(): UseGame {
  const [store, setStore] = useState<GameStore | null>(null);
  const [client, setClient] = useState<EngineClient | null>(null);
  const [ready, setReady] = useState(false);
  const [meta, setMeta] = useState<EngineMeta | null>(null);
  const [error, setError] = useState<string | null>(null);
  const url = capsuleUrlFor();

  useEffect(() => {
    const c = createWorkerEngineClient(url);
    setClient(c);
    setStore(createGameStore(c, 'w'));
    let alive = true;
    c
      .whenReady()
      .then((m) => {
        if (!alive) return;
        setMeta(m);
        setReady(true);
      })
      .catch((e: Error) => alive && setError(e.message));
    return () => {
      alive = false;
      setReady(false);
      setMeta(null);
      setClient(null);
      c.dispose();
    };
  }, [url]);

  return { store, client, capsuleUrl: url, ready, meta, error };
}
