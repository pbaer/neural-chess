// useTelescope — wires the visualization to the live game + worker.
//
//  * Loads a main-thread Capsule (graph + weights) so weight heatmaps and the
//    version-neutral DAG have real data (the worker keeps its own copy for
//    inference; the capsule is tiny).
//  * On each position change (while `enabled`) runs ONE traced forward via the
//    worker and publishes it to the non-reactive traceStore. The play loop
//    itself never requests a trace, so it pays nothing with tracing disabled.
//  * Exposes the traceStore version as a reactive value via useSyncExternalStore.

import { useEffect, useState, useSyncExternalStore } from 'react';
import { Chess } from 'chess.js';
import {
  buildLegalMaskAndMap,
  buildModelGraph,
  chessToBoardState,
  epTargetFromFen,
  featurize,
  fetchCapsule,
  traceStore,
  type Capsule,
  type Color,
  type EngineClient,
  type ModelGraph,
} from '../../../core/index.ts';

/** Reactive handle to the latest trace version (re-renders on each new trace). */
export function useTraceVersion(): number {
  return useSyncExternalStore(traceStore.subscribe, traceStore.version, traceStore.version);
}

export interface TelescopeData {
  capsule: Capsule | null;
  graph: ModelGraph | null;
  ready: boolean;
  error: string | null;
}

export function useTelescope(
  client: EngineClient | null,
  capsuleUrl: string,
  fen: string,
  enabled: boolean,
): TelescopeData {
  const [capsule, setCapsule] = useState<Capsule | null>(null);
  const [graph, setGraph] = useState<ModelGraph | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Load the capsule on the main thread once the telescope is opened.
  useEffect(() => {
    if (!enabled || capsule) return;
    let alive = true;
    fetchCapsule(capsuleUrl)
      .then((c) => {
        if (!alive) return;
        setCapsule(c);
        setGraph(buildModelGraph(c.manifest));
      })
      .catch((e: Error) => alive && setError(e.message));
    return () => {
      alive = false;
    };
  }, [enabled, capsuleUrl, capsule]);

  // Run a traced forward whenever the position changes (telescope open + ready).
  useEffect(() => {
    if (!enabled || !client) return;
    let alive = true;
    (async () => {
      try {
        const chess = new Chess(fen);
        const turn = chess.turn() as Color;
        const ep = epTargetFromFen(fen);
        const board = chessToBoardState(chess, { epSquare: ep, isRepetition: () => false });
        const planes = featurize(board, 'float');
        const { mask } = buildLegalMaskAndMap(chess);
        const reply = await client.forward(planes, mask, true);
        if (!alive || !reply.trace) return;
        traceStore.set(reply.trace, {
          fen,
          turn,
          value: reply.value,
          bestLegalIndex: reply.bestLegalIndex,
        });
      } catch (e) {
        if (alive) setError(`Trace failed: ${(e as Error).message}`);
      }
    })();
    return () => {
      alive = false;
    };
  }, [enabled, client, fen]);

  // Clear the trace when the telescope unmounts (closed).
  useEffect(() => () => traceStore.clear(), []);

  return { capsule, graph, ready: !!capsule, error };
}
