// traceStore — a NON-reactive module singleton holding the latest forward-pass
// trace, publishing only a monotonic `version` to subscribers.
//
// The trace is multi-MB of typed-array activations; it must never live in
// reactive (immer/structural-sharing) state. Instead this singleton owns the
// immutable trace and bumps an integer `version` on each replacement. UI binds
// to `version` (O(1) reactivity) and pulls zero-copy Float32Array subviews via
// `entry(name)` (O(MB) data, untracked).
//
// Presentation-agnostic: no DOM, no React. A React presentation adapts it with
// useSyncExternalStore(subscribe, version); a future presentation can poll
// `version()` however it likes.

/** One recorded intermediate (matches the worker's SerializedTrace entries). */
export interface TraceEntryData {
  name: string;
  data: Float32Array;
  shape: number[];
}

/** The serialized trace shape produced by the inference worker. */
export interface TracePayload {
  entries: TraceEntryData[];
  bytes?: number;
  granularity?: string;
  capped?: boolean;
}

/** Side-channel metadata about the traced position (drives the readout panels). */
export interface TraceMeta {
  /** FEN of the position this trace was captured for. */
  fen: string;
  /** Side to move when the trace was captured ('w'|'b'); the trace is in the
   *  engine frame (moving-side-at-bottom), so this tells the UI whether the
   *  board was vertically mirrored. */
  turn: 'w' | 'b';
  /** Value head output in [-1,1], side-to-move perspective. */
  value: number;
  /** argmax-over-legal flat policy index, or -1. */
  bestLegalIndex: number;
}

export interface TraceSnapshot {
  payload: TracePayload;
  meta: TraceMeta;
}

type Listener = () => void;

let current: TraceSnapshot | null = null;
let index: Map<string, TraceEntryData> | null = null;
let version = 0;
const listeners = new Set<Listener>();

function reindex(): void {
  if (!current) {
    index = null;
    return;
  }
  const m = new Map<string, TraceEntryData>();
  for (const e of current.payload.entries) m.set(e.name, e);
  index = m;
}

export const traceStore = {
  /** Replace the current trace and notify subscribers (bumps `version`). */
  set(payload: TracePayload, meta: TraceMeta): void {
    current = { payload, meta };
    reindex();
    version += 1;
    for (const fn of listeners) fn();
  },

  /** Drop the current trace (e.g. when the telescope closes). */
  clear(): void {
    if (!current) return;
    current = null;
    index = null;
    version += 1;
    for (const fn of listeners) fn();
  },

  /** Monotonic version counter — the reactive handle. */
  version(): number {
    return version;
  },

  /** The full current snapshot (trace + meta), or null. */
  snapshot(): TraceSnapshot | null {
    return current;
  },

  /** Zero-copy lookup of one recorded intermediate by name, or undefined. */
  entry(name: string): TraceEntryData | undefined {
    return index?.get(name);
  },

  /** True if an intermediate by this name is present in the current trace. */
  has(name: string): boolean {
    return index?.has(name) ?? false;
  },

  /** Subscribe to version changes; returns an unsubscribe fn. */
  subscribe(fn: Listener): () => void {
    listeners.add(fn);
    return () => listeners.delete(fn);
  },
};

export type TraceStore = typeof traceStore;
