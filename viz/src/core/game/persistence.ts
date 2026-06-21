// Game-state persistence across a page refresh (localStorage).
//
// We persist ONLY the game (the position, as a start FEN + the move list, plus
// the human's color and the play settings) — NOT the model/worker, which reloads
// fresh. On startup the game is replayed back into chess.js.
//
// ─────────────────────────────────────────────────────────────────────────────
//  FORWARD / BACKWARD COMPATIBILITY CONTRACT  (read before changing this format)
// ─────────────────────────────────────────────────────────────────────────────
// This blob lives in the user's browser and can OUTLIVE the code that wrote it:
// a visitor may have a game in progress, then refresh into a newly deployed
// version of the app. Restore therefore MUST tolerate older — and unknown —
// shapes without crashing or silently corrupting the game. Rules for evolving it:
//
//   1. The stable CORE is { startFen, moves[] } — plain chess that never changes
//      meaning. Keep it. Restore is "replay `moves` from `startFen`".
//   2. Everything else (settings) is OPTIONAL: read each field with a DEFAULT, so
//      a blob written by an older version (missing the field) still loads, and a
//      blob from a newer version (extra/unknown fields) is read field-by-field
//      and the extras ignored — never throw on a key you don't recognise.
//   3. Only bump SCHEMA for a *breaking* change to the CORE. On a schema newer
//      than we understand, or any parse/validation failure, return null → start a
//      fresh game (lose the saved one gracefully rather than crash).
//   4. Prefer ADDITIVE changes (new optional fields w/ defaults) over breaking
//      ones. If you must break the CORE, write a migration from the old shape.
//
// Tests: parsePersisted() is pure (no storage) so the compat rules above are unit
// -tested directly in persistence.test.ts. Keep those tests green when evolving.

import type { Color } from '../engine/index.ts';
import type { MctsSettings } from './gameStore.ts';

const KEY = 'neural-chess:game';
/** Bump ONLY for a breaking change to the CORE { startFen, moves }. */
const SCHEMA = 1;

export interface PersistedMove {
  from: string;
  to: string;
  promotion?: string;
}

export interface PersistedGame {
  /** Position the game started from (null = standard initial position). */
  startFen: string | null;
  /** Moves played from `startFen`, in order. The board is restored by replaying. */
  moves: PersistedMove[];
  humanColor: Color;
  assist: boolean;
  variety: number;
  mcts: MctsSettings;
}

/** Defaults the caller supplies so settings fall back cleanly (avoids a runtime
 *  import cycle with gameStore's DEFAULT_* constants). */
export interface RestoreDefaults {
  humanColor: Color;
  assist: boolean;
  variety: number;
  mcts: MctsSettings;
}

/** localStorage if usable, else null (node/SSR, private mode, sandboxed iframe). */
function storage(): Storage | null {
  try {
    if (typeof localStorage === 'undefined') return null;
    return localStorage;
  } catch {
    return null; // access itself can throw when storage is disabled
  }
}

function num(x: unknown, fallback: number): number {
  return typeof x === 'number' && Number.isFinite(x) ? x : fallback;
}

/**
 * Pure validator/normaliser (no storage) — the heart of the compat contract.
 * Takes already-JSON-parsed input and returns a fully-defaulted PersistedGame,
 * or null if the CORE is missing/malformed or the schema is from the future.
 */
export function parsePersisted(o: unknown, d: RestoreDefaults): PersistedGame | null {
  if (!o || typeof o !== 'object') return null;
  const r = o as Record<string, unknown>;
  // A schema newer than this build understands → discard gracefully.
  if (typeof r.schema === 'number' && r.schema > SCHEMA) return null;
  // CORE: moves[] must be present and well-formed (from/to strings).
  if (!Array.isArray(r.moves)) return null;
  const moves: PersistedMove[] = [];
  for (const m of r.moves) {
    if (!m || typeof m !== 'object') return null;
    const mm = m as Record<string, unknown>;
    if (typeof mm.from !== 'string' || typeof mm.to !== 'string') return null;
    moves.push({
      from: mm.from,
      to: mm.to,
      ...(typeof mm.promotion === 'string' ? { promotion: mm.promotion } : {}),
    });
  }
  const startFen = typeof r.startFen === 'string' ? r.startFen : null;
  // OPTIONAL settings → default when absent/invalid (old & new blobs both load).
  const mctsIn = (r.mcts && typeof r.mcts === 'object' ? r.mcts : {}) as Record<string, unknown>;
  return {
    startFen,
    moves,
    humanColor: r.humanColor === 'b' ? 'b' : r.humanColor === 'w' ? 'w' : d.humanColor,
    assist: typeof r.assist === 'boolean' ? r.assist : d.assist,
    variety: num(r.variety, d.variety),
    mcts: {
      enabled: typeof mctsIn.enabled === 'boolean' ? mctsIn.enabled : d.mcts.enabled,
      sims: num(mctsIn.sims, d.mcts.sims),
      cPuct: num(mctsIn.cPuct, d.mcts.cPuct),
      cutoffThreshold: num(mctsIn.cutoffThreshold, d.mcts.cutoffThreshold),
    },
  };
}

/** Load + validate the saved game, or null if none / unusable. */
export function loadPersisted(defaults: RestoreDefaults): PersistedGame | null {
  const s = storage();
  if (!s) return null;
  let raw: string | null;
  try {
    raw = s.getItem(KEY);
  } catch {
    return null;
  }
  if (!raw) return null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return null;
  }
  return parsePersisted(parsed, defaults);
}

export function savePersisted(g: PersistedGame): void {
  const s = storage();
  if (!s) return;
  try {
    s.setItem(KEY, JSON.stringify({ schema: SCHEMA, ...g }));
  } catch {
    // quota exceeded / serialization issue — non-fatal, just don't persist.
  }
}

export function clearPersisted(): void {
  const s = storage();
  if (!s) return;
  try {
    s.removeItem(KEY);
  } catch {
    // ignore
  }
}
