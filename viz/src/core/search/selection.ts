// Value-adaptive, bounded move-selection variety — the shared FINAL move picker
// for BOTH play modes (One-Shot policy + MCTS visit counts).
//
// The contract (presentation-agnostic, pure):
//   Given a move distribution D over legal moves, the position value V (side-to-
//   move, [-1,1]), and a user "Move variety" setting S in [0,1], pick ONE move
//   such that:
//     1. We never leave the "reasonable set" — the model's own good moves — so we
//        never play a move a strong player would call bad, no matter how high S is
//        or how winning V is (a hard floor).
//     2. We sample ∝ D^(1/T) over that reasonable set under a VALUE-ADAPTIVE
//        temperature T(V,S): T → 0 when losing (sharpen to the best move, maximise
//        strength), T ≈ baseline(S) when even (interesting variety), T larger when
//        winning (give the human a chance) — but always bounded by the set.
//     3. At S=0 selection is fully deterministic (always the top move).
//
// This is NOT a chess heuristic (Principle-3 safe): it consumes only the model's
// own policy/value (or the search's own visit/Q stats). It adds no chess knowledge.

/** One candidate move in the distribution D. */
export interface SelectionCandidate {
  /** Distribution mass: policy prob (One-Shot) or visit count/fraction (MCTS).
   *  Need not be normalized — only relative magnitudes matter. */
  weight: number;
  /** Optional side-to-move value for THIS move, in [-1,1]. MCTS supplies the
   *  backed-up child Q (root mover's view); used for the Q-margin floor so the
   *  search never samples a move it judged losing. One-Shot omits it. */
  q?: number;
}

/** Tunable selection constants (the value→temperature curve + reasonable-set floor). */
export interface SelectionConfig {
  /** Reasonable set: at most this many top candidates by weight. */
  topK: number;
  /** Reasonable set: drop candidates whose weight < relFloor · (top weight). */
  relFloor: number;
  /** Reasonable set (MCTS only): drop candidates whose q < (best q) − qMargin. */
  qMargin: number;
  /** Baseline temperature at S=1, V=0 (even position, max variety slider). */
  baseTemp: number;
  /** Temperature gain at V=+1 (fully winning) relative to V=0. */
  winGain: number;
}

/**
 * Defaults (the chosen curve):
 *  - Reasonable set = top-5 policy/visit moves with weight ≥ 12% of the top move's,
 *    and (MCTS) Q within 0.10 of the best — so candidates are always moves the model
 *    actually likes; everything else is unreachable.
 *  - T(V,S) = S · baseTemp · g(V), with g(0)=1, g(+1)=winGain. The losing side is
 *    g(V)=(1+V)² (quadratic) so it sharpens FAST as the model falls behind —
 *    "clearly lost" play is effectively deterministic (top move); the winning side
 *    is linear up to winGain. baseTemp 0.7, winGain 2.5 → T ranges 0 (losing / S=0)
 *    up to ≈1.75 (fully winning at S=1). P(top) highest when losing, lowest winning.
 */
export const DEFAULT_SELECTION_CONFIG: SelectionConfig = {
  topK: 5,
  relFloor: 0.12,
  qMargin: 0.1,
  baseTemp: 0.7,
  winGain: 2.5,
};

/** Inputs that vary per move. */
export interface SelectionParams {
  /** Position value V (side-to-move perspective), in [-1,1]. */
  value: number;
  /** User "Move variety" setting S, in [0,1]. 0 = deterministic top move. */
  variety: number;
  /** RNG (default Math.random); injected for deterministic tests. */
  rng?: () => number;
}

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

/**
 * Value-adaptive sampling temperature T(V,S).
 *  - S=0 → 0 (deterministic, regardless of V).
 *  - V=−1 (losing) → 0 (sharpen to the best move, maximise strength).
 *  - V=0  (even)   → S · baseTemp (interesting variety).
 *  - V=+1 (winning)→ S · baseTemp · winGain (more spread; give the human a chance).
 * Monotonically non-decreasing in V (more winning ⇒ more variety) and in S.
 */
export function valueAdaptiveTemperature(
  value: number,
  variety: number,
  cfg: SelectionConfig = DEFAULT_SELECTION_CONFIG,
): number {
  const S = clamp(variety, 0, 1);
  const V = clamp(value, -1, 1);
  if (S <= 0) return 0;
  // g(V): 0 at V=−1, 1 at V=0, winGain at V=+1; monotonic. Quadratic on the losing
  // side so strength is maximised (near-deterministic) once clearly behind.
  const g = V <= 0 ? (1 + V) * (1 + V) : 1 + V * (cfg.winGain - 1);
  return S * cfg.baseTemp * Math.max(0, g);
}

/**
 * The bounded "reasonable set": indices (into `candidates`) of the moves we are
 * ever willing to play. Top-K by weight, gated by a relative-weight floor and (if
 * q is present) a Q-margin floor. The single top-weight move is ALWAYS included so
 * the hard floor holds even if its own Q dips below the margin.
 */
export function reasonableSetIndices(
  candidates: readonly SelectionCandidate[],
  cfg: SelectionConfig = DEFAULT_SELECTION_CONFIG,
): number[] {
  const n = candidates.length;
  if (n === 0) return [];

  let topIdx = 0;
  for (let i = 1; i < n; i++) if (candidates[i].weight > candidates[topIdx].weight) topIdx = i;
  const topWeight = candidates[topIdx].weight;

  let bestQ = -Infinity;
  let haveQ = false;
  for (const c of candidates) {
    if (c.q !== undefined) {
      haveQ = true;
      if (c.q > bestQ) bestQ = c.q;
    }
  }

  const order = [...candidates.keys()].sort((a, b) => candidates[b].weight - candidates[a].weight);
  const set: number[] = [];
  for (const idx of order) {
    if (set.length >= cfg.topK) break;
    const c = candidates[idx];
    if (topWeight > 0 && c.weight < cfg.relFloor * topWeight) continue;
    if (haveQ && c.q !== undefined && c.q < bestQ - cfg.qMargin) continue;
    set.push(idx);
  }
  if (!set.includes(topIdx)) set.unshift(topIdx);
  return set;
}

/**
 * Pick ONE candidate (returns its index, or −1 if none). Restricts to the
 * reasonable set, then samples ∝ weight^(1/T) under the value-adaptive T. At T≈0
 * (losing, or S=0) returns the top-weight move deterministically.
 */
export function selectMoveIndex(
  candidates: readonly SelectionCandidate[],
  params: SelectionParams,
  cfg: SelectionConfig = DEFAULT_SELECTION_CONFIG,
): number {
  const set = reasonableSetIndices(candidates, cfg);
  if (set.length === 0) return -1;

  // The top-weight move within the set (the deterministic / argmax choice).
  let topIdx = set[0];
  for (const idx of set) if (candidates[idx].weight > candidates[topIdx].weight) topIdx = idx;
  if (set.length === 1) return topIdx;

  const T = valueAdaptiveTemperature(params.value, params.variety, cfg);
  if (T <= 1e-3) return topIdx;

  const rng = params.rng ?? Math.random;
  const weights = set.map((idx) => Math.pow(Math.max(candidates[idx].weight, 0), 1 / T));
  let total = 0;
  for (const w of weights) total += w;
  if (!(total > 0)) return topIdx;

  let r = rng() * total;
  for (let k = 0; k < set.length; k++) {
    r -= weights[k];
    if (r <= 0) return set[k];
  }
  return set[set.length - 1];
}
