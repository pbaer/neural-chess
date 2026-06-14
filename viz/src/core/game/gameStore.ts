// gameStore — the chess game + play-loop logic (presentation-agnostic).
//
// A vanilla Zustand store (React-free) so any presentation can bind to it.
// chess.js is the rules authority; the model only RANKS among chess.js's own
// legal moves. Play loop on the model's turn:
//   featurize(boardState,'float') → buildLegalMaskAndMap → worker.forward
//     → bestLegalIndex → indexToMove.get(idx) → chess.move(...)
// The index→move map performs the black-to-move un-rotation (sq^56) implicitly
// and robustly (it was built by rotating chess.js moves into the engine frame).

import { createStore } from 'zustand/vanilla';
import { Chess, validateFen, type Move as ChessMove, type Square } from 'chess.js';
import { featurize, type Color } from '../engine/index.ts';
import type { EngineClient } from './engineClient.ts';
import type { SearchOptions, SearchSnapshot } from '../search/types.ts';
import {
  algToIdx,
  idxToAlg,
  chessToBoardState,
  buildLegalMaskAndMap,
  epTargetAfterMove,
  epTargetFromFen,
} from './chessAdapter.ts';

export type PromotionPiece = 'q' | 'r' | 'b' | 'n';
// 'choosing' = pick-mode is on and the model is awaiting the user's pick among
// its top legal moves (no move applied yet).
export type GameStatus = 'idle' | 'thinking' | 'choosing' | 'over';

export interface ModelMoveInfo {
  /** UCI string, e.g. "g8f6" (+ promotion letter). */
  uci: string;
  san: string;
  fromIdx: number;
  toIdx: number;
  /** Value head output in [-1,1], side-to-move (the model's) perspective. */
  value: number;
  /** Renormalized policy probability of the chosen move. */
  policyProb: number;
}

/** One ranked legal move the model considered (pick-mode candidate list). */
export interface ModelCandidate {
  from: string; // algebraic, e.g. 'g1'
  to: string;
  promotion?: PromotionPiece;
  fromIdx: number;
  toIdx: number;
  uci: string;
  san: string;
  /** Renormalized policy probability over legal moves (the legal probs sum to 1). */
  prob: number;
}

/** How many top legal moves to surface in pick-mode. */
export const TOP_K_CANDIDATES = 5;

/**
 * Default move-selection temperature for auto-play. 0 = always the model's single
 * best move (deterministic); higher = more variety. ~0.6 keeps play recognisably
 * strong while making every game different — the "interesting" default.
 */
export const DEFAULT_TEMPERATURE = 0.6;

/**
 * Default MCTS ("think harder") settings. Tuned so a search lands within the
 * ≤1–2s per-move budget on the deployed nano model (forward is cheap, so a few
 * hundred sims fit). c_puct 1.5 is the project's tuned value; temperature 0 plays
 * the most-visited (strongest) move.
 */
export const MCTS_DEFAULTS = { enabled: false, sims: 320, timeMs: 1500, cPuct: 1.5, temperature: 0 } as const;

/**
 * How long (ms) to flash the move MCTS just chose — using the same gold/red
 * highlight the pick-mode picker shows on hover — before it lands on the board.
 * A short "here's the move I chose" beat after the search finishes.
 */
export const MCTS_FLASH_MS = 500;

/**
 * How long (ms) the non-MCTS auto-play path surfaces the model's TOP policy moves
 * as prob-weighted arrows before flashing+playing the chosen one. Presents a single
 * forward pass like "MCTS with essentially one simulation": show the move
 * distribution → flash the pick → play. Roughly matches MCTS_FLASH_MS so the two
 * paths feel consistent.
 */
export const AUTOPLAY_PREVIEW_MS = 500;

/** Live-adjustable MCTS settings. */
export interface MctsSettings {
  enabled: boolean;
  /** Simulation budget (hard cap). */
  sims: number;
  /** Wall-clock budget in ms (hard cap; whichever of sims/time hits first wins). */
  timeMs: number;
  /** PUCT exploration constant. */
  cPuct: number;
  /** Root selection temperature (0 = strongest/most-visited; >0 adds variety). */
  temperature: number;
}

/**
 * Pick a legal move index from the policy under a softmax temperature. T≈0 returns
 * the deterministic argmax (the model's top move); larger T samples more widely.
 * Operates on the RAW legal logits so the sampled distribution is the true policy.
 */
function sampleLegalIndex(
  logits: Float32Array,
  legalIndices: number[],
  bestLegalIndex: number,
  temperature: number,
  rng: () => number = Math.random,
): number {
  if (temperature <= 1e-3 || legalIndices.length <= 1) return bestLegalIndex;
  let maxLogit = -Infinity;
  for (const i of legalIndices) if (logits[i] > maxLogit) maxLogit = logits[i];
  const weights = legalIndices.map((i) => Math.exp((logits[i] - maxLogit) / temperature));
  let total = 0;
  for (const w of weights) total += w;
  let r = rng() * total;
  for (let k = 0; k < legalIndices.length; k++) {
    r -= weights[k];
    if (r <= 0) return legalIndices[k];
  }
  return legalIndices[legalIndices.length - 1];
}

export interface GameState {
  fen: string;
  turn: Color;
  humanColor: Color;
  /** SAN move list (chess.js history). */
  sanHistory: string[];
  status: GameStatus;
  resultText: string | null;
  inCheck: boolean;
  /** Highlight squares for the most recent move (either side), or null. */
  lastMove: { fromIdx: number; toIdx: number } | null;
  /** Details of the model's most recent move (for the value/move readout). */
  lastModelMove: ModelMoveInfo | null;
  /** When true, the model surfaces its top moves and waits for the user to pick. */
  pickMode: boolean;
  /** Softmax temperature for auto-play move selection (0 = deterministic top move). */
  temperature: number;
  /** Ranked top-K legal moves awaiting a pick (only set while status==='choosing'). */
  candidates: ModelCandidate[] | null;
  /**
   * The move MCTS just chose, briefly published BEFORE it's applied so the board
   * can flash it with the same highlight as a hovered pick-mode candidate. Null
   * except during that ~MCTS_FLASH_MS beat.
   */
  flashMove: ModelCandidate | null;
  /**
   * Top policy moves briefly surfaced (prob-weighted arrows) right before a
   * non-MCTS auto-play move lands — the "show the move distribution" beat that
   * mirrors MCTS's live arrows. Null except during that ~AUTOPLAY_PREVIEW_MS beat.
   */
  previewMoves: ModelCandidate[] | null;
  /** MCTS ("think harder") settings — see MctsSettings. */
  mcts: MctsSettings;
  /** Live (while thinking) or last-completed MCTS search snapshot, or null. */
  search: SearchSnapshot | null;
  error: string | null;

  // --- actions ---
  humanMove(fromIdx: number, toIdx: number, promotion?: PromotionPiece): boolean;
  newGame(humanColor?: Color): void;
  loadFen(fen: string): boolean;
  undo(): void;
  /** Toggle pick-mode. Turning it off mid-choice auto-plays the top suggestion. */
  setPickMode(on: boolean): void;
  /** Set the auto-play move-selection temperature (clamped ≥ 0). */
  setTemperature(t: number): void;
  /** Toggle MCTS ("think harder"). Off = one-shot argmax (today's behavior). */
  setMctsEnabled(on: boolean): void;
  /** Patch MCTS settings (sims / timeMs / cPuct / temperature). */
  setMctsSettings(patch: Partial<MctsSettings>): void;
  /** Play the i-th candidate (pick-mode). No-op unless status==='choosing'. */
  chooseModelMove(i: number): void;
  /** Legal destination indices for a piece on `fromIdx` (current side only). */
  legalTargets(fromIdx: number): number[];
}

export type GameStore = ReturnType<typeof createGameStore>;

export function createGameStore(engine: EngineClient, initialHumanColor: Color = 'w') {
  // Non-reactive engine state (kept out of the reactive snapshot).
  let chess = new Chess();
  let repCounts = new Map<string, number>();
  let epSquare: number | null = null;
  let lastMove: { fromIdx: number; toIdx: number } | null = null;
  // Ranked top-K candidates awaiting a pick + the model's value for that position
  // (pick-mode). Held as closure state and synced into the snapshot via commit().
  let candidates: ModelCandidate[] | null = null;
  let pendingValue = 0;
  // Monotonic token to drop stale async model replies after reset/newGame/load.
  let thinkToken = 0;

  const repKey = () => chess.fen().split(' ').slice(0, 4).join(' ');
  const recordPosition = () => repCounts.set(repKey(), (repCounts.get(repKey()) ?? 0) + 1);

  function boardState() {
    const count = repCounts.get(repKey()) ?? 0;
    return chessToBoardState(chess, { epSquare, isRepetition: (n) => count >= n });
  }

  function statusResult(): { status: GameStatus; resultText: string | null } {
    if (!chess.isGameOver()) return { status: 'idle', resultText: null };
    if (chess.isCheckmate()) {
      return { status: 'over', resultText: `Checkmate — ${chess.turn() === 'w' ? 'Black' : 'White'} wins` };
    }
    if (chess.isStalemate()) return { status: 'over', resultText: 'Draw — stalemate' };
    if (chess.isInsufficientMaterial()) return { status: 'over', resultText: 'Draw — insufficient material' };
    if (chess.isThreefoldRepetition()) return { status: 'over', resultText: 'Draw — threefold repetition' };
    return { status: 'over', resultText: 'Draw — fifty-move rule' };
  }

  const store = createStore<GameState>()((set, get) => {
    /** Build the reactive snapshot from current chess state + closure fields. */
    function commit(extra?: Partial<GameState>): void {
      const sr = statusResult();
      set({
        fen: chess.fen(),
        turn: chess.turn() as Color,
        sanHistory: chess.history(),
        status: sr.status,
        resultText: sr.resultText,
        inCheck: chess.inCheck(),
        lastMove,
        candidates,
        error: null,
        ...extra,
      });
    }

    /** Apply a fully-specified model move and publish its readout. */
    function applyModelMove(mv: { from: string; to: string; promotion?: PromotionPiece }, value: number, prob: number): void {
      const applied = chess.move({ from: mv.from, to: mv.to, promotion: mv.promotion ?? 'q' });
      epSquare = epTargetAfterMove(applied);
      lastMove = { fromIdx: algToIdx(mv.from), toIdx: algToIdx(mv.to) };
      recordPosition();
      const info: ModelMoveInfo = {
        uci: mv.from + mv.to + (mv.promotion ?? ''),
        san: applied.san,
        fromIdx: lastMove.fromIdx,
        toIdx: lastMove.toIdx,
        value,
        policyProb: prob,
      };
      candidates = null;
      commit({ lastModelMove: info, flashMove: null, previewMoves: null });
    }

    async function runModel(): Promise<void> {
      const token = ++thinkToken;
      candidates = null;
      set({ status: 'thinking', candidates: null, search: null, flashMove: null, previewMoves: null });

      // MCTS ("think harder") path: search off the main thread, visualize live,
      // then play the most-visited root move. Uses ONLY the model's P/V.
      if (get().mcts.enabled && engine.search) {
        const m = get().mcts;
        const repCountsObj: Record<string, number> = {};
        for (const [k, v] of repCounts) repCountsObj[k] = v;
        const opts: SearchOptions = {
          sims: m.sims,
          timeMs: m.timeMs,
          cPuct: m.cPuct,
          temperature: m.temperature,
          repCounts: repCountsObj,
        };
        let result;
        try {
          result = await engine.search(chess.fen(), opts, (snap) => {
            if (token === thinkToken) set({ search: snap });
          });
        } catch (e) {
          if (token !== thinkToken) return;
          commit({ status: 'idle', error: `Search failed: ${(e as Error).message}` });
          return;
        }
        if (token !== thinkToken) return; // superseded by reset/newGame/load
        set({ search: result.snapshot });
        if (!result.move) {
          commit({ error: 'Search produced no legal move.' });
          return;
        }
        // Flash the chosen move with the pick-mode hover highlight for a brief
        // beat, then play it — a consistent "here's the move I chose" moment.
        const mv = result.move;
        set({
          flashMove: {
            from: mv.from,
            to: mv.to,
            promotion: mv.promotion,
            fromIdx: mv.fromIdx,
            toIdx: mv.toIdx,
            uci: mv.uci,
            san: mv.san,
            prob: result.prior,
          },
        });
        await new Promise((r) => setTimeout(r, MCTS_FLASH_MS));
        if (token !== thinkToken) return; // superseded during the flash beat
        applyModelMove(
          { from: mv.from, to: mv.to, promotion: mv.promotion },
          result.value,
          result.prior,
        );
        return;
      }

      const planes = featurize(boardState(), 'float');
      const { mask, indexToMove } = buildLegalMaskAndMap(chess);
      let reply;
      try {
        reply = await engine.forward(planes, mask, false);
      } catch (e) {
        if (token !== thinkToken) return;
        commit({ status: 'idle', error: `Inference failed: ${(e as Error).message}` });
        return;
      }
      if (token !== thinkToken) return; // superseded by reset/newGame/load

      if (get().pickMode) {
        // Surface the top-K legal moves (ranked by policy prob) and wait for a pick.
        const ranked: ModelCandidate[] = [];
        for (const [idx, mv] of indexToMove) {
          ranked.push({
            from: mv.from,
            to: mv.to,
            promotion: mv.promotion as PromotionPiece | undefined,
            fromIdx: algToIdx(mv.from),
            toIdx: algToIdx(mv.to),
            uci: mv.from + mv.to + (mv.promotion ?? ''),
            san: mv.san,
            prob: reply.policyProbs[idx] ?? 0,
          });
        }
        ranked.sort((a, b) => b.prob - a.prob);
        if (ranked.length === 0) {
          commit({ error: 'Model produced no legal move.' });
          return;
        }
        candidates = ranked.slice(0, TOP_K_CANDIDATES);
        pendingValue = reply.value;
        commit({ status: 'choosing' });
        return;
      }

      // Auto-play: sample among the legal moves under the current temperature
      // (T=0 reproduces the original argmax-over-legal behavior exactly).
      const legalIndices = [...indexToMove.keys()];
      const chosen = sampleLegalIndex(reply.policyLogits, legalIndices, reply.bestLegalIndex, get().temperature);
      const mv = indexToMove.get(chosen);
      if (!mv) {
        commit({ error: 'Model produced no legal move.' });
        return;
      }

      // Present the single forward pass like "MCTS with one simulation":
      //   1) surface the top-K policy moves as prob-weighted arrows (the move
      //      distribution), 2) flash the selected move with the pick-mode
      //      gold/red highlight, then 3) play it. Each beat is token-guarded so a
      //      New Game / Load FEN / undo during it cancels cleanly.
      const ranked: ModelCandidate[] = [];
      for (const [idx, m] of indexToMove) {
        ranked.push({
          from: m.from,
          to: m.to,
          promotion: m.promotion as PromotionPiece | undefined,
          fromIdx: algToIdx(m.from),
          toIdx: algToIdx(m.to),
          uci: m.from + m.to + (m.promotion ?? ''),
          san: m.san,
          prob: reply.policyProbs[idx] ?? 0,
        });
      }
      ranked.sort((a, b) => b.prob - a.prob);

      // 1) Show the top moves' distribution.
      set({ previewMoves: ranked.slice(0, TOP_K_CANDIDATES) });
      await new Promise((r) => setTimeout(r, AUTOPLAY_PREVIEW_MS));
      if (token !== thinkToken) return; // superseded during the preview beat

      // 2) Flash the chosen move (reuses the pick-mode hover highlight).
      set({
        previewMoves: null,
        flashMove: {
          from: mv.from,
          to: mv.to,
          promotion: mv.promotion as PromotionPiece | undefined,
          fromIdx: algToIdx(mv.from),
          toIdx: algToIdx(mv.to),
          uci: mv.from + mv.to + (mv.promotion ?? ''),
          san: mv.san,
          prob: reply.policyProbs[chosen] ?? 0,
        },
      });
      await new Promise((r) => setTimeout(r, MCTS_FLASH_MS));
      if (token !== thinkToken) return; // superseded during the flash beat

      // 3) Play it.
      applyModelMove(
        { from: mv.from, to: mv.to, promotion: mv.promotion as PromotionPiece | undefined },
        reply.value,
        reply.policyProbs[chosen] ?? 0,
      );
    }

    function startGame(humanColor: Color, fen?: string): void {
      thinkToken++; // cancel any in-flight reply
      engine.cancelSearch?.(); // stop any in-flight worker search
      chess = fen ? new Chess(fen) : new Chess();
      repCounts = new Map();
      epSquare = fen ? epTargetFromFen(fen) : null;
      lastMove = null;
      candidates = null;
      recordPosition();
      commit({ humanColor, lastModelMove: null, search: null, flashMove: null, previewMoves: null });
      if (!chess.isGameOver() && chess.turn() !== humanColor) void runModel();
    }

    // ---- initial state ----
    recordPosition();
    const sr0 = statusResult();
    return {
      fen: chess.fen(),
      turn: chess.turn() as Color,
      humanColor: initialHumanColor,
      sanHistory: chess.history(),
      status: sr0.status,
      resultText: sr0.resultText,
      inCheck: chess.inCheck(),
      lastMove: null,
      lastModelMove: null,
      pickMode: false,
      temperature: DEFAULT_TEMPERATURE,
      candidates: null,
      flashMove: null,
      previewMoves: null,
      mcts: { ...MCTS_DEFAULTS },
      search: null,
      error: null,

      humanMove(fromIdx, toIdx, promotion) {
        const s = get();
        if (s.status === 'thinking' || s.status === 'over') return false;
        if (chess.turn() !== s.humanColor) return false;
        let applied: ChessMove;
        try {
          applied = chess.move({ from: idxToAlg(fromIdx), to: idxToAlg(toIdx), promotion: promotion ?? 'q' });
        } catch {
          return false; // illegal
        }
        epSquare = epTargetAfterMove(applied);
        lastMove = { fromIdx, toIdx };
        recordPosition();
        commit();
        if (!chess.isGameOver() && chess.turn() !== s.humanColor) void runModel();
        return true;
      },

      newGame(humanColor) {
        startGame(humanColor ?? get().humanColor);
      },

      setPickMode(on) {
        set({ pickMode: on });
        // Turning auto-play back on while waiting on a pick resolves to the top move.
        if (!on && get().status === 'choosing') get().chooseModelMove(0);
      },

      setTemperature(t) {
        set({ temperature: Math.max(0, Number.isFinite(t) ? t : 0) });
      },

      setMctsEnabled(on) {
        set({ mcts: { ...get().mcts, enabled: on } });
      },

      setMctsSettings(patch) {
        const cur = get().mcts;
        const next: MctsSettings = { ...cur };
        if (patch.sims !== undefined && Number.isFinite(patch.sims)) next.sims = Math.max(1, Math.round(patch.sims));
        if (patch.timeMs !== undefined && Number.isFinite(patch.timeMs)) next.timeMs = Math.max(100, Math.round(patch.timeMs));
        if (patch.cPuct !== undefined && Number.isFinite(patch.cPuct)) next.cPuct = Math.max(0, patch.cPuct);
        if (patch.temperature !== undefined && Number.isFinite(patch.temperature)) next.temperature = Math.max(0, patch.temperature);
        if (patch.enabled !== undefined) next.enabled = patch.enabled;
        set({ mcts: next });
      },

      chooseModelMove(i) {
        if (get().status !== 'choosing' || !candidates) return;
        const cand = candidates[i];
        if (!cand) return;
        applyModelMove({ from: cand.from, to: cand.to, promotion: cand.promotion }, pendingValue, cand.prob);
      },

      loadFen(fen) {
        if (!validateFen(fen).ok) {
          set({ error: `Invalid FEN: ${validateFen(fen).error ?? 'parse error'}` });
          return false;
        }
        startGame(get().humanColor, fen);
        return true;
      },

      undo() {
        if (get().status === 'thinking') return;
        // Undo back to the human's turn (undo the model reply too if present).
        const undone: boolean[] = [chess.undo() != null];
        if (!chess.isGameOver() && chess.turn() !== get().humanColor) undone.push(chess.undo() != null);
        if (!undone.some(Boolean)) return;
        // Rebuild repetition counts from scratch (cheap: short history).
        repCounts = new Map();
        const replay = new Chess();
        repCounts.set(replay.fen().split(' ').slice(0, 4).join(' '), 1);
        for (const m of chess.history({ verbose: true })) {
          replay.move({ from: m.from, to: m.to, promotion: m.promotion });
          const k = replay.fen().split(' ').slice(0, 4).join(' ');
          repCounts.set(k, (repCounts.get(k) ?? 0) + 1);
        }
        const hist = chess.history({ verbose: true });
        const last = hist[hist.length - 1];
        epSquare = last ? epTargetAfterMove(last) : null;
        lastMove = last ? { fromIdx: algToIdx(last.from), toIdx: algToIdx(last.to) } : null;
        candidates = null;
        commit({ lastModelMove: null, search: null, flashMove: null, previewMoves: null });
      },

      legalTargets(fromIdx) {
        if (chess.turn() !== get().humanColor || get().status !== 'idle') return [];
        return chess
          .moves({ square: idxToAlg(fromIdx) as Square, verbose: true })
          .map((m) => algToIdx(m.to));
      },
    };
  });

  return store;
}
