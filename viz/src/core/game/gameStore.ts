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
export type GameStatus = 'idle' | 'thinking' | 'over';

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

/** One ranked legal move the model predicts (suggestion / preview / flash list). */
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

/** How many top legal moves to surface (move-assistant suggestions / previews). */
export const TOP_K_CANDIDATES = 5;

/**
 * Default move-selection temperature for auto-play. 0 = always the model's single
 * best move (deterministic); higher = more variety. ~0.6 keeps play recognisably
 * strong while making every game different — the "interesting" default.
 */
export const DEFAULT_TEMPERATURE = 0.6;

/**
 * Default MCTS ("think harder") settings. The search runs a FIXED number of
 * simulations each move (no wall-clock budget) for predictability — 50 is the
 * project's best in-budget setting and is cheap on the deployed nano model.
 * c_puct 1.5 is the project's tuned value; temperature 0 plays the most-visited
 * (strongest) move.
 */
export const MCTS_DEFAULTS = { enabled: false, sims: 50, cPuct: 1.5, temperature: 0 } as const;

/** Simulation-count bounds for the MCTS control (fixed sims, no time budget). */
export const MCTS_MIN_SIMS = 10;
export const MCTS_MAX_SIMS = 300;

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
  /** Number of simulations to run each move (exact; no wall-clock budget). */
  sims: number;
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
  /**
   * Move-assistant toggle. When on, the model is run on the HUMAN's position each
   * of the human's turns and its top moves for the human's own side are surfaced
   * (arrows + a candidate list) as hints. The human still makes their own move.
   */
  assist: boolean;
  /** Softmax temperature for auto-play move selection (0 = deterministic top move). */
  temperature: number;
  /**
   * Move-assistant suggestions: the model's top-K predicted moves for the HUMAN's
   * own side at the current position. Set only on the human's turn while `assist`
   * is on; null otherwise.
   */
  suggestions: ModelCandidate[] | null;
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
  /** Toggle the move assistant. On (the human's turn) → surface move suggestions. */
  setAssist(on: boolean): void;
  /** Set the auto-play move-selection temperature (clamped ≥ 0). */
  setTemperature(t: number): void;
  /** Toggle MCTS ("think harder"). Off = one-shot argmax (today's behavior). */
  setMctsEnabled(on: boolean): void;
  /** Patch MCTS settings (sims / cPuct / temperature). */
  setMctsSettings(patch: Partial<MctsSettings>): void;
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
  // Monotonic token to drop stale async model replies after reset/newGame/load.
  let thinkToken = 0;
  // Separate token for the (human-turn) move-assistant forward, so a stale
  // suggestion reply is dropped after any position change.
  let suggestToken = 0;

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
        error: null,
        ...extra,
      });
    }

    /** Build the prob-ranked top-K legal moves from a forward reply (for the
     *  move-assistant suggestion list / arrows). */
    function rankCandidates(
      indexToMove: Map<number, ChessMove>,
      policyProbs: Float32Array,
    ): ModelCandidate[] {
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
          prob: policyProbs[idx] ?? 0,
        });
      }
      ranked.sort((a, b) => b.prob - a.prob);
      return ranked;
    }

    /**
     * Move assistant: on the HUMAN's turn (assist on), run the model on the current
     * position and publish its top moves for the human's own side as suggestions.
     * A single forward pass — the model only ranks chess.js's legal moves; the human
     * still makes their own move. Token-guarded so a stale reply is dropped.
     */
    async function computeSuggestions(): Promise<void> {
      if (!get().assist) return;
      if (chess.isGameOver() || chess.turn() !== get().humanColor) return;
      const token = ++suggestToken;
      const planes = featurize(boardState(), 'float');
      const { mask, indexToMove } = buildLegalMaskAndMap(chess);
      let reply;
      try {
        reply = await engine.forward(planes, mask, false);
      } catch {
        return; // suggestions are advisory; swallow failures silently
      }
      // Drop if superseded, or if it's no longer the human's turn / assist was off.
      if (token !== suggestToken || !get().assist) return;
      if (chess.isGameOver() || chess.turn() !== get().humanColor) return;
      const ranked = rankCandidates(indexToMove, reply.policyProbs);
      set({ suggestions: ranked.length > 0 ? ranked.slice(0, TOP_K_CANDIDATES) : null });
    }

    /** Recompute suggestions if it's the human's turn (assist on), else clear them. */
    function refreshSuggestions(): void {
      if (get().assist && !chess.isGameOver() && chess.turn() === get().humanColor) {
        void computeSuggestions();
      } else if (get().suggestions) {
        suggestToken++;
        set({ suggestions: null });
      }
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
      commit({ lastModelMove: info, flashMove: null, previewMoves: null });
      // Back to the human → surface move-assistant suggestions if enabled.
      refreshSuggestions();
    }

    async function runModel(): Promise<void> {
      const token = ++thinkToken;
      // Model's turn → drop any human-turn suggestions.
      suggestToken++;
      set({ status: 'thinking', suggestions: null, search: null, flashMove: null, previewMoves: null });

      // MCTS ("think harder") path: search off the main thread, visualize live,
      // then play the most-visited root move. Uses ONLY the model's P/V.
      if (get().mcts.enabled && engine.search) {
        const m = get().mcts;
        const repCountsObj: Record<string, number> = {};
        for (const [k, v] of repCounts) repCountsObj[k] = v;
        const opts: SearchOptions = {
          sims: m.sims,
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
      //      distribution), 2) flash the selected move with the gold/red highlight,
      //      then 3) play it. Each beat is token-guarded so a New Game / Load FEN /
      //      undo during it cancels cleanly.
      const ranked = rankCandidates(indexToMove, reply.policyProbs);

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
      suggestToken++; // drop any pending suggestion reply
      recordPosition();
      commit({ humanColor, lastModelMove: null, suggestions: null, search: null, flashMove: null, previewMoves: null });
      if (!chess.isGameOver() && chess.turn() !== humanColor) void runModel();
      else refreshSuggestions(); // human to move → assistant suggestions if enabled
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
      assist: false,
      temperature: DEFAULT_TEMPERATURE,
      suggestions: null,
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
        suggestToken++; // the human moved → drop their suggestions
        recordPosition();
        commit({ suggestions: null });
        if (!chess.isGameOver() && chess.turn() !== s.humanColor) void runModel();
        return true;
      },

      newGame(humanColor) {
        startGame(humanColor ?? get().humanColor);
      },

      setAssist(on) {
        set({ assist: on });
        refreshSuggestions(); // compute on the human's turn, else clear
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
        if (patch.sims !== undefined && Number.isFinite(patch.sims)) {
          next.sims = Math.min(MCTS_MAX_SIMS, Math.max(MCTS_MIN_SIMS, Math.round(patch.sims)));
        }
        if (patch.cPuct !== undefined && Number.isFinite(patch.cPuct)) next.cPuct = Math.max(0, patch.cPuct);
        if (patch.temperature !== undefined && Number.isFinite(patch.temperature)) next.temperature = Math.max(0, patch.temperature);
        if (patch.enabled !== undefined) next.enabled = patch.enabled;
        set({ mcts: next });
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
        suggestToken++;
        commit({ lastModelMove: null, suggestions: null, search: null, flashMove: null, previewMoves: null });
        refreshSuggestions(); // back on the human's turn → suggestions if enabled
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
