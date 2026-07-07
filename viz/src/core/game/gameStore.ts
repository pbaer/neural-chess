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
import { selectMoveIndex } from '../search/selection.ts';
import {
  algToIdx,
  idxToAlg,
  chessToBoardState,
  buildLegalMaskAndMap,
  epTargetAfterMove,
  epTargetFromFen,
} from './chessAdapter.ts';
import { loadPersisted, savePersisted, clearPersisted } from './persistence.ts';

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

/**
 * One value-head reading, recorded each time the model moves, re-expressed in a
 * fixed frame: + favors White, − favors Black (unlike ModelMoveInfo.value, which
 * is from the side-to-move's perspective). Used to plot how the model's judged
 * White-vs-Black advantage shifted over the game in the move list.
 */
export interface ValueSample {
  /** Half-move index (1 = after White's 1st move); indexes sanHistory[ply-1]. */
  ply: number;
  /** Value head in [-1,1] re-framed to White(+)/Black(−). */
  whiteValue: number;
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
 * Default "Move variety" S in [0,1] (shared by both modes). 0 = always the model's
 * single best move (deterministic). ~0.5 keeps play recognisably strong while making
 * every game different — the "interesting" default. Selection is value-adaptive:
 * the model sharpens to the best move when losing and explores more when winning,
 * but never leaves its set of reasonable moves (see core/search/selection.ts).
 */
export const DEFAULT_VARIETY = 0.5;

/** Early-cutoff visit-fraction threshold default for MCTS (top move's share). */
export const MCTS_DEFAULT_CUTOFF = 0.7;

/**
 * Default MCTS ("think harder") settings. `sims` is now the MAXIMUM simulation
 * budget — the search cuts off early once a single move is clearly best (see
 * cutoffThreshold). 50 is the project's best in-budget setting and is cheap on the
 * deployed nano model. c_puct 1.5 is the project's tuned value.
 */
export const MCTS_DEFAULTS = { enabled: false, sims: 100, cPuct: 1.5, cutoffThreshold: MCTS_DEFAULT_CUTOFF } as const;

/** Max-simulation bounds for the MCTS control. */
export const MCTS_MIN_SIMS = 10;
export const MCTS_MAX_SIMS = 300;
/** Early-cutoff threshold bounds for the MCTS control. */
export const MCTS_MIN_CUTOFF = 0.5;
export const MCTS_MAX_CUTOFF = 0.95;

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
  /** MAXIMUM simulations per move (the search may stop earlier — see cutoffThreshold). */
  sims: number;
  /** PUCT exploration constant. */
  cPuct: number;
  /** Early-cutoff visit-fraction threshold in [0,1] (top move's share of visits). */
  cutoffThreshold: number;
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
   * Value-head reading per model move so far, in a fixed White(+)/Black(−) frame
   * (see ValueSample). One entry is appended each time the model moves; shown in
   * the move list as the history of the model's advantage judgement.
   */
  valueHistory: ValueSample[];
  /**
   * Move-assistant toggle. When on, the model is run on the HUMAN's position each
   * of the human's turns and its top moves for the human's own side are surfaced
   * (arrows + a candidate list) as hints. The human still makes their own move.
   */
  assist: boolean;
  /** "Move variety" S in [0,1] — shared by both modes (0 = deterministic top move). */
  variety: number;
  /**
   * Move-assistant suggestions: the model's top-K predicted moves for the HUMAN's
   * own side at the current position. Set only on the human's turn while `assist`
   * is on; null otherwise.
   */
  suggestions: ModelCandidate[] | null;
  /**
   * The move about to land, briefly published BEFORE it's applied so the board can
   * flash it with the same gold/red highlight as a hovered pick-mode candidate.
   * Used by EVERY move — the model's (MCTS / auto-play pick) AND the human's (board
   * click or assist-suggestion click) — so every move shows the same brief "here's
   * the move I'm about to make" beat. Null except during that ~MCTS_FLASH_MS beat.
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
  /** Set the "Move variety" S, clamped to [0,1]. */
  setVariety(s: number): void;
  /** Select the play mode: false = One-Shot (default), true = MCTS ("think harder"). */
  setMctsEnabled(on: boolean): void;
  /** Patch MCTS settings (sims / cPuct / cutoffThreshold). */
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

  // The position the current game started from (null = standard start). Tracked so
  // persistence + repetition rebuilds replay from the right root (also for FEN games).
  let startFen: string | null = null;

  /** Rebuild repetition counts by replaying the whole game from `startFen`. */
  function rebuildRepCounts(): void {
    repCounts = new Map();
    const rp = new Chess(startFen ?? undefined);
    repCounts.set(rp.fen().split(' ').slice(0, 4).join(' '), 1);
    for (const m of chess.history({ verbose: true })) {
      rp.move({ from: m.from, to: m.to, promotion: m.promotion });
      const k = rp.fen().split(' ').slice(0, 4).join(' ');
      repCounts.set(k, (repCounts.get(k) ?? 0) + 1);
    }
  }

  // ── Restore a persisted game across a page refresh (board + settings only; the
  //    model/worker reloads fresh). Defensive: any malformed/incompatible blob is
  //    ignored and we just start a normal game.
  const restoreDefaults = {
    humanColor: initialHumanColor,
    assist: false,
    variety: DEFAULT_VARIETY,
    mcts: { ...MCTS_DEFAULTS } as MctsSettings,
    valueHistory: [] as ValueSample[],
  };
  const persisted = loadPersisted(restoreDefaults);
  const restored = { ...restoreDefaults };
  if (persisted) {
    try {
      const c = new Chess(persisted.startFen ?? undefined);
      for (const m of persisted.moves) c.move({ from: m.from, to: m.to, promotion: m.promotion });
      chess = c;
      startFen = persisted.startFen ?? null;
      restored.humanColor = persisted.humanColor;
      restored.assist = persisted.assist;
      restored.variety = persisted.variety;
      restored.mcts = persisted.mcts;
      // Keep only readings for plies that still exist in the replayed game (guards
      // against a hand-edited / truncated blob) — same invariant undo maintains.
      restored.valueHistory = persisted.valueHistory.filter(
        (v) => v.ply >= 1 && v.ply <= chess.history().length,
      );
      const verbose = chess.history({ verbose: true });
      const lastV = verbose[verbose.length - 1];
      lastMove = lastV ? { fromIdx: algToIdx(lastV.from), toIdx: algToIdx(lastV.to) } : null;
      epSquare = epTargetFromFen(chess.fen());
    } catch {
      // Corrupt/incompatible save → discard and start fresh.
      chess = new Chess();
      startFen = null;
      lastMove = null;
      epSquare = null;
      clearPersisted();
    }
  }

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

    /** Save the current game (position + settings) so a refresh restores it.
     *  Persists only landed moves — never a mid-flash / "thinking" transient. */
    function persist(): void {
      savePersisted({
        startFen,
        moves: chess.history({ verbose: true }).map((m) => ({
          from: m.from,
          to: m.to,
          ...(m.promotion ? { promotion: m.promotion } : {}),
        })),
        humanColor: get().humanColor,
        assist: get().assist,
        variety: get().variety,
        mcts: { ...get().mcts },
        valueHistory: get().valueHistory,
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
      // Record the value head in a fixed White(+)/Black(−) frame. `value` is from
      // the side-to-move's (the model's) perspective, so +1 means White ahead iff
      // the model played White.
      const modelColor: Color = get().humanColor === 'w' ? 'b' : 'w';
      const sample: ValueSample = { ply: chess.history().length, whiteValue: modelColor === 'w' ? value : -value };
      commit({
        lastModelMove: info,
        valueHistory: [...get().valueHistory, sample],
        flashMove: null,
        previewMoves: null,
      });
      persist();
      // Back to the human → surface move-assistant suggestions if enabled.
      refreshSuggestions();
    }

    /**
     * Drive an already-validated HUMAN move through the SAME flash beat the model
     * uses: briefly publish it as `flashMove` (gold/red highlight) for ~MCTS_FLASH_MS,
     * THEN apply it to chess.js — so human and model moves land identically. The move
     * is NOT applied to chess.js yet at call time (the caller trial-applied + undid it
     * only to validate / read its SAN), so the board still shows the pre-move position
     * with the flash arrow overlaid. Token-guarded exactly like the model path: a New
     * Game / Load FEN bumps `thinkToken` and the post-beat check drops the apply; the
     * `status: 'thinking'` set here disables the board and blocks a second human move
     * (and blocks undo) for the duration of the beat — no stale or double move.
     */
    async function flashThenApplyHumanMove(
      mv: { from: string; to: string; promotion?: PromotionPiece },
      san: string,
    ): Promise<void> {
      const token = ++thinkToken;
      suggestToken++; // the human committed → drop their suggestions
      set({
        status: 'thinking',
        suggestions: null,
        search: null,
        previewMoves: null,
        flashMove: {
          from: mv.from,
          to: mv.to,
          promotion: mv.promotion,
          fromIdx: algToIdx(mv.from),
          toIdx: algToIdx(mv.to),
          uci: mv.from + mv.to + (mv.promotion ?? ''),
          san,
          prob: 1, // single highlighted move → full-strength arrow
        },
      });
      await new Promise((r) => setTimeout(r, MCTS_FLASH_MS));
      if (token !== thinkToken) return; // superseded (new game / load fen) during the beat
      // Apply for real now (re-applies the move the caller trial-validated + undid).
      const applied = chess.move({ from: mv.from, to: mv.to, promotion: mv.promotion ?? 'q' });
      epSquare = epTargetAfterMove(applied);
      lastMove = { fromIdx: algToIdx(mv.from), toIdx: algToIdx(mv.to) };
      recordPosition();
      commit({ suggestions: null, flashMove: null });
      persist();
      if (!chess.isGameOver() && chess.turn() !== get().humanColor) void runModel();
      else refreshSuggestions();
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
          variety: get().variety,
          cutoffThreshold: m.cutoffThreshold,
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

      // Auto-play: value-adaptive, bounded selection over the legal-masked policy.
      // D = policy probs, V = the value head. variety=0 reproduces the original
      // argmax-over-legal behavior exactly; otherwise sample within the model's
      // reasonable moves under a temperature that sharpens when losing / relaxes
      // when winning. Shared with MCTS via core/search/selection.ts.
      const legalIndices = [...indexToMove.keys()];
      const pick = selectMoveIndex(
        legalIndices.map((i) => ({ weight: reply.policyProbs[i] ?? 0 })),
        { value: reply.value, variety: get().variety },
      );
      const chosen = pick >= 0 ? legalIndices[pick] : reply.bestLegalIndex;
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
      startFen = fen ?? null;
      repCounts = new Map();
      epSquare = fen ? epTargetFromFen(fen) : null;
      lastMove = null;
      suggestToken++; // drop any pending suggestion reply
      recordPosition();
      commit({ humanColor, lastModelMove: null, valueHistory: [], suggestions: null, search: null, flashMove: null, previewMoves: null });
      persist();
      if (!chess.isGameOver() && chess.turn() !== humanColor) void runModel();
      else refreshSuggestions(); // human to move → assistant suggestions if enabled
    }

    // ---- initial state ----
    // rebuildRepCounts handles both a fresh game (just the start position) and a
    // restored multi-move game (all positions, for threefold detection).
    rebuildRepCounts();
    const sr0 = statusResult();
    // If a restored game left the MODEL to move, resume it once the store is wired
    // (microtask, so set/get are live). engine.forward/search await the worker, so
    // this is safe even before weights finish loading. Else, surface assist
    // suggestions if they were enabled. (No-op for a fresh game: human is to move.)
    if (!chess.isGameOver() && chess.turn() !== restored.humanColor) {
      queueMicrotask(() => void runModel());
    } else if (restored.assist) {
      queueMicrotask(() => refreshSuggestions());
    }
    return {
      fen: chess.fen(),
      turn: chess.turn() as Color,
      humanColor: restored.humanColor,
      sanHistory: chess.history(),
      status: sr0.status,
      resultText: sr0.resultText,
      inCheck: chess.inCheck(),
      lastMove,
      lastModelMove: null,
      valueHistory: restored.valueHistory,
      assist: restored.assist,
      variety: restored.variety,
      suggestions: null,
      flashMove: null,
      previewMoves: null,
      mcts: restored.mcts,
      search: null,
      error: null,

      humanMove(fromIdx, toIdx, promotion) {
        const s = get();
        // Re-entry guard: `status: 'thinking'` (set by the flash beat below or by the
        // model) blocks starting a second move mid-flash. 'over' blocks moves too.
        if (s.status === 'thinking' || s.status === 'over') return false;
        if (chess.turn() !== s.humanColor) return false;
        // Validate by trial-applying, then immediately undo — we re-apply after the
        // flash beat. This keeps chess.js legality/SAN/promotion semantics identical
        // to a direct chess.move while leaving the board on the pre-move position so
        // the flash arrow overlays it (the move hasn't "landed" yet).
        let trial: ChessMove;
        try {
          trial = chess.move({ from: idxToAlg(fromIdx), to: idxToAlg(toIdx), promotion: promotion ?? 'q' });
        } catch {
          return false; // illegal
        }
        chess.undo(); // revert — only needed it to validate & read san/promotion
        // Drive it through the SAME flash beat the model uses, then apply.
        void flashThenApplyHumanMove(
          { from: trial.from, to: trial.to, promotion: trial.promotion as PromotionPiece | undefined },
          trial.san,
        );
        return true;
      },

      newGame(humanColor) {
        startGame(humanColor ?? get().humanColor);
      },

      setAssist(on) {
        set({ assist: on });
        persist();
        refreshSuggestions(); // compute on the human's turn, else clear
      },

      setVariety(s) {
        set({ variety: Math.min(1, Math.max(0, Number.isFinite(s) ? s : 0)) });
        persist();
      },

      setMctsEnabled(on) {
        set({ mcts: { ...get().mcts, enabled: on } });
        persist();
      },

      setMctsSettings(patch) {
        const cur = get().mcts;
        const next: MctsSettings = { ...cur };
        if (patch.sims !== undefined && Number.isFinite(patch.sims)) {
          next.sims = Math.min(MCTS_MAX_SIMS, Math.max(MCTS_MIN_SIMS, Math.round(patch.sims)));
        }
        if (patch.cPuct !== undefined && Number.isFinite(patch.cPuct)) next.cPuct = Math.max(0, patch.cPuct);
        if (patch.cutoffThreshold !== undefined && Number.isFinite(patch.cutoffThreshold)) {
          next.cutoffThreshold = Math.min(MCTS_MAX_CUTOFF, Math.max(MCTS_MIN_CUTOFF, patch.cutoffThreshold));
        }
        if (patch.enabled !== undefined) next.enabled = patch.enabled;
        set({ mcts: next });
        persist();
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
        // Rebuild repetition counts from scratch (cheap: short history; replays
        // from startFen so FEN-started games are handled correctly too).
        rebuildRepCounts();
        const hist = chess.history({ verbose: true });
        const last = hist[hist.length - 1];
        epSquare = last ? epTargetAfterMove(last) : epTargetFromFen(chess.fen());
        lastMove = last ? { fromIdx: algToIdx(last.from), toIdx: algToIdx(last.to) } : null;
        suggestToken++;
        // Drop value readings for plies that no longer exist after the undo.
        const kept = get().valueHistory.filter((v) => v.ply <= chess.history().length);
        commit({ lastModelMove: null, valueHistory: kept, suggestions: null, search: null, flashMove: null, previewMoves: null });
        persist();
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
