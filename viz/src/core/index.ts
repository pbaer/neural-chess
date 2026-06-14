// THE CORE CONTRACT — the presentation-agnostic public API.
//
// Presentations (desktop, future mobile, lesson-embed) import ONLY from here.
// Nothing in this surface knows about React, the DOM, or desktop layout.

// Engine (inference kernel + types).
export {
  createEngine,
  createEngineFromBytes,
  createEngineFromCapsule,
  fetchCapsule,
  Capsule,
  featurize,
  applyInt8Truncation,
  encodeMove,
  decodeMove,
  legalMask,
  legalPolicySoftmax,
  pieceToMoveProbs,
  rotateSquare,
  NUM_MOVES,
  NUM_MOVE_TYPES,
  NUM_PLANES,
} from './engine/index.ts';
export type {
  Engine,
  EngineResult,
  ForwardOptions,
  BoardState,
  PieceInfo,
  Color,
  PieceType,
  Move,
  FeaturizeMode,
  Trace,
  TraceEntry,
  CapsuleConfig,
  CapsuleManifest,
  GraphStage,
  TensorIndexEntry,
} from './engine/index.ts';

// Engine transport (worker client).
export { createWorkerEngineClient } from './game/engineClient.ts';
export type { EngineClient, EngineMeta, ForwardReply } from './game/engineClient.ts';

// Game + play loop.
export { createGameStore, MCTS_DEFAULTS, MCTS_MIN_SIMS, MCTS_MAX_SIMS } from './game/gameStore.ts';
export type { GameStore, GameState, GameStatus, ModelMoveInfo, ModelCandidate, PromotionPiece, MctsSettings } from './game/gameStore.ts';
export {
  algToIdx,
  idxToAlg,
  chessToBoardState,
  buildLegalMaskAndMap,
  epTargetFromFen,
  epTargetAfterMove,
} from './game/chessAdapter.ts';

// ── Visualization core (M3): trace singleton, model-graph walk, trace index,
//    and the kind-keyed content registry. All presentation-agnostic.
export { traceStore } from './state/traceStore.ts';
export type { TraceStore, TracePayload, TraceEntryData, TraceMeta, TraceSnapshot } from './state/traceStore.ts';

export { buildModelGraph, shapeSummary, idIndex } from './model-graph/graph.ts';
export type { ModelGraph, GraphNode, GraphSource } from './model-graph/graph.ts';

export { traceFieldsFor, outputTraceKey } from './model-graph/traceIndex.ts';
export type { TraceField } from './model-graph/traceIndex.ts';

export { content, CONTENT } from './content/registry.ts';
export type { ContentCard } from './content/registry.ts';

// ── Search (MCTS) — optional "think harder" move generator + its visualization.
export { MCTS, makeEngineEvaluator, runSearch } from './search/index.ts';
export type {
  Evaluator,
  LeafEvaluation,
  SearchContext,
  MCTSConfig,
  RunSearchHooks,
  MoveLite,
  RootChildStat,
  SearchSnapshot,
  SearchOptions,
  SearchResult,
} from './search/index.ts';
