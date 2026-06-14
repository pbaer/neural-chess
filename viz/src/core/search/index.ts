// Search (MCTS) public surface — presentation-agnostic.

export { MCTS } from './mcts.ts';
export type { Evaluator, LeafEvaluation, SearchContext, MCTSConfig } from './mcts.ts';
export { makeEngineEvaluator } from './evaluator.ts';
export { runSearch, MIN_SIMS, DEFAULT_CUTOFF_THRESHOLD } from './run.ts';
export type { RunSearchHooks } from './run.ts';
export {
  selectMoveIndex,
  reasonableSetIndices,
  valueAdaptiveTemperature,
  DEFAULT_SELECTION_CONFIG,
} from './selection.ts';
export type { SelectionCandidate, SelectionConfig, SelectionParams } from './selection.ts';
export type {
  MoveLite,
  RootChildStat,
  SearchSnapshot,
  SearchOptions,
  SearchResult,
} from './types.ts';
