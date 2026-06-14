// Search (MCTS) public surface — presentation-agnostic.

export { MCTS } from './mcts.ts';
export type { Evaluator, LeafEvaluation, SearchContext, MCTSConfig } from './mcts.ts';
export { makeEngineEvaluator } from './evaluator.ts';
export { runSearch } from './run.ts';
export type { RunSearchHooks } from './run.ts';
export type {
  MoveLite,
  RootChildStat,
  SearchSnapshot,
  SearchOptions,
  SearchResult,
} from './types.ts';
