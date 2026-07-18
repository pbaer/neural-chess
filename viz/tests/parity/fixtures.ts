// Shared parity-test fixtures: load the capsule + golden vectors from disk and
// expose helpers. Pure Node (fs), no DOM.
//
// The capsule (capsule.json + weights.bin) is the SHIPPING artifact and lives in
// public/weights/; the golden vectors are dev-only test fixtures and live here
// under tests/parity/fixtures/ (regenerated via scripts/export/gen_golden.py).

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import {
  createEngineFromBytes,
  type Engine,
  type BoardState,
  type Color,
  type PieceType,
} from '../../src/core/engine/index.ts';

const WEIGHTS_DIR = fileURLToPath(new URL('../../public/weights/v3.1-nano/', import.meta.url));
const GOLDEN_DIR = fileURLToPath(new URL('./fixtures/v3.1-nano/', import.meta.url));

function readArrayBuffer(path: string): ArrayBuffer {
  const buf = readFileSync(path);
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
}

function readJSON<T>(path: string): T {
  return JSON.parse(readFileSync(path, 'utf-8')) as T;
}

export interface TensorRef {
  offset: number;
  length: number;
  shape: number[];
}

export interface BoardDump {
  turn: Color;
  pieces: Array<{ square: number; color: Color; type: PieceType }>;
  castling: { wk: boolean; wq: boolean; bk: boolean; bq: boolean };
  ep: number | null;
  halfmove: number;
  fullmove: number;
  isRep2: boolean;
  isRep3: boolean;
}

export interface LegalMoveDump {
  from: number;
  to: number;
  prom: number;
  index: number;
  decFrom: number;
  decTo: number;
  decProm: number;
}

export interface GoldenCase {
  name: string;
  fen: string;
  turn: Color;
  isWhite: boolean;
  realBoard: BoardDump;
  rotatedBoard: BoardDump;
  tensors: Record<string, TensorRef>;
  value: number;
  legalIndices: number[];
  bestLegalIndex: number;
  decodedBest: { from: number; to: number; prom: number };
  moveReal: { from: number; to: number; prom: number; uci: string };
  legalMoves: LegalMoveDump[];
}

export interface Golden {
  model_id: string;
  checkpoint: string;
  config: Record<string, number | boolean>;
  numMoves: number;
  bin_floats: number;
  erfGrid: { x: number[]; erf: number[] };
  rotateSquareRef: number[];
  cases: GoldenCase[];
}

export function loadEngine(): Engine {
  const manifest = readJSON<any>(WEIGHTS_DIR + 'capsule.json');
  const weights = readArrayBuffer(WEIGHTS_DIR + 'weights.bin');
  return createEngineFromBytes(manifest, weights);
}

export function loadGolden(): { golden: Golden; tensor: (ref: TensorRef) => Float32Array } {
  const golden = readJSON<Golden>(GOLDEN_DIR + 'golden.json');
  const binBuf = readArrayBuffer(GOLDEN_DIR + 'golden.bin');
  const all = new Float32Array(binBuf);
  const tensor = (ref: TensorRef): Float32Array => all.subarray(ref.offset, ref.offset + ref.length);
  return { golden, tensor };
}

/** Build a BoardState from a dumped python-chess board (no chess.js needed). */
export function buildBoardState(d: BoardDump): BoardState {
  const pieces = d.pieces.map((p) => ({ square: p.square, color: p.color, type: p.type }));
  const bySquare = new Map<number, PieceType>(pieces.map((p) => [p.square, p.type] as const));
  return {
    turn: d.turn,
    pieces: () => pieces,
    hasKingsideCastlingRights: (c: Color) => (c === 'w' ? d.castling.wk : d.castling.bk),
    hasQueensideCastlingRights: (c: Color) => (c === 'w' ? d.castling.wq : d.castling.bq),
    epSquare: d.ep,
    halfmoveClock: d.halfmove,
    fullmoveNumber: d.fullmove,
    isRepetition: (n: number) => (n >= 3 ? d.isRep3 : n >= 2 ? d.isRep2 : true),
    pieceTypeAt: (sq: number) => bySquare.get(sq) ?? null,
  };
}

// ---- numeric comparison helpers ----

export function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}

export function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}
