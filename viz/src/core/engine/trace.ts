// TraceRecorder — captures every intermediate of one forward pass.
//
// The trace is rebuilt per forward and is bounded by a single pass (regardless
// of game length). Granularity levels let callers trade memory for detail;
// a byte cap auto-downgrades when exceeded. Recorded tensors are COPIED (owned),
// never references into engine scratch buffers.
//
// For M1 the parity tests run at 'full' granularity and compare every recorded
// stage against the PyTorch golden.

export type TraceGranularity = 'outputs' | 'stage' | 'full';

export interface TraceEntry {
  data: Float32Array;
  shape: number[];
}

export interface Trace {
  entries: Map<string, TraceEntry>;
  bytes: number;
  granularity: TraceGranularity;
  capped: boolean;
}

const RANK: Record<TraceGranularity, number> = { outputs: 0, stage: 1, full: 2 };

export interface TraceOptions {
  granularity?: TraceGranularity;
  maxBytes?: number;
}

export class TraceRecorder {
  private entries = new Map<string, TraceEntry>();
  private bytes = 0;
  private capped = false;
  readonly granularity: TraceGranularity;
  readonly maxBytes: number;

  constructor(opts: TraceOptions = {}) {
    this.granularity = opts.granularity ?? 'full';
    this.maxBytes = opts.maxBytes ?? 64 * 1024 * 1024;
  }

  /**
   * Record an intermediate. `level` is the minimum granularity at which this
   * tensor is kept (default 'full'); coarser runs skip it. Data is copied.
   */
  record(name: string, data: Float32Array, shape: number[], level: TraceGranularity = 'full'): void {
    if (RANK[this.granularity] < RANK[level]) return;
    const nbytes = data.length * 4;
    if (this.bytes + nbytes > this.maxBytes) {
      this.capped = true;
      return;
    }
    this.entries.set(name, { data: Float32Array.from(data), shape: shape.slice() });
    this.bytes += nbytes;
  }

  has(name: string): boolean {
    return this.entries.has(name);
  }

  get(name: string): TraceEntry | undefined {
    return this.entries.get(name);
  }

  finalize(): Trace {
    return {
      entries: this.entries,
      bytes: this.bytes,
      granularity: this.granularity,
      capped: this.capped,
    };
  }
}
