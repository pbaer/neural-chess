// Contextual-explanation registry — keyed by stage/op `kind`, NOT by any
// architecture version. Every stage kind present in the hero capsule has a card;
// a few op-level kinds (attention, softmax, ffn, …) back the math-lens blurbs.
// A new model expressible in these kinds renders its explanations with zero
// changes; a genuinely new kind is a localized add here.
//
// Each card has five slots authored for a HS-calculus reader:
//   what  — one plain sentence
//   how   — the mechanism / arithmetic
//   why   — the design rationale / intuition
//   dims  — where it sits + its shape (filled generically by callers)
//   principle — optional note tying to the project's "discovered, not injected"
//               framing (load-bearing for the geometry bias)

export interface ContentCard {
  title: string;
  what: string;
  how: string;
  why: string;
  principle?: string;
}

export const CONTENT: Record<string, ContentCard> = {
  input_planes: {
    title: 'Input planes',
    what: 'The board position translated into plain numbers the network can read — a stack of 21 small 8×8 grids, one grid per feature.',
    how: 'Each of the 21 “planes” is an 8×8 grid with one cell per square. Twelve planes mark where each kind of piece sits (your six piece types, then the opponent’s); four flag castling rights; one marks the en-passant square; and the last four hold the move clocks, whose turn it is, and whether this position has occurred before. The side to move is always rotated to the bottom, so the network always reads the board from the mover’s point of view.',
    why: 'A neural network only does arithmetic on numbers, so the position has to become numbers first. Using a separate grid for each feature keeps everything lined up with the squares it describes, which makes the board’s geometry easy to work with.',
    principle: 'Everything here is just what you could write down by looking at the board — piece locations, castling rights, move counts. No engine evaluation or hand-built chess “knowledge” is ever fed in.',
  },
  embed: {
    title: 'Embed (per-square)',
    what: 'Gives every square its own short list of numbers — a “feature vector” — summarizing what the planes say about it.',
    how: 'This is a 1×1 convolution, which is just a small matrix multiply applied to each square on its own: it takes that square’s 21 plane values and mixes them into d new numbers (out[c] = Σ over planes of W[c,plane]·value + b[c]), then keeps only the positive part (ReLU). The very same weights are reused for all 64 squares.',
    why: 'The raw planes are mostly 0s and 1s. This step blends them into a richer description of “what is on and around this square” before any square starts comparing notes with the others.',
  },
  stem_conv: {
    title: 'Convolutional stem',
    what: 'Optional early layers that let each square peek at its immediate neighbours.',
    how: 'A k×k convolution slides a small learned window across the board, so each output square is computed from a patch of nearby squares; ReLU then keeps the positive part. (Any BatchNorm has been folded into these weights, so the layer already carries its own scaling.)',
    why: 'It builds in cheap local structure — things like “there’s a pawn directly in front of me” — before the transformer lets far-apart squares interact.',
  },
  tokenize: {
    title: 'Tokenize + position',
    what: 'Lays the 8×8 board out as a list of 64 “tokens” (one per square) and stamps each with a learned code for where it sits.',
    how: 'Square s (= rank×8 + file) becomes token number s, carrying that square’s feature vector. A learned position vector, pos_emb[s], is then added on top so each of the 64 tokens has its own distinct “address”.',
    why: 'The attention step that follows treats its inputs as an unordered set — without an address, the token for e4 and the token for h1 would look identical. These position codes are how the model keeps track of the layout, and it actually learns the board’s geometry into them.',
  },
  block: {
    title: 'Transformer block',
    what: 'One full round of the transformer: every square looks around at the others, then each square rethinks on its own.',
    how: 'Two sub-steps, each added back onto a running “residual stream”. First x ← x + Attention(LayerNorm(x)), where attention lets each square pull in information from any other square. Then x ← x + FeedForward(LayerNorm(x)), which re-processes each square individually. (LayerNorm just keeps the numbers in a sane range; “+ x” means each step nudges the stream rather than replacing it.)',
    why: 'Stacking several of these blocks lets information travel back and forth across the whole board, building up more and more abstract ideas about the position. Adding to the stream instead of overwriting it keeps earlier information available and helps the deep network train smoothly.',
  },
  layernorm: {
    title: 'LayerNorm',
    what: 'Rescales one token’s list of numbers so they have a consistent size before the next step.',
    how: 'For a single token: subtract the average of its d numbers, divide by how spread out they are (the standard deviation, plus a tiny ε so we never divide by zero), then apply a learned per-feature stretch (γ) and shift (β).',
    why: 'Deep networks misbehave when activations grow or shrink unpredictably. Normalizing each token keeps every layer working in a comfortable range; the learned γ and β let the model dial that back wherever it helps.',
  },
  policy_head: {
    title: 'Policy head',
    what: 'Scores every move the position could make — one number for each (from-square, move-type) pair.',
    how: 'A single linear layer turns each of the 64 square-tokens into 73 numbers, giving a 64×73 grid = 4672 raw scores (“logits”). The 73 move types cover 56 sliding moves (8 directions × up to 7 squares), 8 knight jumps, and 9 underpromotions. Illegal moves are then crossed out, and a softmax turns what remains into probabilities that add up to 100%.',
    why: 'Describing a move as “start here, move like this” lets one fixed-size output cover every possible chess move — including promotions — without having to list each position’s legal moves separately.',
  },
  value_head: {
    title: 'Value head',
    what: 'Boils the whole position down to a single “who is winning” number between −1 and +1.',
    how: 'Average the 64 token vectors into one summary vector (mean-pool), then run it through a tiny network: Linear → ReLU → Linear → tanh. The final tanh squashes the answer into [−1, +1]: about +1 means the side to move looks winning, −1 losing, 0 roughly equal.',
    why: 'Alongside the move probabilities, a single score for how good the position is helps judge the situation — and here that judgment is learned purely from how real human games turned out.',
  },

  // ---- op-level kinds (math lens) ----
  attention: {
    title: 'Geometry self-attention',
    what: 'Each square decides how much attention to pay to every other square, then gathers information from the ones it cares about.',
    how: 'Every square produces a query (q — “what am I looking for”), a key (k — “what I offer”), and a value (v — “what I’d hand over”). The match between two squares is score = (q·k)/√d + a learned bias for their relative board offset. A softmax turns each square’s 64 scores into weights that add to 1, and the square’s new content is that weighted blend of the others’ values — computed in several independent “heads” and then combined.',
    why: 'This is how far-apart squares talk to each other in a single step — a rook can attend straight down its file. Splitting the work into heads lets the model track several kinds of relationship at once.',
    principle: 'The relative-offset bias starts at exactly zero and is shaped only by training on games. The model discovers for itself which offsets (knight L-shapes, shared diagonals, …) matter — none of that geometry is hand-coded.',
  },
  softmax: {
    title: 'Softmax',
    what: 'Turns a row of raw scores into probabilities that add up to 100%.',
    how: 'p_j = e^(s_j) / Σ over k of e^(s_k): exponentiate every score, then divide by their total. Bigger scores get exponentially more weight, and it’s the gaps between scores (not how large they are overall) that decide how sharply the top one wins.',
    why: 'It converts “how much do I prefer each option” into a clean, smoothly adjustable set of weights — exactly what attention needs to softly blend squares, and what the policy head needs to turn move scores into move probabilities.',
  },
  ffn: {
    title: 'Feed-forward network',
    what: 'Lets each square rework its own features after it has gathered information from the others.',
    how: 'Two linear layers with a smooth gate between them: expand from d up to a wider 4d, apply GELU (a softer version of ReLU that lets a little of the negative part through), then shrink back to d. The same weights run on every square independently.',
    why: 'Attention moves information between squares; this step gives the model room to transform and recombine whatever each square has just learned.',
  },
  residual: {
    title: 'Residual add',
    what: 'Adds a step’s output back onto its input instead of replacing it.',
    how: 'x_out = x_in + step(LayerNorm(x_in)). The “residual stream” running through the network is only ever nudged by each step, never overwritten.',
    why: 'Keeping a straight-through path means information from early layers survives to the end, and the network can be made deep without the training signal fading — each block just makes a small, safe edit.',
  },
};

/** Look up the explanation card for a stage/op kind, or undefined. */
export function content(kind: string): ContentCard | undefined {
  return CONTENT[kind];
}
