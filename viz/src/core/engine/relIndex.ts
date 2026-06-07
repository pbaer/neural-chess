// Recompute the (64,64) relative-position index for the geometry bias.
//
// This buffer (`rel_idx` in the PyTorch model) is registered with
// persistent=False, so it is NOT in the state_dict and NOT exported in the
// capsule — we regenerate it deterministically here, exactly matching
// src/v3/model.py::_rel_index().
//
//   for query square i, key square j:
//     dr = (i//8 - j//8) + 7   in [0,14]
//     df = (i%8  - j%8 ) + 7   in [0,14]
//     relIdx[i][j] = dr*15 + df   in [0,224]
//
// The per-head learnable table `rel_bias` has shape (n_heads, 225); the bias
// added to attention score[h][i][j] is rel_bias[h][relIdx[i][j]].

/** Returns a flat Int32Array of length 64*64; index [i*64 + j]. */
export function relIndex(): Int32Array {
  const idx = new Int32Array(64 * 64);
  for (let i = 0; i < 64; i++) {
    const ri = i >> 3;
    const fi = i & 7;
    for (let j = 0; j < 64; j++) {
      const rj = j >> 3;
      const fj = j & 7;
      const dr = ri - rj + 7;
      const df = fi - fj + 7;
      idx[i * 64 + j] = dr * 15 + df;
    }
  }
  return idx;
}
