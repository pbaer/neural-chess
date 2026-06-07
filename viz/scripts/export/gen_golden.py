# -*- coding: utf-8 -*-
"""Generate golden parity vectors for the TS engine.

Runs the hero ChessModelV3 in fp32 eval on CPU over a fixed FEN set and dumps:
 - input planes (float + int8-truncated) per FEN
 - EVERY intermediate of the forward pass (stem/embed, tokenize, per-block
   ln1/attn-scores/attn-probs/attn-out/post-attn/ln2/ffn/post-ffn, final_ln,
   policy logits, value)
 - legal mask / argmax-over-legal / decoded move (rotated + un-rotated)
 - per-FEN board state (so the TS featurize/moves tests need no chess.js)
 - a dense erf reference grid (for the exact-erf GELU)

The tower is reconstructed manually (mirroring src/v3/model.py op-for-op) so the
fused SDPA's internal scores/probs are recoverable; we cross-check the manual
policy/value against the real model() output to guarantee fidelity.

GPU is OFF-LIMITS (training is using it): everything runs on CPU.

Usage (from repo root):
  .venv/Scripts/python.exe viz/scripts/export/gen_golden.py \
      --ckpt model/v3/v3.1-nano/v3.1-nano_e0015.pt \
      --out-dir viz/public/weights/v3.1-nano
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, ROOT)

import chess  # noqa: E402

from src.v2.featurize import featurize, rotate_square  # noqa: E402
from src.v2.moves import NUM_MOVES, decode_move, encode_move, legal_mask  # noqa: E402
from src.v3.inference import load_v3_model  # noqa: E402


# ---------------------------------------------------------------- board states

def board_from_moves(sans):
    b = chess.Board()
    for s in sans:
        b.push_san(s)
    return b


def make_cases():
    """Return [(name, chess.Board)]. The repetition cases are built via moves
    so is_repetition() is populated; the rest are loaded from FEN."""
    cases = []
    cases.append(('startpos', chess.Board()))
    # black to move, rotation, no ep
    cases.append(('black_to_move', chess.Board(
        'rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1')))
    # en passant available (white to move, exd6 legal)
    cases.append(('en_passant', chess.Board(
        'rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3')))
    # near promotion, white pawn a7
    cases.append(('near_promotion_white', chess.Board('8/P7/8/8/8/8/8/k6K w - - 0 1')))
    # near promotion, black pawn a2 (exercises rotation + promotion)
    cases.append(('near_promotion_black', chess.Board('K6k/8/8/8/8/8/p7/8 b - - 0 1')))
    # mixed castling rights (white K-side, black Q-side only)
    cases.append(('mixed_castling', chess.Board('r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1')))
    # twofold repetition (plane-20 = 0.5): startpos seen twice, white to move
    cases.append(('twofold_rep', board_from_moves(['Nf3', 'Nf6', 'Ng1', 'Ng8'])))
    # threefold repetition (plane-20 = 1.0): startpos seen three times
    cases.append(('threefold_rep', board_from_moves(
        ['Nf3', 'Nf6', 'Ng1', 'Ng8', 'Nf3', 'Nf6', 'Ng1', 'Ng8'])))
    return cases


def dump_board(b):
    pieces = []
    for sq, pc in b.piece_map().items():
        pieces.append({
            'square': sq,
            'color': 'w' if pc.color == chess.WHITE else 'b',
            'type': chess.piece_symbol(pc.piece_type),  # lowercase letter
        })
    return {
        'turn': 'w' if b.turn == chess.WHITE else 'b',
        'pieces': pieces,
        'castling': {
            'wk': b.has_kingside_castling_rights(chess.WHITE),
            'wq': b.has_queenside_castling_rights(chess.WHITE),
            'bk': b.has_kingside_castling_rights(chess.BLACK),
            'bq': b.has_queenside_castling_rights(chess.BLACK),
        },
        'ep': b.ep_square if b.ep_square is not None else None,
        'halfmove': b.halfmove_clock,
        'fullmove': b.fullmove_number,
        'isRep2': bool(b.is_repetition(2)),
        'isRep3': bool(b.is_repetition(3)),
    }


# ---------------------------------------------------------------- manual forward

@torch.no_grad()
def manual_forward(model, x):
    """Reconstruct ChessModelV3.forward op-for-op, returning every intermediate.
    `inter` keys match the TS TraceRecorder keys exactly."""
    cfg = model.config
    C = cfg.d_model
    nh = cfg.n_heads
    hd = C // nh
    inter = {}

    # --- stem (real modules => exact stem output the folded TS embed reproduces)
    h = model.relu(model.input_bn(model.input_conv(x)))   # (1,C,8,8)
    h = model.stem(h)
    inter['embed'] = h.squeeze(0).clone()                 # (C,8,8)

    # --- tokenize
    t = h.flatten(2).transpose(1, 2) + model.pos_emb       # (1,64,C)
    inter['tokenize'] = t.squeeze(0).clone()               # (64,C)

    for i, blk in enumerate(model.blocks):
        t_in = t
        ln1 = F.layer_norm(t_in, (C,), blk.ln1.weight, blk.ln1.bias, eps=1e-5)
        inter[f'block.{i}.ln1'] = ln1.squeeze(0).clone()

        qkv = F.linear(ln1, blk.attn.qkv.weight, blk.attn.qkv.bias)  # (1,64,3C)
        qkv = qkv.reshape(1, 64, 3, nh, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                    # (1,nh,64,hd)
        scores = (q @ k.transpose(-2, -1)) * blk.attn.scale  # (1,nh,64,64)
        if blk.attn.geometry_bias:
            bias = blk.attn.rel_bias[:, blk.attn.rel_idx].unsqueeze(0)  # (1,nh,64,64)
            scores = scores + bias
        probs = torch.softmax(scores, dim=-1)
        inter[f'block.{i}.attn.scores'] = scores.squeeze(0).clone()    # (nh,64,64)
        inter[f'block.{i}.attn.probs'] = probs.squeeze(0).clone()
        out = probs @ v                                    # (1,nh,64,hd)
        merged = out.transpose(1, 2).reshape(1, 64, C)
        attn = F.linear(merged, blk.attn.proj.weight, blk.attn.proj.bias)
        inter[f'block.{i}.attn'] = attn.squeeze(0).clone()

        post_attn = t_in + attn
        inter[f'block.{i}.postAttn'] = post_attn.squeeze(0).clone()
        ln2 = F.layer_norm(post_attn, (C,), blk.ln2.weight, blk.ln2.bias, eps=1e-5)
        inter[f'block.{i}.ln2'] = ln2.squeeze(0).clone()
        h1 = F.linear(ln2, blk.ffn[0].weight, blk.ffn[0].bias)
        a1 = F.gelu(h1)                                    # exact-erf (approximate='none')
        ffn = F.linear(a1, blk.ffn[2].weight, blk.ffn[2].bias)
        inter[f'block.{i}.ffn'] = ffn.squeeze(0).clone()
        post_ffn = post_attn + ffn
        inter[f'block.{i}.postFfn'] = post_ffn.squeeze(0).clone()

        # cross-check against the real block
        ref = blk(t_in)
        assert torch.allclose(ref, post_ffn, atol=1e-5, rtol=1e-4), \
            f'block {i} manual mismatch'
        t = post_ffn

    final = F.layer_norm(t, (C,), model.final_ln.weight, model.final_ln.bias, eps=1e-5)
    inter['final_ln'] = final.squeeze(0).clone()
    p = F.linear(final, model.policy_head.weight, model.policy_head.bias).reshape(1, NUM_MOVES)
    pooled = final.mean(dim=1)
    v1 = model.relu(F.linear(pooled, model.value_fc1.weight, model.value_fc1.bias))
    val = torch.tanh(F.linear(v1, model.value_fc2.weight, model.value_fc2.bias))
    inter['policy_logits'] = p.squeeze(0).clone()
    return inter, p, val


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device('cpu')
    model = load_v3_model(args.ckpt, device=device)
    model.eval()
    cfg = model.config

    blob = []
    off = 0  # float offset

    def add(arr):
        nonlocal off
        a = np.ascontiguousarray(np.asarray(arr, dtype='<f4'))
        flat = a.reshape(-1)
        entry = {'offset': off, 'length': int(flat.size), 'shape': list(a.shape)}
        blob.append(flat.tobytes())
        off += int(flat.size)
        return entry

    cases_out = []
    for name, board in make_cases():
        is_white = board.turn == chess.WHITE
        x_np = featurize(board)                       # (21,8,8) float32
        x = torch.from_numpy(x_np).unsqueeze(0).to(device)

        inter, p_manual, v_manual = manual_forward(model, x)
        # fidelity: manual == real model()
        p_model, v_model = model(x)
        assert torch.allclose(p_manual, p_model, atol=1e-5, rtol=1e-4), f'{name}: policy mismatch'
        assert torch.allclose(v_manual, v_model, atol=1e-5, rtol=1e-4), f'{name}: value mismatch'

        logits_np = p_manual.squeeze(0).numpy().astype(np.float64)
        value = float(v_manual.squeeze().item())

        rotated_board = board if is_white else board.mirror()
        mask = legal_mask(rotated_board)              # (4672,) bool
        legal_idx = [int(i) for i in np.flatnonzero(mask)]
        legal_logits = np.where(mask, logits_np, -1e30)
        best = int(legal_logits.argmax())
        decoded = decode_move(best, rotated_board)
        if is_white:
            move_real = decoded
        else:
            move_real = chess.Move(rotate_square(decoded.from_square),
                                   rotate_square(decoded.to_square),
                                   promotion=decoded.promotion)

        # encode/decode round-trip per legal move (in the rotated frame)
        legal_moves = []
        for mv in rotated_board.legal_moves:
            idx = encode_move(mv)
            dec = decode_move(idx, rotated_board)
            legal_moves.append({
                'from': mv.from_square, 'to': mv.to_square,
                'prom': mv.promotion if mv.promotion is not None else 0,
                'index': int(idx),
                'decFrom': dec.from_square, 'decTo': dec.to_square,
                'decProm': dec.promotion if dec.promotion is not None else 0,
            })

        # tensors
        tensors = {}
        tensors['planes'] = add(x_np)
        tensors['planesTrained'] = add(x_np.astype(np.int8).astype(np.float32))
        for key in ['embed', 'tokenize']:
            tensors[key] = add(inter[key].numpy())
        for i in range(cfg.n_blocks):
            for suf in ['ln1', 'attn.scores', 'attn.probs', 'attn', 'postAttn', 'ln2', 'ffn', 'postFfn']:
                k = f'block.{i}.{suf}'
                tensors[k] = add(inter[k].numpy())
        tensors['final_ln'] = add(inter['final_ln'].numpy())
        tensors['policy_logits'] = add(inter['policy_logits'].numpy())

        cases_out.append({
            'name': name,
            'fen': board.fen(),
            'turn': 'w' if is_white else 'b',
            'isWhite': bool(is_white),
            'realBoard': dump_board(board),
            'rotatedBoard': dump_board(rotated_board),
            'tensors': tensors,
            'value': value,
            'legalIndices': legal_idx,
            'bestLegalIndex': best,
            'decodedBest': {'from': decoded.from_square, 'to': decoded.to_square,
                            'prom': decoded.promotion if decoded.promotion is not None else 0},
            'moveReal': {'from': move_real.from_square, 'to': move_real.to_square,
                         'prom': move_real.promotion if move_real.promotion is not None else 0,
                         'uci': move_real.uci()},
            'legalMoves': legal_moves,
        })
        print(f'  case {name:22s} turn={"w" if is_white else "b":1s} '
              f'legal={len(legal_idx):3d} best={best:5d} move={move_real.uci()} value={value:+.4f}')

    # erf reference grid
    xs = np.linspace(-6.0, 6.0, 2401)
    extra = np.array([-0.5, 0.5, 1.0 / math.sqrt(2), 0.123456, 3.5, -3.5, 0.0])
    xs = np.concatenate([xs, extra])
    erf_ref = {'x': [float(v) for v in xs], 'erf': [float(math.erf(v)) for v in xs]}

    # rotate_square reference for all 64 squares
    rot_ref = [int(rotate_square(s)) for s in range(64)]

    os.makedirs(args.out_dir, exist_ok=True)
    raw = b''.join(blob)
    with open(os.path.join(args.out_dir, 'golden.bin'), 'wb') as f:
        f.write(raw)
    golden = {
        'model_id': model_id_from(args.ckpt),
        'checkpoint': os.path.relpath(args.ckpt, ROOT).replace('\\', '/'),
        'config': {
            'd_model': cfg.d_model, 'n_heads': cfg.n_heads, 'n_blocks': cfg.n_blocks,
            'ffn_mult': cfg.ffn_mult, 'stem_kernel': cfg.stem_kernel,
            'stem_blocks': cfg.stem_blocks, 'value_hidden': cfg.value_hidden,
            'geometry_bias': cfg.geometry_bias, 'input_planes': cfg.input_planes,
        },
        'numMoves': NUM_MOVES,
        'bin_floats': off,
        'erfGrid': erf_ref,
        'rotateSquareRef': rot_ref,
        'cases': cases_out,
    }
    with open(os.path.join(args.out_dir, 'golden.json'), 'w') as f:
        json.dump(golden, f)
    print(f'golden -> {args.out_dir}  ({len(cases_out)} cases, {off:,} floats, {len(raw):,} B)')


def model_id_from(ckpt):
    return os.path.basename(os.path.dirname(os.path.abspath(ckpt)))


if __name__ == '__main__':
    main()
