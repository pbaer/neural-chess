# -*- coding: utf-8 -*-
"""Fold an eval-mode Conv2d(bias=False) + BatchNorm2d into one conv-with-bias,
so the browser engine needs no BatchNorm op (one fewer thing to reimplement and
one fewer parity surface). BN eps matches PyTorch's default (1e-5)."""
import numpy as np


def fold_conv_bn(conv_w, gamma, beta, run_mean, run_var, eps=1e-5):
    """conv_w: (O,I,kh,kw); BN params: (O,). Returns (W', b') float32."""
    s = gamma / np.sqrt(run_var + eps)
    W = (conv_w * s.reshape(-1, 1, 1, 1)).astype(np.float32)
    b = (beta - run_mean * s).astype(np.float32)
    return W, b
