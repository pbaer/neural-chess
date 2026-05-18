# -*- coding: utf-8 -*-
"""Thin dispatcher: parses CLI, loads a PolicyEngine, drives UCI protocol."""
import sys

import torch

from src.inference_api import load_policy_engine
from src.uci_protocol import run_uci


def main(model_filename):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_filename is None:
        model_filename = 'model'
    policy_engine = load_policy_engine(model_filename, device=device)
    run_uci(policy_engine)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) >= 2 else None)
