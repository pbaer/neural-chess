# -*- coding: utf-8 -*-
"""Smoke test: drive the v1 model against Stockfish for 10 quick games."""
import os
import sys

# Allow `python eval/v1/test.py` from the repo root or anywhere else.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from src.engine import create_engine  # noqa: E402
from src.game_loop import play_engine  # noqa: E402
from src.inference_api import load_policy_engine  # noqa: E402
from src.stats import print_stats  # noqa: E402


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Default to the v1 archive location; override on the CLI if you want
    # a different checkpoint.
    ckpt = sys.argv[1] if len(sys.argv) >= 2 else 'model/v1/checkpoints/model_e0015.pt'
    policy_engine = load_policy_engine(ckpt, device=device)
    stockfish = os.path.join(_REPO_ROOT, 'bin', 'stockfish.exe')
    engine = create_engine(path=stockfish, depth=0, skill_level=0)

    try:
        stats = play_engine(policy_engine, engine, 10)
        print_stats(stats)
    finally:
        engine.quit()


if __name__ == '__main__':
    main()
