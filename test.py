# -*- coding: utf-8 -*-
import torch
from model import load_model
from play import create_engine, play_engine, print_stats

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model('model', device=device)
engine = create_engine(depth=0, skill_level=0)

try:
    stats = play_engine(model, engine, 10, device=device)
    print_stats(stats)
finally:
    engine.quit()
