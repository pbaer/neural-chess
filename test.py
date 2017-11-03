# -*- coding: utf-8 -*-
from model import load_model
from stockfish import Stockfish
from play import play_engine
from play import print_stats
import tensorflow as tf

model = load_model('modelD_e0027')
stats = play_engine(model, 10)
print()
print_stats(stats)