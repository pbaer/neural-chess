# -*- coding: utf-8 -*-
from model import load_model
from stockfish import Stockfish
from play import play_engine
from play import print_stats
import tensorflow as tf

model = load_model('modelDb_e0037')
engine = Stockfish(depth=0, param={'Skill Level':0})
stats = play_engine(model, engine, 10)
print_stats(stats)