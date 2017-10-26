# -*- coding: utf-8 -*-
from model import load_model
from stockfish import Stockfish
from play import play_engine

model = load_model('modelB3')
engine = Stockfish(depth=0, param={'Skill Level':0})
play_engine(model, engine, 10)