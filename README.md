# neural-chess

This is a simple chess engine that uses a neural network trained on historical games from the http://chessgames.com database.
It has not been taught anything about how to play chess, other than using a python chess library to prevent it from making illegal moves
(if it's about to attempt an illegal move, it keeps examining its next-best guess for that board position until it finds a legal move;
currently, about 70% of its move attempts are legal).

The model uses a three fully-connected hidden layers with 2,000 nodes per layer. The input is the board position and the output
is the predicted move, based on observing board + move pairs by winning players in the historical game database.

Against Stockfish (a popular chess engine), on its most rudimentary setting (depth=0, skill=0), the model wins about
1% of games (and draws 4% of games). Yes, **it's really quite terrible** by modern chess engine standards. But it seems to hold its own
against novice human players (i.e. me) and I was pleasantly surprised it is ever able to win against a real chess engine at all, considering
that it has learned everything about how to play by itself.

## Setup

Make sure these Python prerequisites are installed:

`pip install tensorflow keras python-chess`

You can get a trained model by downloading these two files into the same directory:

https://alpenglow.blob.core.windows.net/neural-chess/model/modelB3.json

https://alpenglow.blob.core.windows.net/neural-chess/model/modelB3.h5

## Play

To try out interactive play, I recommend using Arena (http://www.playwitharena.com):

1. Select **Engines | Manage**
2. Select **New**
3. Set **Command Line** to the location of `python.exe`
4. Set **Command Line Parameters** to `<your repo root dir>\neural-chess\uci.py <path to the model JSON file>`
5. Start a new game and **let the engine play white** (er, I only trained it on white. I should fix that). You do this by selecting **Game | Move Now!** to let it go first.
6. Subsequently, the engine will automatically move white after you move black.

## Engine vs. Engine

To play repeatedly against a real chess engine, tweak `test.py` with the correct model file location and desired number of games:

```
model = load_model(<path to the model JSON file>)
engine = Stockfish(depth=0, param={'Skill Level':0})
play_engine(model, engine, <number of games>)
```

*(feel free to use stronger Stockfish settings if you have more faith in my engine than I do)*
