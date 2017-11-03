# neural-chess

This is a simple chess engine that uses a neural network trained on historical games from the http://chessgames.com database.
It has not been taught anything about how to play chess or how to evaluate board positions. It does not look ahead to explore the consequences of possible moves. It only uses an existing chess library to prevent it from making illegal moves
(if it's about to attempt an illegal move, it keeps examining its next-best guess for that board position until it finds a legal move;
currently, about 80% of its move attempts are legal).

The model uses five fully-connected hidden layers with 3,000 nodes per layer. The input is the board position and the output
is the predicted move, based on observing board + move pairs by winning players in the historical game database.

Against Stockfish (a popular chess engine), on its most rudimentary setting (depth=0, skill=0), the model wins about 1.6% of games (and draws 10.6% of games). Yes, **it's really quite terrible** by modern chess engine standards. But it seems to hold its own against novice human players (i.e. me) and I was pleasantly surprised it is ever able to win against a real chess engine at all, considering that it has learned everything about how to play by itself.

## Setup

Make sure these Python prerequisites are installed:

`pip install tensorflow keras python-chess azure-storage-blob`

You can get a trained model by downloading these two files into the same directory:

https://alpenglow.blob.core.windows.net/neural-chess/model/modelD_e0027.json

https://alpenglow.blob.core.windows.net/neural-chess/model/modelD_e0027.h5

## Play

To try out interactive play, I recommend using Arena (http://www.playwitharena.com):

1. Select **Engines | Manage**
2. Select **New**
3. Browse to the location of `python.exe`
4. Set **Command Line Parameters** to `<your repo root dir>\neural-chess\uci.py <path to the model JSON file>`
5. Select **Start this engine right now** and give it a few seconds to load up (the little gray computer icon below the board should turn black)
7. Start a new game and **let the engine play white** (er, I only trained it on white. I should fix that). You do this by selecting **Game | Move Now!** to let it go first.
8. Subsequently, the engine will automatically move white after you move black.

## Engine vs. Engine

To play repeatedly against a real chess engine, tweak `test.py` with the correct model file location and desired number of games:

```
model = load_model(<path to the model JSON file>)
engine = Stockfish(depth=0, param={'Skill Level':0})
play_engine(model, engine, <number of games>)
```

*(feel free to use stronger Stockfish settings if you have more faith in my engine than I do)*
