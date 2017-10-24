# -*- coding: utf-8 -*-
"""
    pystockfish
    ~~~~~~~~~~~~~~~

    Wraps the Stockfish chess engine.  Assumes stockfish is
    executable at the root level.

    Built on Ubuntu 12.1 tested with Stockfish 120212.
    
    :copyright: (c) 2013 by Jarret Petrillo.
    :license: GNU General Public License, see LICENSE for more details.
"""

# PBaer: forked from pystockfish and renamed Engine to Stockfish
# bestmove() doesn't properly handle checkmate case...

import subprocess
from random import randint

class Stockfish(subprocess.Popen):
    """
    This initiates the Stockfish chess engine with Ponder set to False.
    'param' allows parameters to be specified by a dictionary object with 'Name' and 'value'
    with value as an integer.

    i.e. the following explicitly sets the default parameters
    {
        "Contempt Factor": 0,
        "Min Split Depth": 0,
        "Threads": 1,
        "Hash": 16,
        "MultiPV": 1,
        "Skill Level": 20,
        "Move Overhead": 30,
        "Minimum Thinking Time": 20,
        "Slow Mover": 80,
    }

    If 'rand' is set to False, any options not explicitly set will be set to the default
    value.

    -----
    USING RANDOM PARAMETERS
    -----
    If you set 'rand' to True, the 'Contempt' parameter will be set to a random value between
    'rand_min' and 'rand_max' so that you may run automated matches against slightly different
    engines.
    """

    def __init__(self, depth=2, ponder=False, param={}):
        subprocess.Popen.__init__(self,
                                  'bin/stockfish',
                                  universal_newlines=True,
                                  stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, )
        self.depth = str(depth)
        self.ponder = ponder
        self.put('uci')
        if not ponder:
            self.setoption('Ponder', False)

        base_param = {
            "Write Debug Log": "false",
            "Contempt": 0,
            "Min Split Depth": 0,
            "Threads": 1,
            "Hash": 16,
            "MultiPV": 1,
            "Skill Level": 20,
            "Move Overhead": 30,
            "Minimum Thinking Time": 20,
            "Slow Mover": 80,
            "UCI_Chess960": "false",
        }

        base_param.update(param)
        self.param = base_param
        for name, value in list(base_param.items()):
            self.setoption(name, value)

    def newgame(self):
        """
        Calls 'ucinewgame' - this should be run before a new game
        """
        #self.setoption('Contempt', randint(-100, 100)) # (PBaer) Always randomize play
        self.put('ucinewgame')
        self.isready()

    def put(self, command):
        self.stdin.write(command + '\n')
        self.stdin.flush()

    def flush(self):
        self.stdout.flush()

    def setoption(self, optionname, value):
        self.put('setoption name %s value %s' % (optionname, str(value)))
        stdout = self.isready()
        if stdout.find('No such') >= 0:
            print("stockfish was unable to set option %s" % optionname)

    def setposition(self, moves=[]):
        """
        Move list is a list of moves (i.e. ['e2e4', 'e7e5', ...]) each entry as a string.  Moves must be in full algebraic notation.
        """
        self.put('position startpos moves %s' % self._movelisttostr(moves))
        self.isready()

    def setfenposition(self, fen):
        """
        set position in fen notation.  Input is a FEN string i.e. "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        """
        self.put('position fen %s' % fen)
        self.isready()

    def go(self):
        self.put('go depth %s' % self.depth)

    def _movelisttostr(self, moves):
        """
        Concatenates a list of strings
        """
        movestr = ''
        for h in moves:
            movestr += h + ' '
        return movestr.strip()

    def bestmove(self):
        last_line = ""
        self.go()
        while True:
            text = self.stdout.readline().strip()
            split_text = text.split(' ')
            if split_text[0] == 'bestmove':
                return {'move': split_text[1],
                        #'ponder': split_text[3], # (PBaer) This doesn't work if the best move results in checkmate
                        'info': last_line
                }
            last_line = text

    def isready(self):
        """
        Used to synchronize the python engine object with the back-end engine.  Sends 'isready' and waits for 'readyok.'
        """
        self.put('isready')
        while True:
            text = self.stdout.readline().strip()
            if text == 'readyok':
                return text