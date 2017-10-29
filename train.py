# -*- coding: utf-8 -*-
import numpy as np
import os
from model import save_model

class TrainingSet():
    FEATURES = 6 * 8 * 8 # 6 unique piece types each on an 8x8 integer board; 1 for white, -1 for black
    OUTPUTS = 64 * 64 # Map from source square to target square. Larger than necessary (includes many impossible moves for any piece)
    
    def __init__(self, max_rows):
        self.X = np.zeros((max_rows, self.FEATURES), dtype='int8')
        self.Y = np.zeros((max_rows, self.OUTPUTS), dtype='int8')
        self.rows = 0
        self.max_rows = max_rows
        
    def reset(self):
        self.rows = 0

    def get(self):
        return self.X[0:self.rows,:], self.Y[0:self.rows,:]

    def is_full(self):
        return self.rows == self.max_rows

    def add_from_file(self, filename):
        data = np.load(filename)
        return self.add_from_data(data)

    def add_from_data(self, data):
        data_rows = data['meta'][0]
        if (self.rows + data_rows > self.max_rows):
            return False
        data_X = data['X']
        data_Y = data['Y']
        self.X[self.rows:(self.rows + data_rows),:] = data_X[0:data_rows,:]
        self.Y[self.rows:(self.rows + data_rows),:] = data_Y[0:data_rows,:]
        self.rows += data_rows
        return True

    def add_from_folder(self, foldername, printonly=False):
        total_rows = 0
        for filename in os.listdir(foldername):
            if not filename.endswith('.npz'):
                continue
            data = np.load(foldername + '/' + filename)
            data_rows = data['meta'][0]
            print("%d rows in %s" % (data_rows, filename))
            total_rows += data_rows
            if printonly:
                continue
            if not self.add_from_data(data):
                total_rows -= data_rows
                print("Training set full, not adding this file.")
                break
        print("%d total rows (%.2fGB expanded)" % (total_rows, (float(total_rows) * (self.FEATURES + self.OUTPUTS))/(1024 * 1024 * 1024)))

    def add_row(self, x, y):
        if self.is_full():
            return False
        self.X[self.rows] = x
        self.Y[self.rows] = y
        self.rows += 1
        return True

    def save_to_file(self, filename):
        meta = np.ndarray((1), dtype=int)
        meta[0] = self.rows
        np.savez_compressed('data/' + filename, X=self.X[0:self.rows,:], Y=self.Y[0:self.rows,:], meta=meta)

def train_forever(model, training_set, save_filename, start_epoch):
#    train = TrainingSet(3600000)
#    train.add_from_file('training_set_0000-1999.npz')
    epoch = start_epoch
    train_X, train_Y = training_set.get()
    while os.path.isfile('.stop') == False:
        model.fit(train_X, train_Y, batch_size=10000, epochs=1)
        save_model(model, save_filename + '_e' + ("%04d" % epoch))
        epoch += 1
    os.remove('.stop')