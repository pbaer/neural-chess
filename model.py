# -*- coding: utf-8 -*-
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

def create_model():
    model = Sequential([
        Dense(2000, input_shape=(384,)),
        Activation('relu'),
        Dense(2000),
        Activation('relu'),
        Dense(2000),
        Activation('relu'),
        Dense(4096),
        Activation('softmax')])
    compile_model(model)
    return model

def compile_model(model):
    model.compile(optimizer='adadelta',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

def load_model(name):
    with open('model/' + name + '.json', 'r') as json_file:
        model_json = json_file.read()
        json_file.close()
    model = keras.models.model_from_json(model_json)
    model.load_weights('model/' + name + '.h5')
    compile_model(model)
    return model

def save_model(model, name):
    model_json = model.to_json()
    with open('model/' + name + '.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights('model/' + name + '.h5')
