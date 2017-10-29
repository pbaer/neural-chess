# -*- coding: utf-8 -*-
from azure.storage.blob import BlockBlobService
from azure.storage.blob import ContentSettings
import keras
from keras.backend.tensorflow_backend import set_session
from keras.layers import Dense, Activation
from keras.models import Sequential
import os
import tensorflow as tf
import time

# Use this to prevent 100% GPU memory usage
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.4
#set_session(tf.Session(config=config))

def create_model():
    model = Sequential([
        Dense(3000, input_shape=(384,)),
        Activation('relu'),
        Dense(3000),
        Activation('relu'),
        Dense(3000),
        Activation('relu'),
        Dense(3000),
        Activation('relu'),
        Dense(4096),
        Activation('softmax')])
    compile_model(model)
    return model

def compile_model(model):
    model.compile(optimizer='adadelta',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

def load_model(filename):
    if filename.endswith('.json'):
        json_filename = filename
        h5_filename = filename[:-5] + '.h5'
    else: # assume local filename root only
        json_filename = 'model/' + filename + '.json'
        h5_filename = 'model/' + filename + '.h5'
    json_filename.replace('\\', '/');
    h5_filename.replace('\\', '/');
    with open(json_filename, 'r') as json_file:
        model_json = json_file.read()
        json_file.close()
    model = keras.models.model_from_json(model_json)
    model.load_weights(h5_filename)
    compile_model(model)
    return model

def save_model(model, filename):
    filename = 'model/' + filename
    model.save_weights(filename + '.h5')
    model_json = model.to_json()
    with open(filename + '.json', 'w') as json_file:
        json_file.write(model_json)

def create_blob_service():
    account_name = None
    account_key = None
    with open('.azurekey.txt') as file:
        account_name = file.readline().strip()
        account_key = file.readline().strip()
    return BlockBlobService(account_name=account_name, account_key=account_key)

def enumerate_local_models(filename_root=''):
    local_models = []
    for filename in os.listdir('model'):
        if not filename.startswith(filename_root) or not filename.endswith('.json'):
            continue
        local_models.append('model/' + filename)
    local_models.sort()
    return local_models

def enumerate_remote_models(blob_service, filename_root=''):
    blob_models = []
    blobs = blob_service.list_blobs('neural-chess')
    for blob in blobs:
        if not blob.name.startswith('model/' + filename_root) or not blob.name.endswith('.json'):
            continue
        blob_models.append(blob.name)
    blob_models.sort()
    return blob_models

def upload_blob_model(blob_service, json_filename):
    h5_filename = json_filename[:-5] + '.h5'
    blob_service.create_blob_from_path('neural-chess', h5_filename, h5_filename, content_settings=ContentSettings(content_type='application/octet-stream'))
    blob_service.create_blob_from_path('neural-chess', json_filename, json_filename, content_settings=ContentSettings(content_type='application/json'))

def download_blob_model(blob_service, json_filename):
    h5_filename = json_filename[:-5] + '.h5'
    blob_service.get_blob_to_path('neural-chess', h5_filename, h5_filename)
    blob_service.get_blob_to_path('neural-chess', json_filename, json_filename)

def synchronize_blob_models(blob_service):
    local_models = enumerate_local_models()
    remote_models = enumerate_remote_models(blob_service)
    for local in local_models:
        if local not in remote_models:
            print("Uploading %s..." % local)
            upload_blob_model(blob_service, local)
    for remote in remote_models:
        if remote not in local_models:
            print("Downloading %s..." % remote)
            download_blob_model(blob_service, remote)

def synchronize_blob_models_forever():
    blob_service = create_blob_service()
    while os.path.isfile('.stopsync') == False:
        synchronize_blob_models(blob_service)
        time.sleep(30)
    os.remove('.stopsync')
