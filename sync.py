# -*- coding: utf-8 -*-
from azure.storage.blob import BlockBlobService
from azure.storage.blob import ContentSettings
import os
import time

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

def synchronize_blob_models(blob_service, upload=True, download=True):
    local_models = enumerate_local_models()
    remote_models = enumerate_remote_models(blob_service)
    if upload:
        for local in local_models:
            if local not in remote_models:
                print("Uploading %s..." % local)
                upload_blob_model(blob_service, local)
    if download:
        for remote in remote_models:
            if remote not in local_models:
                print("Downloading %s..." % remote)
                download_blob_model(blob_service, remote)

def synchronize_blob_models_forever(upload=True, download=True):
    blob_service = create_blob_service()
    while os.path.isfile('.stopsync') == False:
        synchronize_blob_models(blob_service, upload, download)
        time.sleep(30)
    os.remove('.stopsync')