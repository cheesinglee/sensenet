import os
import gzip
import json
import requests

USER_HOME = os.path.expanduser('~')
CACHE_DIRECTORY = os.path.join(USER_HOME, '.bigml_mimir')
CNN_METADATA_FILE = 'cnn_metadata.json'

S3_BUCKET = 'https://s3.amazonaws.com/bigml-cnns/'

def download_to_file(network_archive, path):
    print(('Downloading %s...' % (S3_BUCKET + network_archive)))
    req = requests.get(S3_BUCKET + network_archive)

    with open(path, 'wb') as fout:
        fout.write(req.content)

def cache_resource_path(resource_name):
    try:
        os.makedirs(CACHE_DIRECTORY)
    except OSError:
        pass

    cache_path = os.path.join(CACHE_DIRECTORY, resource_name)

    if (not os.path.exists(cache_path)) or os.path.isdir(cache_path):
        download_to_file(resource_name, cache_path)

    return cache_path

with open(cache_resource_path(CNN_METADATA_FILE), 'r') as fin:
    PRETRAINED_CNN_METADATA = json.load(fin)

def get_pretrained_network(network_name):
    return {
        'metadata': PRETRAINED_CNN_METADATA[network_name],
        'layers': None
    }

def cnn_resource_path(network, readout):
    metadata = network['metadata']
    network_path = metadata['base_image_network'] + '_' + metadata['version']

    if readout:
        archive = network_path + '_readout.json.gz'
    else:
        archive = network_path + '.json.gz'

    return cache_resource_path(archive)

def get_resource(network, readout):
    with gzip.open(cnn_resource_path(network, readout), 'rb') as fin:
        return json.loads(fin.read().decode('utf-8'))

def get_pretrained_layers(network):
    return get_resource(network, False)

def get_pretrained_readout(network):
    return get_resource(network, True)
