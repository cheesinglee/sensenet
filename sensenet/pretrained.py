import os
import gzip
import json
import requests

USER_HOME = os.path.expanduser('~')
CACHE_DIRECTORY = os.path.join(USER_HOME, '.bigml_sensenet')
CNN_METADATA_FILE = 'sensenet_metadata.json'

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
    if network_name in PRETRAINED_CNN_METADATA:
        return PRETRAINED_CNN_METADATA[network_name]
    else:
        raise KeyError('%s is not a pretrained network' % network_name)

def load_pretrained_weights(model, network):
    metadata = network['metadata']
    network_path = metadata['base_image_network'] + '_' + metadata['version']
    archive = cache_resource_path(network_path + '.h5')

    model.load_weights(archive)
