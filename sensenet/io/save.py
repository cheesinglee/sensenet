import os
import tempfile

import sensenet.importers
tf = sensenet.importers.import_tensorflow()

from sensenet.constants import NUMERIC, CATEGORICAL, IMAGE_PATH

TYPE_ASSET_NAME = 'sensenet_input_types'
TYPE_CODES = {NUMERIC: 0, CATEGORICAL: 1, IMAGE_PATH: 2}

def assets_for_deepnet(network, uid):
    tmpdir = tempfile.mkdtemp(suffix='_sensenet', prefix=uid)
    tmpfile = os.path.join(tmpdir, TYPE_ASSET_NAME)

    with open(tmpfile, 'w') as fout:
        for p in network['preprocess']:
            fout.write("%d " % TYPE_CODES[p['type']])

    return {
        'input_types': tmpfile,
        'temporary_directory': tmpdir
    }

def write_model(tf_model, assets_dict, output_dir):
    for k in assets_dict:
        if k != 'temporary_directory':
            setattr(tf_model, k, tf.saved_model.Asset(assets_dict[k]))

    tf_model.save(output_dir)

    for k in assets_dict:
        if k != 'temporary_directory':
            delattr(tf_model, k)
            os.remove(assets_dict[k])

    if 'temporary_directory' in assets_dict:
        os.rmdir(assets_dict['temporary_directory'])

def write_deepnet(tf_model, network, output_dir):
    assets = assets_for_deepnet(network, output_dir)
    write_model(tf_model, assets, output_dir)
