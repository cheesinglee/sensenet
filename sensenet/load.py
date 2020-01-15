import sensenet.importers
np = sensenet.importers.import_numpy()

from sensenet.constants import NUMERIC, CATEGORICAL, IMAGE_PATH
from sensenet.graph.image import read_fn

def get_index(alist, value):
    try:
        return alist.index(value)
    except ValueError:
        return -1

def load_points(model, points, image_directory):
    preprocs = model['preprocess']
    nimages = [p['type'] for p in preprocs].count(IMAGE_PATH)
    img_idx = 0

    inputs = {}

    if nimages:
        image_network = model['image_network']
        image_shape = image_network['metadata']['input_image_shape']
        image_input_shape = [len(points), nimages] + image_shape
        inputs['image_X'] = np.zeros(image_input_shape, dtype=np.float32)

        reader = read_fn(image_network, image_directory)
    else:
        reader = None

    inputs['raw_X'] = np.zeros((len(points), len(preprocs)), dtype=np.float32)

    for i, proc in enumerate(preprocs):
        pidx = proc['index']
        values = proc.get('values', None)

        if proc['type'] == NUMERIC:
            for j, p in enumerate(points):
                inputs['raw_X'][j, i] = p[pidx]
        elif proc['type'] == CATEGORICAL:
            for j, p in enumerate(points):
                inputs['raw_X'][j, i] = get_index(values, p[pidx])
        elif proc['type'] == IMAGE_PATH:
            for j, p in enumerate(points):
                inputs['image_X'][j, img_idx, ...] = reader(p[pidx])

            img_idx += 1
        else:
            raise ValueError('Unknown processor type "%s"' % proc['type'])

    return inputs
