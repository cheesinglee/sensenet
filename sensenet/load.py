import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()

from sensenet.constants import NUMERIC, CATEGORICAL, IMAGE_PATH

def get_index(alist, value):
    try:
        return alist.index(value)
    except ValueError:
        return -1

def load_points(model, points):
    preprocessors = model['preprocess']

    rows = len(points)
    cols = len(preprocessors)

    inputs = {
        'numeric': np.zeros((rows, cols), dtype=np.float32),
        'image': [list() for _ in range(rows)]
    }

    for i, proc in enumerate(preprocessors):
        pidx = proc['index']
        values = proc.get('values', None)

        if proc['type'] == NUMERIC:
            for j, p in enumerate(points):
                inputs['numeric'][j, i] = float(p[pidx])
        elif proc['type'] == CATEGORICAL:
            for j, p in enumerate(points):
                inputs['numeric'][j, i] = get_index(values, str(p[pidx]))
        elif proc['type'] == IMAGE_PATH:
            for j, p in enumerate(points):
                inputs['image'][j].append(str(p[pidx]))
        else:
            raise ValueError('Unknown processor type "%s"' % proc['type'])

    inputs['image'] = np.array(inputs['image'])

    return inputs
