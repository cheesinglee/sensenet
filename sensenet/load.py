import sensenet.importers
np = sensenet.importers.import_numpy()

from sensenet.constants import NUMERIC, CATEGORICAL, IMAGE_PATH

def get_index(alist, value):
    try:
        return alist.index(value)
    except ValueError:
        return -1

def load_points(model, points):
    preprocessors = model['preprocess']
    outarray = np.zeros((len(points), len(preprocessors)), dtype=np.float32)

    for i, proc in enumerate(preprocessors):
        pidx = proc['index']
        values = proc.get('values', None)

        if proc['type'] == NUMERIC:
            for j, p in enumerate(points):
                outarray[j, i] = p[pidx]
        elif proc['type'] == CATEGORICAL:
            for j, p in enumerate(points):
                outarray[j, i] = get_index(values, p[pidx])
        elif proc['type'] == IMAGE_PATH:
            pass
        else:
            raise ValueError('Unknown processor type "%s"' % proc['type'])

    return {'input_X': outarray}
