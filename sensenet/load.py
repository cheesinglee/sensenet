import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()

from sensenet.constants import NUMERIC, CATEGORICAL, IMAGE_PATH

def load_points(preprocessors, points):
    rows = len(points)
    cols = len(preprocessors)

    inputs = {
        'numeric': np.zeros((rows, cols), dtype=np.float32),
        'string': [list() for _ in range(rows)]
    }

    for i, proc in enumerate(preprocessors):
        pidx = proc['index']
        values = proc.get('values', None)

        if proc['type'] == NUMERIC:
            for j, p in enumerate(points):
                inputs['numeric'][j, i] = float(p[pidx])
        elif proc['type'] in [IMAGE_PATH, CATEGORICAL]:
            for j, p in enumerate(points):
                inputs['string'][j].append(str(p[pidx]))
        else:
            raise ValueError('Unknown processor type "%s"' % proc['type'])

    inputs['string'] = np.array(inputs['string'])

    return inputs
