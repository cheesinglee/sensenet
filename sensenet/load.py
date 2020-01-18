import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()

from sensenet.constants import NUMERIC, CATEGORICAL, IMAGE_PATH

def get_index(alist, value):
    try:
        return alist.index(value)
    except ValueError:
        return -1

def load_points(model, points, image_directory):
    preprocs = model['preprocess']
    rows = len(points)
    inputs = {'raw_X': np.zeros((rows, len(preprocs)), dtype=np.float32)}

    for i, proc in enumerate(preprocs):
        pidx = proc['index']
        values = proc.get('values', None)

        if proc['type'] == NUMERIC:
            for j, p in enumerate(points):
                inputs['raw_X'][j, i] = float(p[pidx])
        elif proc['type'] == CATEGORICAL:
            for j, p in enumerate(points):
                inputs['raw_X'][j, i] = get_index(values, str(p[pidx]))
        elif proc['type'] == IMAGE_PATH:
            if 'image_paths' not in inputs:
                inputs['image_paths'] = [list() for _ in range(rows)]

            for j, p in enumerate(points):
                inputs['image_paths'][j].append(str(p[pidx]))
        else:
            raise ValueError('Unknown processor type "%s"' % proc['type'])

    if 'image_paths' in inputs:
        inputs['image_paths'] = np.array(inputs['image_paths'])

    return inputs
