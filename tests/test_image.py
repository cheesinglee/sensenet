import time

import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()

from PIL import Image

from sensenet.graph.image import image_tensor

PATH_PREFIX = 'tests/data/images/digits/'
EXTRA_PARAMS = {'path_prefix': PATH_PREFIX}

def read_image(path, input_image_shape):
    img = Image.open(path)

    if input_image_shape:
        in_shape = input_image_shape[:-1]

        if input_image_shape[-1] == 1:
            itype = 'L'
        elif input_image_shape[-1] == 3:
            itype = 'RGB'
        else:
            raise ValueError('%d is not a valid number of channels' %
                             input_image_shape[-1])
    else:
        in_shape = img.size
        itype = 'RGB'

    img = img.convert(itype)

    if img.size != in_shape:
        if img.size[0] * img.size[1] > in_shape[0] * in_shape[1]:
            img = img.resize(in_shape, Image.NEAREST)
        else:
            img = img.resize(in_shape, Image.NEAREST)

    return img

def show_and_wait(img_like, wait_time=2):
    anarray = np.array(img_like, dtype=np.uint8)

    if len(anarray.shape) == 3 and anarray.shape[-1] == 1:
        anarray = np.squeeze(anarray)

    print(anarray.shape)

    if anarray.shape[-1] == 3:
        img = Image.fromarray(anarray, 'RGB')
    else:
        img = Image.fromarray(anarray, 'L')

    img.show()
    time.sleep(wait_time)

def images_equal(tf_out, pil_out):
    tfints = tf_out
    pilints = np.array(pil_out, dtype=np.uint8)

    if len(pilints.shape) == 2:
        pilints = np.expand_dims(pilints, axis=2)

    # for i, rows in enumerate(zip(tfints, pilints)):
    #     trow, prow = rows
    #     for j, cols in enumerate(zip(trow, prow)):
    #         tpx, ppx = cols
    #         for t, p in zip(tpx, ppx):
    #             if t != p:
    #                 print((i, j), t, p)

    assert np.array_equal(tfints.shape, pilints.shape)
    return np.array_equal(tfints, pilints)

def test_tf_reading_single():
    shapes = [[32, 32, 3], [64, 48, 3], [28, 28, 1], [64, 48, 1], [16, 24, 3]]
    one_per = dict(EXTRA_PARAMS)

    images = ['3/10.png', '8/125.png']
    arow = [images[0], images[1]]
    acol = [[images[0]], [images[1]]]

    with tf.Session() as sess:
        one_per['image_paths'] = tf.placeholder(tf.string, (None, 1))

        for shape in shapes:
            in_shape = [shape[i] for i in [1, 0, 2]]

            truth = {p: read_image(PATH_PREFIX + p, shape) for p in images}
            tshape = list(np.array(truth[images[0]]).shape)
            assert tshape[:2] == in_shape[:2]

            one_out = image_tensor(one_per, shape)
            one_in = one_per['image_paths']
            single = one_out.eval({one_in: np.array([[images[0]]])})
            double = one_out.eval({one_in: np.array(acol)})

            assert 5 == len(single.shape) == len(double.shape)
            assert 1 == single.shape[1] == double.shape[1]
            assert 1 == single.shape[0]
            assert 2 == double.shape[0]
            assert shape[-1] == single.shape[-1] == double.shape[-1]

            for ten in [single, double]:
                for row in ten:
                    for image in row:
                        assert len(image.shape) == 3
                        assert in_shape == list(image.shape)

            assert images_equal(single[0][0], truth[images[0]])
            assert images_equal(double[0][0], truth[images[0]])
            assert images_equal(double[1][0], truth[images[1]])

def test_tf_reading_multi():
    shapes = [[32, 32, 3], [64, 48, 1]]
    two_per = dict(EXTRA_PARAMS)

    images = ['3/10.png', '8/125.png']
    arow = [images[0], images[1]]
    acol = [[images[0]], [images[1]]]

    with tf.Session() as sess:
        two_per['image_paths'] = tf.placeholder(tf.string, (None, 2))

        for shape in shapes:
            truth = {p: read_image(PATH_PREFIX + p, shape) for p in images}
            in_shape = [shape[i] for i in [1, 0, 2]]

            truth = {p: read_image(PATH_PREFIX + p, shape) for p in images}
            tshape = list(np.array(truth[images[0]]).shape)
            assert tshape[:2] == in_shape[:2]

            two_out = image_tensor(two_per, shape)
            two_in = two_per['image_paths']
            one_row = two_out.eval({two_in: np.array([arow])})
            triple = two_out.eval({two_in: np.array([arow, arow, arow[::-1]])})

            assert 5 == len(one_row.shape) == len(triple.shape)
            assert 2 == one_row.shape[1] == triple.shape[1]
            assert 1 == one_row.shape[0]
            assert 3 == triple.shape[0]
            assert shape[-1] == one_row.shape[-1] == triple.shape[-1]

            for ten in [one_row, triple]:
                for row in ten:
                    for image in row:
                        assert len(image.shape) == 3
                        assert in_shape == list(image.shape)

            assert images_equal(one_row[0][0], truth[arow[0]])
            assert images_equal(one_row[0][1], truth[arow[1]])
            assert images_equal(triple[0][0], triple[1][0])
            assert images_equal(triple[0][1], triple[1][1])
            assert not images_equal(triple[0][0], triple[2][0])
            assert not images_equal(triple[0][1], triple[2][1])
            assert images_equal(triple[0][0], triple[2][1])
            assert images_equal(triple[0][1], triple[2][0])
