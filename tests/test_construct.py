import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()
kl = sensenet.importers.import_keras_layers()

from sensenet.layers.block import BlockLayer
from sensenet.layers.construct import layer_sequence
from sensenet.preprocess.image import get_image_reader_fn

from .utils import make_model

def test_create_simple():
    network = {
        'layers': [
            {
                'type': 'dense',
                'number_of_nodes': 8,
                'seed': 42,
                'activation_function': 'leaky_relu',
                'weights': 'glorot_uniform',
                'offset': 'zeros'
            },
            {
                'type': 'dense',
                'number_of_nodes': 16,
                'seed': 43,
                'activation_function': 'softmax',
                'weights': 'glorot_normal',
                'offset': 'ones'
            }
        ]
    }

    lseq = layer_sequence(network)
    model = make_model(lseq, 4)

    assert len(lseq) == 2

    for sizes, layer in zip([(4, 8), (8, 16)], lseq):
        fan_in, fan_out = sizes
        weights = layer.get_weights()

        assert weights[0].shape == sizes
        assert weights[1].shape == (fan_out,), weights[1].shape

        assert len(set(weights[0].flatten())) == fan_in * fan_out
        assert np.all(weights[0] > -2.0)
        assert np.all(weights[0] < 2.0)

        if fan_out == 16:
            assert np.all(weights[1] == 1.0), weights[1]
        else:
            assert np.all(weights[1] == 0.0), weights[1]

def test_create_residual():
    network = {
        'layers': [
            {
                'type': 'dense',
                'number_of_nodes': 6,
                'seed': 42,
                'activation_function': 'leaky_relu',
                'weights': 'glorot_uniform',
                'offset': 'zeros'
            },
            {
                'type': 'dense_residual_block',
                'activation_function': 'softplus',
                'identity_path': [],
                'dense_path': [
                    {
                        'type': 'dense',
                        'number_of_nodes': 6,
                        'seed': 42,
                        'activation_function': None,
                        'weights': 'glorot_uniform',
                        'offset': 'zeros'
                    },
                    {
                        'type': 'batch_normalization',
                        'beta': 'ones',
                        'gamma': 'zeros',
                        'mean': 'ones',
                        'variance': 'zeros'
                    },
                    {
                        'type': 'activation',
                        'activation_function': 'relu',
                    },
                    {
                        'type': 'dense',
                        'number_of_nodes': 6,
                        'seed': 42,
                        'activation_function': None,
                        'weights': 'glorot_uniform',
                        'offset': 'zeros'
                    },
                    {
                        'type': 'batch_normalization',
                        'beta': 'ones',
                        'gamma': 'ones',
                        'mean': 'zeros',
                        'variance': 'zeros'
                    },
                    {
                        'type': 'activation',
                        'activation_function': 'softmax',
                    }
                ]
            },
            {
                'type': 'dense',
                'number_of_nodes': 4,
                'seed': 42,
                'activation_function': 'softmax',
                'weights': 'glorot_uniform',
                'offset': 'zeros'
            }
        ]
    }

    lseq = layer_sequence(network)
    model = make_model(lseq, 4)

    assert len(lseq) == 3

    assert [kl.Dense, BlockLayer, kl.Dense] == [type(layer) for layer in lseq]

    dense_path_types =[kl.Dense, kl.BatchNormalization, kl.Activation] * 2
    block_paths = lseq[1]._paths[0]

    assert dense_path_types == [type(layer) for layer in block_paths]

    for layer in [block_paths[0], block_paths[3]]:
        assert type(layer) == kl.Dense

        weights, offset = layer.get_weights()
        assert weights.shape == (6, 6)
        assert offset.shape == (6,)

def show_outputs(images, reader, model):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10,10))

    for n, image in enumerate(images[:16]):
        ax = plt.subplot(4, 4, n + 1)
        img = np.expand_dims(reader(image), axis=0)
        img_pred = np.minimum(255, model(img, training=True)[0].numpy())

        plt.imshow(np.minimum(255, img_pred.astype(np.uint8)))
        plt.axis('off')

    plt.show()

def test_dropblock():
    image_shape = (None, 64, 64, 3)
    network = {
        'layers': [{'type': 'dropout',
                    'dropout_type': 'block',
                    'block_size': 7,
                    'rate': 0.1}]
    }

    reader = get_image_reader_fn(image_shape, 'file', 'tests/data/images')
    lseq = layer_sequence(network)
    model = make_model(lseq, image_shape[1:])
    pizzas = ['pizza_people.jpg'] * 16

    # show_outputs(pizzas, reader, model)

    for image in pizzas:
        img = np.expand_dims(reader(image), axis=0)
        model(img, training=True)
