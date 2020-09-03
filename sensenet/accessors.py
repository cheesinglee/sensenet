from sensenet.constants import NUMERIC, CATEGORICAL, BOUNDING_BOX

def get_image_shape(anobject):
    if type(anobject) == dict:
        if 'image_network' in anobject:
            ishape = anobject['image_network']['metadata']['input_image_shape']
        elif 'metadata' in anobject:
            ishape = anobject['metadata']['input_image_shape']
        else:
            msg = 'Cannot find image shape in dict with keys %s'
            raise ValueError(msg % str(sorted(anobject.keys())))
    elif type(anobject) in [list, tuple]:
        ishape = anobject
    else:
        try:
            ishape = anobject.input_image_shape
        except AttributeError:
            msg = 'Cannot find image shape in objective of type "%s"'
            raise ValueError(msg % type(anobject))

    return [None, ishape[1], ishape[0], ishape[2]]

def get_output_exposition(model):
    if 'output_exposition' in model:
        return model['output_exposition']
    elif 'networks' in model:
        first_network = model['networks'][0]
        return get_output_exposition(first_network)
    else:
        raise ValueError('Could not locate output_exposition')

def number_of_classes(model):
    outex = get_output_exposition(model)

    if outex['type'] == NUMERIC:
        return 1
    elif outex['type'] in [CATEGORICAL, BOUNDING_BOX]:
        return len(outex['values'])
    else:
        raise ValueError('Output exposition is type "%s"' % outex['type'])

def get_layer(model, layer_type, names):
    for layer in model.layers:
        if type(layer) == layer_type:
            if names is None or layer.name in names:
                return layer

    msg = 'Could not find layer of type %s' % str(layer_type)

    if names is not None:
        msg += 'with name in %s' % str(names)

    raise ValueError(msg)

def is_yolo_model(network):
    image_net = network.get('image_network', None)
    return image_net and 'yolo' in image_net['metadata']['base_image_network']
