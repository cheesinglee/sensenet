from sensenet.constants import NUMERIC, CATEGORICAL, BOUNDING_BOX, MASKS

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

def get_anchors(network):
    base = network['metadata']['base_image_network']
    anchors = network['metadata']['anchors']

    return [[anchors[idx] for idx in mask] for mask in MASKS[base]]
