OPTIONAL = {
    'bounding_box_threshold': [1e-8, 1.0],
    'iou_threshold': [1e-8, 1.0],
    'load_pretrained_weights': bool,
    'input_image_format': str,
    'image_path_prefix': str
}

REQUIRED = {
}

ATTRIBUTES = {}
ATTRIBUTES.update(OPTIONAL)
ATTRIBUTES.update(REQUIRED)

class Settings(object):
    def __init__(self, amap):
        for key in ATTRIBUTES.keys():
            if key not in amap:
                self.__setattr__(key, None)
            else:
                self.__setattr__(key, amap[key])

    def __setattr__(self, name, value):
        if name not in ATTRIBUTES:
            raise AttributeError('"%s" is not a valid field' % name)

        if value is not None:
            validator = ATTRIBUTES[name]

            if type(validator) == list:
                if len(validator) == 2 and type(validator[0]) == float:
                    assert validator[0] <= value <= validator[1]
                elif validator[0] == list:
                    assert type(value) == list
                    for v in value:
                        assert type(v) == validator[1]
                elif validator[0] == dict:
                    ktype, vtype = validator[1],
                    for key in value:
                        assert type(key) == ktype, (ktype, key)
                        assert type(value[key]) == vtype, (vtype, value[key])
                else:
                    assert value in validator, (name, value)
            elif type(validator) == type:
                assert type(value) == validator, str((type(value), validator))
            else:
                raise ValueError('Validator is "%s"' % str(validator))

        super().__setattr__(name, value)

    def __getattribute__(self, name):
        value = super().__getattribute__(name)

        if value is None and name in REQUIRED:
            raise AttributeError('"%s" not in settings and is required' % name)

        return value
