OPTIONAL = {
    'bounding_box_threshold': [1e-8, 1.0],
    'image_path_prefix': str,
    'input_image_format': str,
    'iou_threshold': [1e-8, 1.0],
    'load_pretrained_weights': bool,
    'max_objects': int,
    'regression_normalize': bool
}

REQUIRED = {}

class Settings(object):
    _required_attributes = REQUIRED
    _attribute_validators = {}
    _attribute_validators.update(OPTIONAL)
    _attribute_validators.update(REQUIRED)

    def __init__(self, amap):
        for key in self.__class__._attribute_validators.keys():
            if key not in amap:
                self.__setattr__(key, None)
            else:
                self.__setattr__(key, amap[key])

        for key in sorted(amap.keys()):
            if key not in self.__class__._attribute_validators:
                raise AttributeError('"%s" is not a valid field' % key)

    def __setattr__(self, name, value):
        if name not in self.__class__._attribute_validators:
            raise AttributeError('"%s" is not a valid field' % name)

        if value is not None:
            validator = self.__class__._attribute_validators[name]

            if type(validator) == list:
                if len(validator) == 2 and type(validator[0]) in [int, float]:
                    assert validator[0] <= value <= validator[1], (name, value)
                elif validator[0] == list:
                    assert type(value) == list, (name, value)
                    for v in value:
                        assert type(v) == validator[1]
                elif validator[0] == dict:
                    assert type(value) == dict, (name, value)
                    ktype, vtype = validator[1]
                    for key in value:
                        assert type(key) == ktype, (name, ktype, key)
                        assert type(value[key]) == vtype, (vtype, value[key])
                else:
                    assert value in validator, (name, value, validator)
            elif type(validator) == type:
                assert type(value) == validator, (name, value, validator)
            else:
                raise ValueError('Validator is "%s"' % str(validator))

        super().__setattr__(name, value)

    def __getattribute__(self, name):
        value = super().__getattribute__(name)

        if value is None and name in self.__class__._required_attributes:
            raise AttributeError('"%s" not in settings and is required' % name)

        return value

def ensure_settings(avalue):
    if type(avalue) == Settings:
        return avalue
    elif type(avalue) == dict:
        return Settings(avalue)
    elif avalue is None:
        return Settings({})
    else:
        raise ValueError('settings input type is "%s"' % type(avalue))
