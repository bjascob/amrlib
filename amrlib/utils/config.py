import json


# Simple class for loading/saving json config data where class attributes
# are the configuration options.
class Config(object):
    def __init__(self, adict=None):
        if adict:
            self.__dict__.update(adict)

    def __str__(self):
        config_dict = vars(self)
        return json.dumps(config_dict, indent=4)

    @classmethod
    def load(cls, fname):
        self = cls()
        with open(fname) as f:
            data = json.load(f)
        for k, v in data.items():
            setattr(self, k, v)
        return self

    def save(self, fname):
        config_dict = vars(self)
        with open(fname, 'w') as f:
            json.dump(config_dict, f, indent=4)
