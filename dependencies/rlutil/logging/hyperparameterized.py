import six

CLSNAME = '__clsname__'
_HYPER_ = '__hyper__'
_HYPERNAME_ = '__hyper_clsname__'


def extract_hyperparams(obj):
    if any([isinstance(obj, type_) for type_ in (int, float, str)]):
        return obj
    elif isinstance(type(obj), Hyperparameterized):
        hypers = getattr(obj, _HYPER_)
        hypers[CLSNAME] = getattr(obj, _HYPERNAME_)
        for attr in hypers:
            hypers[attr] = extract_hyperparams(hypers[attr])
        return hypers
    return type(obj).__name__

class Hyperparameterized(type):
    def __new__(self, clsname, bases, clsdict):
        old_init = clsdict.get('__init__', bases[0].__init__)
        def init_wrapper(inst, *args, **kwargs):
            hyper = getattr(inst, _HYPER_, {})
            hyper.update(kwargs)
            setattr(inst, _HYPER_, hyper)

            if getattr(inst, _HYPERNAME_, None) is None:
                setattr(inst, _HYPERNAME_, clsname)
            return old_init(inst, *args, **kwargs)
        clsdict['__init__'] = init_wrapper

        cls = super(Hyperparameterized, self).__new__(self, clsname, bases, clsdict)
        return cls


@six.add_metaclass(Hyperparameterized)
class HyperparamWrapper(object):
    def __init__(self, **hyper_kwargs):
        pass

