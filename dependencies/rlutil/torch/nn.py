from torch.nn import *
import torch.nn as _nn
import six

from rlutil.torch.pytorch_util import default_device

# define a new metaclass which overrides the "__call__" function
# https://stackoverflow.com/questions/16017397/injecting-function-call-after-init-with-decorator
class _NewInitCaller(type):
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj._post_init()
        return obj

@six.add_metaclass(_NewInitCaller)
class Module(_nn.Module):
    def __init__(self):
        super(Module, self).__init__()

    def _post_init(self):
        self.to(device=default_device())

