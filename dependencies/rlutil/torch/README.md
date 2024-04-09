# rlutil.torch

A light wrapper around torch which allows for default device placement.

# Example
In torch, you typically have to do this to write device-agnostic code:
```python
import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

torch.zeros(5, device=device)
torch.tensor(np.array([1]), device=device)

module = MyModule()  # inherits from torch.nn.Module
module.to(device=device)
```
It's quite clunky to always pass around the device.

rlutil.torch instead keeps a global default device.
```python
import rlutil.torch as torch
torch.set_gpu(True)  # This is a special command to set the default device

torch.zeros(5)
torch.tensor(np.array([1]))

module = MyModule() # inherits from rlutil.torch.nn.Module
```

