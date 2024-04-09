import torch
import numpy as np

def th_quat_from_isaac_to_real(quat):
    return torch.hstack([quat[:,1:],quat[:,0].reshape(-1,1)])

def np_quat_from_isaac_to_real(quat):
    return np.concatenate([quat[:,1:],quat[:,0]])