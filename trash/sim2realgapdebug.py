import wandb
import numpy as np

wandb.init(project="sim2realdebug")

datasim = np.load("demos/isaac-envsimsysid/demo_0_states.npy")[:, -9:]
datareal = np.load("demos/sysidreal/states_0_0.npy")

datasimact = np.load("demos/isaac-envsimsysid/demo_0_actions.npy")
datarealact = np.load("demos/sysidreal/actions_0_0.npy")

for i in range(len(datareal[0])):
    error = {"step":i}
    for j in range(9):
        error[str(j)] = abs(datareal[0][i][j]-datasim[i][j])
    assert datasimact[i] == datarealact[0][i]
    wandb.log(error)
    print(error)
