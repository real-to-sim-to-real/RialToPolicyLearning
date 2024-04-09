# RialTo Policy Learning
This repository provides the official implementation of the RialTo system, as proposed in *Reconciling Reality through Simulation: A Real-to-Sim-to-Real approach for Robust Manipulation*
The manuscript is available on [arXiv](https://arxiv.org/abs/2403.03949). See the [project page](https://real-to-sim-to-real.github.io/RialTo/)

If you use this codebase, please cite

```
@article{torne2024reconciling,
  title={Reconciling Reality through Simulation: A Real-to-Sim-to-Real Approach for Robust Manipulation},
  author={Torne, Marcel and Simeonov, Anthony and Li, Zechu and Chan, April and Chen, Tao and Gupta, Abhishek and Agrawal, Pulkit},
  journal={arXiv preprint arXiv:2403.03949},
  year={2024}
}
```
## Installation

1. Download omniverse: https://www.nvidia.com/en-us/omniverse/

2. Install isaac-sim 2022.2.1 (from omniverse launcher)

Launch isaac-sim to complete the installation

3. Install orbit https://docs.omniverse.nvidia.com/isaacsim/latest/ext_omni_isaac_orbit.html

```
git clone git@github.com:NVIDIA-Omniverse/orbit.git
git checkout f2d97bdcddb3005d17d0ebd1546c7064bc7ae8bc

```

```
export ISAACSIM_PATH=<path to isaacsim>
```

```
# enter the cloned repository
cd orbit
# create a symbolic link
ln -s ${ISAACSIM_PATH} _isaac_sim
```

Create conda environment from orbit
```
./orbit.sh --conda isaac-sim
conda activate isaac-sim
orbit -i
orbit -e
```

Make sure orbit was correctly set up:
```
python -c "import omni.isaac.orbit; print('Orbit configuration is now complete.')"
```

4. Clone repo
```
git clone git@github.com:IAILeveragingRealToSim/PolicyLearning.git
git checkout dev-massive-huge
cd PolicyLearning
pip install -r requirements.txt
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cu117.html --no-index
pip install torch-cluster -f https://data.pyg.org/whl/torch-1.13.0+cu117.html --no-index
```

```
conda install conda-build
conda develop dependencies
```

You should be ready to go :) Test it our running the teleoperation script below

## Pipeline:

![alt text](https://github.com/real-to-sim-to-real/RialToPolicyLearning/blob/main/materials/teaserv14.pdf)

- Create your environment using the [RialTo GUI](https://github.com/real-to-sim-to-real/RialToGUI)


## Running the code
![alt text](https://github.com/real-to-sim-to-real/RialToPolicyLearning/blob/main/materials/taskrandomizationv2.pdf)

### Collect teleoperation data in sim

- booknshelf
```
python collect_demos_teleop_franka.py --env_name=isaac-env --img_width=1024 --img_height=1024 --demo_folder=booknshelvenew --max_path_length=150 --extra_params=booknshelve,booknshelve,booknshelve_debug_mid_randomness --usd_path=/home/marcel/USDAssets/scenes --offset=1 
```

- mugandshelf
```
python collect_demos_teleop_franka.py --env_name=isaac-env --img_width=1024 --img_height=1024 --demo_folder=mugandshelfnew2 --max_path_length=150 --extra_params=mugandshelf,mugandshelf_mid_rot_randomness --usd_path=/home/marcel/USDAssets/scenes
```
- cabinet 
```
python collect_demos_teleop_franka.py --env_name=isaac-env --img_width=1024 --img_height=1024 --demo_folder=cabinet --max_path_length=150 --extra_params=cabinet,cabinet_mid_randomness --usd_path=/home/marcel/USDAssets/scenes
```
- drawer
```
python collect_demos_teleop_franka.py --env_name=isaac-env --img_width=1024 --img_height=1024 --demo_folder=drawertras --max_path_length=150 --extra_params=wooden_drawer_bigger,drawer_debug_high_randomness --usd_path=/home/marcel/USDAssets/scenes
```
- dishinrack
```
python collect_demos_teleop_franka.py --env_name=isaac-env --img_width=1024 --img_height=1024 --demo_folder=dishinrackv3 --max_path_length=200 --extra_params=dishnrackv2,dishnrack_high_randomness,no_action_rand
```
- kitchentoaster
```
python collect_demos_teleop_franka.py --env_name=isaac-env --img_width=1024 --img_height=1024 --demo_folder=kitchentoaster --max_path_length=150 --extra_params=kitchentoaster,kitchentoaster_randomness,no_action_rand
```

### RL-finetuning

- cabinet
```
python launch_ppo.py --num_envs=4096 --n_steps=90 --max_path_length=90 --usd_path=/home/marcel/USDAssets/scenes --bc_loss --bc_coef=0.1 --filename=isaac-envcabinet --extra_params=cabinet,cabinet_mid_rot_randomness,no_action_rand --datafolder=demos --num_demos=12 --ppo_batch_size=31257 --run_path=locobot-learn/cabinet.usdppo-finetune/rw4wp46f --model_name=model_policy_567 --from_ppo
```
- mugandshelf
```
python launch_ppo.py --num_envs=4096 --n_steps=150 --max_path_length=150 --bc_loss --bc_coef=0.1 --filename=isaac-envmugandshelf --extra_params=mugandshelf,mugandshelf_mid_rot_randomness,no_action_rand --datafolder=demos --num_demos=6 --ppo_batch_size=31257 --model_name=model_policy_279 --run_path=locobot-learn/mugandshelf.usdppo-finetune/ef5nbwyb --from_ppo
```
- booknshelve
```
python launch_ppo.py --num_envs=4096 --n_steps=130 --max_path_length=130 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --bc_loss --bc_coef=0.1 --filename=booknshelvesreal2sim --datafolder=/data/pulkitag/data/marcel/data --num_demos=14 --ppo_batch_size=31257 --num_envs=2048
```
- kitchen toaster
```
python launch_ppo.py --num_envs=4096 --n_steps=130 --max_path_length=130 --extra_params=kitchentoaster,kitchentoaster_randomness,no_action_rand --bc_loss --bc_coef=0.1 --filename=isaac-envkitchentoaster --datafolder=demos --num_demos=12 --ppo_batch_size=31257 --num_envs=2048 --usd_path=/home/marcel/USDAssets/scenes
```
- dishsinklab
 ```
python launch_ppo.py --num_envs=4096 --n_steps=110 --max_path_length=110 --extra_params=dishsinklab,dishsinklab_low_randomness,no_action_rand --bc_loss --bc_coef=0.1 --filename=isaac-envdishsinklablow --datafolder=demos --num_demos=15 --ppo_batch_size=31257 --num_envs=2048 --usd_path=/home/marcel/USDAssets/scenes
```
### Visualize PPO
```
python distillation.py --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --max_path_length=130 --filename=trash --run_path=locobot-learn/booknshelve.usdppo-finetune/ruk0ihdc --model_name=policy_finetune_step_210 --datafolder=/data/pulkitag/data/marcel/data/ --visualize_traj --max_demos=10 --sensors=rgb,pointcloud
```
### Distillation from synthetic pointclouds
- booknshelve
```
python distillation.py --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --max_path_length=130 --filename=booknshelvenewsynthetic --run_path=locobot-learn/booknshelve.usdppo-finetune/ruk0ihdc --model_name=policy_finetune_step_210 --datafolder=/data/pulkitag/data/marcel/data/ --use_synthetic_pcd --max_demos=15000 --eval_freq=1 --use_state --visualize_traj
```
- mugandshelf
```
python distillation.py --max_path_length=150 --extra_params=mugandshelf,mugandshelf_mid_rot_randomness,no_action_rand --filename=mugandshelfsynthetic --run_path=locobot-learn/mugandshelf.usdppo-finetune/ke2wi6xb --model_name=policy_finetune_step_369 --datafolder=/data/pulkitag/data/marcel/data/ --use_synthetic_pcd --max_demos=15000 --eval_freq=1 --use_state --visualize_traj
```
- cabinet
```
python distillation.py --max_path_length=90 --extra_params=cabinet,cabinet_mid_rot_randomness,no_action_rand --filename=cabinetsynthetic --run_path=locobot-learn/cabinet.usdppo-finetune/uqn3jbmg --model_name=policy_finetune_step_393 --datafolder=/data/pulkitag/data/marcel/data/ --use_synthetic_pcd --max_demos=15000 --eval_freq=1 --use_state --visualize_traj
```
- mugupright
```
python distillation.py --max_path_length=100 --extra_params=mugupright,mugupright_mid_randomness,no_action_rand --filename=muguprightsynthetic --run_path=locobot-learn/mugupright.usdppo-finetune/5vocshfn --model_name=model_policy_830 --from_ppo --datafolder=/data/pulkitag/data/marcel/data/ --use_synthetic_pcd --max_demos=15000 --eval_freq=1 --use_state --visualize_traj --usd_path=/home/marcel/USDAssets/scenes 
```
### Distillation from sim pcd
- cabinet

```
python distillation.py --max_path_length=90 --extra_params=cabinet,cabinet_mid_rot_randomness,no_action_rand --filename=cabinetsimpcd --run_path=locobot-learn/cabinet.usdppo-finetune/uqn3jbmg --model_name=policy_finetune_step_393 --datafolder=/data/pulkitag/data/marcel/data/ --max_demos=2500 --eval_freq=1 --use_state --visualize_traj --policy_batch_size=32 --num_envs=9 --run_path_student=locobot-learn/distillation_cabinet/47r0waz0 --model_name_student=policy_distill_step_71
```
- mugupright
```
python distillation.py --max_path_length=100 --extra_params=mugupright,mugupright_mid_randomness,no_action_rand --filename=muguprightsimpcd --run_path=locobot-learn/mugupright.usdppo-finetune/5vocshfn --model_name=model_policy_830 --from_ppo --datafolder=/home/marcel/data/ --max_demos=2500 --eval_freq=1 --use_state --visualize_traj --num_envs=12 --usd_path=/home/marcel/USDAssets/scenes --run_path_student=locobot-learn/distillation_mugupright/tjlds4ds --model_name_student=policy_distill_step_407
```
- mugandshelf
```
python distillation.py --max_path_length=150 --extra_params=mugandshelf,mugandshelf_mid_rot_randomness,no_action_rand --filename=mugandshelfsimpcd --run_path=locobot-learn/mugandshelf.usdppo-finetune/ke2wi6xb --model_name=policy_finetune_step_369 --datafolder=/data/pulkitag/data/marcel/data/ --max_demos=2500 --eval_freq=1 --use_state --num_envs=12 --visualize_traj --model_name_student=policy_distill_step_381 --run_path_student=locobot-learn/distillation_mugandshelf/valc8x68
```
- booknshelve
```
python distillation.py --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --max_path_length=130 --filename=booknshelvenewsimpcd --run_path=locobot-learn/booknshelve.usdppo-finetune/ruk0ihdc --model_name=policy_finetune_step_210 --datafolder=/data/pulkitag/data/marcel/data/ --num_envs=9 --max_demos=2500 --eval_freq=1 --use_state --visualize_traj
```
### Collecting distractor data
TODO
```
python distillation.py --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --max_path_length=130 --filename=booknshelvenewfromcam --run_path=locobot-learn/booknshelve.usdppo-finetune/ruk0ihdc --model_name=policy_finetune_step_210 --datafolder=/data/pulkitag/data/marcel/data/ --visualize_traj --run_path_student= --model_name_student= --max_demos=2000
```
### Running Dagger



## Running in the real world
### Install environment
We use [Polymetis](https://facebookresearch.github.io/fairo/polymetis/). You would need to install polymetis on the robot side. We give the instructions on how to install and create the environments on the GPU side.

```
conda create -n franka-env-new-cuda python=3.8.15
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c pytorch -c fair-robotics -c aihabitat -c conda-forge polymetis
```

- Clone and install [airobot](https://github.com/Improbable-AI/airobot)
- Clone and install [improbable_rdt]()



### Evaluation
```
conda activate franka-env-new-cuda
export RDT_SOURCE_DIR=~/improbable_rdt/src/rdt
export PB_PLANNING_SOURCE_DIR=~/improbable_rdt/pybullet-planning/
```

- dishinrack
```
python evaluate_policy_real.py --run_path=locobot-learn/distillation_dishnrack/zmglvp19 --model_name=policy_distill_step_67 --max_path_length=130 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --background_loop --total_loop_time=0.3 --interp_steps=30 --start_interp_offset=3
```

### Collect teleop data in the real world
```
conda activate franka-env-new-cuda
export RDT_SOURCE_DIR=~/improbable_rdt/src/rdt
export PB_PLANNING_SOURCE_DIR=~/improbable_rdt/pybullet-planning/
```
```
python teleop_franka.py --demo_folder=mugandshelfreal --offset=0 --hz=2 --max_path_length=100 --extra_params=booknshelve,booknshelve_debug_low_randomness,booknshelve2
```
```
python teleop_franka.py --demo_folder=cupntrashreal --offset=0 --hz=2 --max_path_length=100 --extra_params=cupntrash,kitchentoaster_randomness,no_action_rand --num_demos=15 --cam_index=1
```
```
python teleop_franka.py --demo_folder=cupntrashreal --offset=0 --hz=2 --max_path_length=100 --extra_params=cupntrash,kitchentoaster_randomness,no_action_rand --num_demos=15 --cam_index=1
```

