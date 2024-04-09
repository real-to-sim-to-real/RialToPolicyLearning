# Policy Learning

## Installation (TODO)

Download omniverse: https://www.nvidia.com/en-us/omniverse/

Install isaac-sim 2022.2.1 (from omniverse launcher)

Launch isaac-sim to complete the installation

Install orbit https://docs.omniverse.nvidia.com/isaacsim/latest/ext_omni_isaac_orbit.html

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

# Pipeline:
- obtain mesh of the desired scene using Polycam photo mode 

- Export GLTB medium quality

- Import into our API, modify the mesh add joints, sites, fixed joint

- Save as .usdz or .usd file

- Create environment:
    - Extract .usdz into folder (double click)
    - open model.usda inside folder with isaac-sim
    - Verify physics are correct by pressing play on isaac-sim
    - Save file into USDAssets/scenes/ as a usd file
    - create entry inside config file for the environment (look at the other examples, e.g. cabinet)
    - create entry for the randomness level (e.g. cabinet_mid_randomness)


## Camera calibration
conda activate franka-env-new-cuda
export RDT_SOURCE_DIR=~/improbable_rdt/src/rdt
export PB_PLANNING_SOURCE_DIR=~/improbable_rdt/pybullet-planning/

cd improbable_rdt/src/rdt

Make sure the camera is pointing on the right direction
python image/multi_realsense_visualizer.py --find_devices

Calibrate camera
tweak the values of the range of motion of the arm in 
Start meshcat-server
python polymetis_robot_utils/panda_rs_cal_client.py --cam_index=2 --robot --reset_pose

copy files to the correct location
cp robot/camera_calibration_files/result/panda/cam_2_calib_base_to_cam.json robot/camera_calibration_files/
check that the calibration worked
python point_cloud/check_fused_pcd.py --cam_index=1

# Final models
```
conda activate franka-env-new-cuda
export RDT_SOURCE_DIR=~/improbable_rdt/src/rdt
export PB_PLANNING_SOURCE_DIR=~/improbable_rdt/pybullet-planning/
```

- dishinrack
python evaluate_policy_real.py --run_path=locobot-learn/distillation_dishnrack/zmglvp19 --model_name=policy_distill_step_67 --max_path_length=130 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --background_loop --total_loop_time=0.3 --interp_steps=30 --start_interp_offset=3

- mugandshelf
python evaluate_policy_real.py --run_path=locobot-learn/distillation_mugandshelf/rne3qxrm --model_name=policy_distill_step_80 --max_path_length=300 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --background_loop --total_loop_time=0.3 --interp_steps=30 --start_interp_offset=3 --gripper_force=100

- booknshelf
python evaluate_policy_real.py --run_path=locobot-learn/distillation_booknshelve/chxoezd4 --model_name=policy_distill_step_24 --max_path_length=300 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --background_loop --total_loop_time=0.35 --interp_steps=40 --start_interp_offset=4 --gripper_force=100

- cabinet
python evaluate_policy_real.py --run_path=locobot-learn/distillation_cabinet/h1c8advw --model_name=policy_distill_step_17 --max_path_length=300 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --total_loop_time=0.5 --interp_steps=70 --start_interp_offset=7 --gripper_force=50 --nb_open=1 --background_loop

python evaluate_policy_real.py --run_path=locobot-learn/distillation_cabinet/h1c8advw --model_name=policy_distill_step_17 --max_path_length=300 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --total_loop_time=0.2 --interp_steps=100 --start_interp_offset=10 --gripper_force=50 --nb_open=1 --background_loop


- drawer
python evaluate_policy_real.py --run_path=locobot-learn/distillation_drawer_bigger/kjjyh92p --model_name=policy_distill_step_85 --max_path_length=300 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --background_loop --total_loop_time=0.3 --interp_steps=30 --start_interp_offset=3 --gripper_force=100 --nb_open=12

- kitchentoaster 
python evaluate_policy_real.py --run_path=locobot-learn/distillation_kitchentoaster/x6pzujl0 --model_name=policy_distill_step_15 --max_path_length=130 --cam_index=1 --extra_params=kitchentoaster,kitchentoaster_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --background_loop --total_loop_time=0.3 --interp_steps=30 --start_interp_offset=3 --gripper_force=100


- fast controller cabinet: 
python evaluate_policy_real.py --run_path=locobot-learn/distillation_cabinet/h1c8advw --model_name=policy_distill_step_17 --max_path_length=300 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --total_loop_time=0.1 --interp_steps=10 --start_interp_offset=5 --gripper_force=50 --nb_open=1 --background_loop

- fast controller drawer: 
python evaluate_policy_real.py --run_path=locobot-learn/distillation_drawer_bigger/kjjyh92p --model_name=policy_distill_step_85 --max_path_length=300 --max_path_length=300 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --total_loop_time=0.1 --interp_steps=10 --start_interp_offset=5 --gripper_force=50 --nb_open=1 --background_loop

- fast controller mug and shelf
python evaluate_policy_real.py --run_path=locobot-learn/distillation_mugandshelf/rne3qxrm --model_name=policy_distill_step_80 --max_path_length=300 --max_path_length=300 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --total_loop_time=0.1 --interp_steps=10 --start_interp_offset=5 --gripper_force=50 --nb_open=1 --background_loop

- booknshelf
python evaluate_policy_real.py --run_path=locobot-learn/distillation_booknshelve/cqcez3jj --model_name=policy_distill_step_24 --max_path_length=300 --max_path_length=300 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --total_loop_time=0.1 --interp_steps=10 --start_interp_offset=5 --gripper_force=50 --nb_open=1 --background_loop

- cupntrash
python evaluate_policy_real.py --run_path=locobot-learn/distillation_cupntrash/k7s5n3jx --model_name=policy_distill_step_144 --max_path_length=130 --cam_index=1 --extra_params=cupntrash,cupntrash_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --total_loop_time=0.1 --interp_steps=10 --start_interp_offset=3 --gripper_force=100 --nb_open=3 --background_loop --total_loop_time=0.1 --interp_steps=10 --start_interp_offset=5 --gripper_force=50 --nb_open=2 --background_loop



#### real-to-sim transfer
python distillation.py --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --max_path_length=130 --filename=boonkshelvefixed --datafolder=demos --max_demos=15 --eval_freq=1 --use_state --visualize_traj --num_trajs_per_step=5


#### Collect teleoperation data
```
eval "$(/scratch/marcel/miniconda3/bin/conda shell.bash hook)"
conda activate isaac-sim
source /scratch/marcel/isaac_sim-2022.2.1/setup_conda_env.sh
```

- bowl:

python collect_demos_teleop_franka.py --env_name=isaac-env --img_width=1024 --img_height=1024 --demo_folder=bowlnrack --max_path_length=100 --extra_params=bowlnrack,bowlnrack_high_randomness

- dish:

python collect_demos_teleop_franka.py --env_name=isaac-env --img_width=1024 --img_height=1024 --demo_folder=dishnrack --max_path_length=100 --extra_params=dishnrack,dishnrack_high_randomness

- mugnrack

python collect_demos_teleop_franka.py --env_name=isaac-env --img_width=1024 --img_height=1024 --demo_folder=mugnrack --max_path_length=100 --extra_params=mugnrack,mugnrack_low_randomness

- booknshelf

python collect_demos_teleop_franka.py --env_name=isaac-env --img_width=1024 --img_height=1024 --demo_folder=booknshelverobust --max_path_length=100 --extra_params=booknshelve,booknshelve_low_randomness

- booknshelf2

python collect_demos_teleop_franka.py --env_name=isaac-env --img_width=1024 --img_height=1024 --demo_folder=booknshelvenew --max_path_length=150 --extra_params=booknshelve,booknshelve,booknshelve_debug_mid_randomness --usd_path=/home/marcel/USDAssets/scenes --offset=1 

python launch_ppo.py --num_envs=2048 --n_steps=60 --max_path_length=60 --model_name=policy_distill_step_745 --run_path=locobot-learn/distillation_student_from_state/g03jmpb5 --extra_params=wooden_drawer_bigger,drawer_debug_more_randomnes --usd_path=/home/marcel/USDAssets/scenes

- mugandshelf

python collect_demos_teleop_franka.py --env_name=isaac-env --img_width=1024 --img_height=1024 --demo_folder=mugandshelfnew2 --max_path_length=150 --extra_params=mugandshelf,mugandshelf_mid_rot_randomness --usd_path=/home/marcel/USDAssets/scenes

- cabinet 

python collect_demos_teleop_franka.py --env_name=isaac-env --img_width=1024 --img_height=1024 --demo_folder=cabinet --max_path_length=150 --extra_params=cabinet,cabinet_mid_randomness --usd_path=/home/marcel/USDAssets/scenes

- mugupright

python collect_demos_teleop_franka.py --env_name=isaac-env --img_width=1024 --img_height=1024 --demo_folder=mugupright --max_path_length=150 --extra_params=mugupright,mugupright_mid_randomness --usd_path=/home/marcel/USDAssets/scenes

- mugupright

python collect_demos_teleop_franka.py --env_name=isaac-env --img_width=1024 --img_height=1024 --demo_folder=drawertras --max_path_length=150 --extra_params=wooden_drawer_bigger,drawer_debug_high_randomness --usd_path=/home/marcel/USDAssets/scenes

- drawer

python collect_demos_teleop_franka.py --env_name=isaac-env --img_width=1024 --img_height=1024 --demo_folder=drawertras --max_path_length=150 --extra_params=wooden_drawer_bigger,drawer_debug_high_randomness --usd_path=/home/marcel/USDAssets/scenes

- drawer real2sim2real

python evaluate_policy_real.py --run_path=locobot-learn/distillation_drawer_bigger/kjjyh92p --model_name=policy_distill_step_85 --max_path_length=300 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --background_loop --total_loop_time=0.3 --interp_steps=30 --start_interp_offset=3 --gripper_force=100 --nb_open=12

- dishinrack

python collect_demos_teleop_franka.py --env_name=isaac-env --img_width=1024 --img_height=1024 --demo_folder=dishinrackv3 --max_path_length=200 --extra_params=dishnrackv2,dishnrack_high_randomness,no_action_rand

- kitchentoaster

python collect_demos_teleop_franka.py --env_name=isaac-env --img_width=1024 --img_height=1024 --demo_folder=kitchentoaster --max_path_length=150 --extra_params=kitchentoaster,kitchentoaster_randomness,no_action_rand
## MIT visit days demo

- cupntrash

python collect_demos_teleop_franka.py --env_name=isaac-env --img_width=1024 --img_height=1024 --demo_folder=cupntrash --max_path_length=150 --extra_params=cupntrash,cupntrash_randomness,no_action_rand

- toynbowl

python collect_demos_teleop_franka.py --env_name=isaac-env --demo_folder=toynbowl --max_path_length=150 --extra_params=toynbowl,toynbowl_randomness,no_action_rand --usd_path=/home/marcel/USDAssets/scenes


- toycarnbowl

python collect_demos_teleop_franka.py --env_name=isaac-env --img_width=1024 --img_height=1024 --demo_folder=toycarnbowl_lowrand --max_path_length=150 --extra_params=toycarnbowl,toycarnbowl_low_randomness,no_action_rand

- dishsinklab
python collect_demos_teleop_franka.py --env_name=isaac-env --demo_folder=dishsinklab --max_path_length=150 --extra_params=dishsinklab,dishsinklab_randomness,no_action_rand --offset=0 --seed=0 --img_width=1024 --img_height=1024 --camera_params=teleop_dishsinklab

- dishsinklab low
python collect_demos_teleop_franka.py --env_name=isaac-env --demo_folder=dishsinklablow --max_path_length=150 --extra_params=dishsinklab,dishsinklab_low_randomness,no_action_rand --offset=0 --seed=0 --img_width=1024 --img_height=1024 --camera_params=teleop_dishsinklab

### distill from state
python distillation.py --max_path_length=150 --extra_params=toynbowl,toynbowl_randomness,no_action_rand --filename=isaac-envtoynbowl --datafolder=demos --student_from_state --max_demos=15 --eval_freq=1 --use_state

### PPO standard

- cabinet

python launch_ppo.py --num_envs=4096 --n_steps=90 --max_path_length=90 --usd_path=/home/marcel/USDAssets/scenes --bc_loss --bc_coef=0.1 --filename=isaac-envcabinet --extra_params=cabinet,cabinet_mid_rot_randomness,no_action_rand --datafolder=demos --num_demos=12 --ppo_batch_size=31257 --run_path=locobot-learn/cabinet.usdppo-finetune/rw4wp46f --model_name=model_policy_567 --from_ppo

- mugandshelf

python launch_ppo.py --num_envs=4096 --n_steps=150 --max_path_length=150 --bc_loss --bc_coef=0.1 --filename=isaac-envmugandshelf --extra_params=mugandshelf,mugandshelf_mid_rot_randomness,no_action_rand --datafolder=demos --num_demos=6 --ppo_batch_size=31257 --model_name=model_policy_279 --run_path=locobot-learn/mugandshelf.usdppo-finetune/ef5nbwyb --from_ppo

- booknshelve
python launch_ppo.py --num_envs=4096 --n_steps=130 --max_path_length=130 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --bc_loss --bc_coef=0.1 --filename=booknshelvesreal2sim --datafolder=/data/pulkitag/data/marcel/data --num_demos=14 --ppo_batch_size=31257 --num_envs=2048

- kitchen toaster

python launch_ppo.py --num_envs=4096 --n_steps=130 --max_path_length=130 --extra_params=kitchentoaster,kitchentoaster_randomness,no_action_rand --bc_loss --bc_coef=0.1 --filename=isaac-envkitchentoaster --datafolder=demos --num_demos=12 --ppo_batch_size=31257 --num_envs=2048 --usd_path=/home/marcel/USDAssets/scenes

- toynbowl

python launch_ppo.py --num_envs=4096 --n_steps=130 --max_path_length=110 --extra_params=toynbowl,toynbowl_randomness,no_action_rand --bc_loss --bc_coef=0.1 --filename=isaac-envtoynbowl --datafolder=demos --num_demos=12 --ppo_batch_size=31257 --num_envs=2048 --usd_path=/home/marcel/USDAssets/scenes

python launch_ppo.py --num_envs=4096 --n_steps=130 --max_path_length=130 --extra_params=kitchentoaster,kitchentoaster_randomness,no_action_rand --bc_loss --bc_coef=0.1 --filename=isaac-envkitchentoaster --datafolder=demos --num_demos=12 --ppo_batch_size=31257 --num_envs=2048 --usd_path=/home/marcel/USDAssets/scenes

python launch_ppo.py --num_envs=4096 --n_steps=110 --max_path_length=110 --extra_params=toynbowl,toynbowl_randomness,no_action_rand --bc_loss --bc_coef=0.1 --filename=isaac-envtoynbowl --datafolder=demos --num_demos=12 --ppo_batch_size=31257 --num_envs=2048 --run_path=locobot-learn/toynbowl.usdppo-finetune/1j00tddd --model_name=model_policy_00 --from_ppo

- toycarnbowl
python launch_ppo.py --num_envs=4096 --n_steps=110 --max_path_length=110 --extra_params=toycarnbowl,toycarnbowl_randomness,no_action_rand --bc_loss --bc_coef=0.1 --filename=isaac-envtoycarnbowl --datafolder=demos --num_demos=15 --ppo_batch_size=31257 --num_envs=2048

- toycarnbowl low
python launch_ppo.py --num_envs=4096 --n_steps=110 --max_path_length=110 --extra_params=toycarnbowl,toycarnbowl_low_randomness,no_action_rand --bc_loss --bc_coef=0.1 --filename=isaac-envtoycarnbowl_lowrand --datafolder=demos --num_demos=15 --ppo_batch_size=31257 --num_envs=2048


- dishsinklab
python launch_ppo.py --num_envs=4096 --n_steps=110 --max_path_length=110 --extra_params=dishsinklab,dishsinklab_randomness,no_action_rand --bc_loss --bc_coef=0.1 --filename=isaac-envdishsinklab --datafolder=demos --num_demos=8 --ppo_batch_size=31257 --num_envs=2048 --usd_path=/home/marcel/USDAssets/scenes

- dishsinklablow
python launch_ppo.py --num_envs=4096 --n_steps=110 --max_path_length=110 --extra_params=dishsinklab,dishsinklab_low_randomness,no_action_rand --bc_loss --bc_coef=0.1 --filename=isaac-envdishsinklablow --datafolder=demos --num_demos=15 --ppo_batch_size=31257 --num_envs=2048 --usd_path=/home/marcel/USDAssets/scenes
### PPO from vision
python launch_ppo.py --num_envs=256 --n_steps=110 --max_path_length=110 --extra_params=booknshelve,booknshelve_low_randomness,no_action_rand --bc_loss --bc_coef=0.1 --filename=booknshelveteleopbig --datafolder=demos --from_vision

### Visualize PPO
python distillation.py --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --max_path_length=130 --filename=trash --run_path=locobot-learn/booknshelve.usdppo-finetune/ruk0ihdc --model_name=policy_finetune_step_210 --datafolder=/data/pulkitag/data/marcel/data/ --visualize_traj --max_demos=10 --sensors=rgb,pointcloud

### Distillation from synthetic
- booknshelve
python distillation.py --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --max_path_length=130 --filename=booknshelvenewsynthetic --run_path=locobot-learn/booknshelve.usdppo-finetune/ruk0ihdc --model_name=policy_finetune_step_210 --datafolder=/data/pulkitag/data/marcel/data/ --use_synthetic_pcd --max_demos=15000 --eval_freq=1 --use_state --visualize_traj

- mugandshelf
python distillation.py --max_path_length=150 --extra_params=mugandshelf,mugandshelf_mid_rot_randomness,no_action_rand --filename=mugandshelfsynthetic --run_path=locobot-learn/mugandshelf.usdppo-finetune/ke2wi6xb --model_name=policy_finetune_step_369 --datafolder=/data/pulkitag/data/marcel/data/ --use_synthetic_pcd --max_demos=15000 --eval_freq=1 --use_state --visualize_traj

- cabinet
python distillation.py --max_path_length=90 --extra_params=cabinet,cabinet_mid_rot_randomness,no_action_rand --filename=cabinetsynthetic --run_path=locobot-learn/cabinet.usdppo-finetune/uqn3jbmg --model_name=policy_finetune_step_393 --datafolder=/data/pulkitag/data/marcel/data/ --use_synthetic_pcd --max_demos=15000 --eval_freq=1 --use_state --visualize_traj

- mugupright
python distillation.py --max_path_length=100 --extra_params=mugupright,mugupright_mid_randomness,no_action_rand --filename=muguprightsynthetic --run_path=locobot-learn/mugupright.usdppo-finetune/5vocshfn --model_name=model_policy_830 --from_ppo --datafolder=/data/pulkitag/data/marcel/data/ --use_synthetic_pcd --max_demos=15000 --eval_freq=1 --use_state --visualize_traj --usd_path=/home/marcel/USDAssets/scenes 

### Dagger from synthetic
- booknshelve
python distillation.py --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --max_path_length=130 --filename=booknshelvenewsynthetic --run_path=locobot-learn/booknshelve.usdppo-finetune/ruk0ihdc --model_name=policy_finetune_step_210 --datafolder=/data/pulkitag/data/marcel/data/ --use_synthetic_pcd --max_demos=15000 --eval_freq=1 --use_state --visualize_traj --dagger --sampling_expert=0 --run_path_student=locobot-learn/distillation_booknshelve/1seunfnw --model_name_student=policy_distill_step_73

### Distillation from sim pcd
- cabinet
python distillation.py --max_path_length=90 --extra_params=cabinet,cabinet_mid_rot_randomness,no_action_rand --filename=cabinetsimpcd --run_path=locobot-learn/cabinet.usdppo-finetune/uqn3jbmg --model_name=policy_finetune_step_393 --datafolder=/data/pulkitag/data/marcel/data/ --max_demos=2500 --eval_freq=1 --use_state --visualize_traj --policy_batch_size=32 --num_envs=9 --run_path_student=locobot-learn/distillation_cabinet/47r0waz0 --model_name_student=policy_distill_step_71

- mugupright
python distillation.py --max_path_length=100 --extra_params=mugupright,mugupright_mid_randomness,no_action_rand --filename=muguprightsimpcd --run_path=locobot-learn/mugupright.usdppo-finetune/5vocshfn --model_name=model_policy_830 --from_ppo --datafolder=/home/marcel/data/ --max_demos=2500 --eval_freq=1 --use_state --visualize_traj --num_envs=12 --usd_path=/home/marcel/USDAssets/scenes --run_path_student=locobot-learn/distillation_mugupright/tjlds4ds --model_name_student=policy_distill_step_407

- mugandshelf
python distillation.py --max_path_length=150 --extra_params=mugandshelf,mugandshelf_mid_rot_randomness,no_action_rand --filename=mugandshelfsimpcd --run_path=locobot-learn/mugandshelf.usdppo-finetune/ke2wi6xb --model_name=policy_finetune_step_369 --datafolder=/data/pulkitag/data/marcel/data/ --max_demos=2500 --eval_freq=1 --use_state --num_envs=12 --visualize_traj --model_name_student=policy_distill_step_381 --run_path_student=locobot-learn/distillation_mugandshelf/valc8x68

- booknshelve

python distillation.py --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --max_path_length=130 --filename=booknshelvenewsimpcd --run_path=locobot-learn/booknshelve.usdppo-finetune/ruk0ihdc --model_name=policy_finetune_step_210 --datafolder=/data/pulkitag/data/marcel/data/ --num_envs=9 --max_demos=2500 --eval_freq=1 --use_state --visualize_traj

TODO

## RL scratch
python distillation.py --max_path_length=150 --extra_params=dishnrackv2,dishnrack_high_randomness,no_action_rand,render_rgb --filename=dishinrackscratch --run_path=locobot-learn/dishinrackv4.usdppo-finetune/udnu9lcu --model_name=model_policy_504 --from_ppo --datafolder=/home/marcel/data --eval_freq=1 --use_state --visualize_traj --policy_batch_size=8 --policy_train_steps=10000 --eval_freq=3 --max_demos=500 --num_envs=12 --max_demos=2000 

### Distillation robustness to distractors
TODO
python distillation.py --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --max_path_length=130 --filename=booknshelvenewfromcam --run_path=locobot-learn/booknshelve.usdppo-finetune/ruk0ihdc --model_name=policy_finetune_step_210 --datafolder=/data/pulkitag/data/marcel/data/ --visualize_traj --run_path_student= --model_name_student= --max_demos=2000

### Evaluation 
- mugandshelf
python evaluate_policy_sim.py --max_path_length=150 --extra_params=mugandshelf,mugandshelf_mid_rot_randomness,no_action_rand --use_state --run_path=locobot-learn/distillation_mugandshelf/xgwom304 --start_model=175 --step_model=1

- cabinet
python evaluate_policy_sim.py --max_path_length=90 --extra_params=cabinet,cabinet_mid_rot_randomness,no_action_rand --run_path=locobot-learn/distillation_cabinet/wi6k26in --use_state --start_model=135 --step_model=1 --num_envs=9

- mugupright
python evaluate_policy_sim.py --max_path_length=100 --extra_params=mugupright,mugupright_mid_randomness,no_action_rand --run_path=locobot-learn/distillation_mugupright/ylnq9z5q --use_state --start_model=291 --step_model=1

- booknshelve
python evaluate_policy_sim.py --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --max_path_length=130  --run_path=locobot-learn/distillation_booknshelve/6ow1dyjb --use_state --start_model=222 --step_model=1 --num_envs=9

### Evaluation in the real world

- booknshelve

python evaluate_policy_real.py --run_path=locobot-learn/distillation_booknshelve/kfih6kka --model_name=policy_distill_step_44 --max_path_length=130 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --reset_if_open
 
 (with dagger)
python evaluate_policy_real.py --run_path=locobot-learn/distillation_booknshelve/ealtoo77 --model_name=policy_distill_step_23 --max_path_length=130 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --reset_if_open


- cabinet

python evaluate_policy_real.py --run_path=locobot-learn/distillation_cabinet/7u34vm05 --model_name=policy_distill_step_80 --max_path_length=130 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 

- mugandshelf

python evaluate_policy_real.py --run_path=locobot-learn/distillation_mugandshelf/tm94az56 --model_name=policy_distill_step_180 --max_path_length=130 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --reset_if_open

python evaluate_policy_real.py --run_path=locobot-learn/distillation_mugandshelf/kmiko08w --model_name=policy_distill_step_129 --max_path_length=130 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --reset_if_open

- mugupright

python evaluate_policy_real.py --run_path=locobot-learn/distillation_mugupright/o99haya5 --model_name=policy_distill_step_10 --max_path_length=130 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2

- drawer

python evaluate_policy_real.py --run_path=locobot-learn/distillation_drawer_bigger/kjjyh92p --model_name=policy_distill_step_31 --max_path_length=130 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=50

- dishnrack (no co training)

python evaluate_policy_real.py --run_path=locobot-learn/distillation_dishnrack/n1l6oorh --model_name=policy_distill_step_7 --max_path_length=130 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=50

python evaluate_policy_real.py --run_path=locobot-learn/distillation_dishnrack/chxoezd4 --model_name=policy_distill_step_30 --max_path_length=130 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100


- dishinrack co-training
, 53

python evaluate_policy_real.py --run_path=locobot-learn/distillation_dishnrack/zmglvp19 --model_name=policy_distill_step_67 --max_path_length=130 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=50

,24
- drawer bc
python evaluate_policy_real.py --run_path=locobot-learn/distillation_drawer_bigger/c1dzacek --model_name=policy_distill_step_24 --max_path_length=130 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=50
- mugandshelf bc
python evaluate_policy_real.py --run_path=locobot-learn/distillation_mugandshelf/srv033kv --model_name=policy_distill_step_16 --max_path_length=130 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=50

- kitchen bc
python evaluate_policy_real.py --run_path=locobot-learn/distillation_kitchentoaster/3x267la7 --model_name=policy_distill_step_15 --max_path_length=130 --cam_index=1 --extra_params=kitchentoaster,kitchentoaster_randomness,no_action_rand --use_state --hz=2 --gripper_force=100


- cabinet bc
python evaluate_policy_real.py --run_path=locobot-learn/distillation_cabinet/pjfkwqsf --model_name=policy_distill_step_58 --max_path_length=130 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100

# final models in sim
- dishinrack
python evaluate_policy_sim.py --run_path=locobot-learn/distillation_dishnrack/zmglvp19 --start=67 --max_path_length=130 --extra_params=dishinrackv2,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --background_loop --total_loop_time=0.3 --interp_steps=30 --start_interp_offset=3

- mugandshelf
python evaluate_policy_real.py --run_path=locobot-learn/distillation_mugandshelf/rne3qxrm --model_name=policy_distill_step_80 --max_path_length=300 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --background_loop --total_loop_time=0.3 --interp_steps=30 --start_interp_offset=3 --gripper_force=100

- booknshelf
python evaluate_policy_real.py --run_path=locobot-learn/distillation_booknshelve/chxoezd4 --model_name=policy_distill_step_24 --max_path_length=300 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --background_loop --total_loop_time=0.35 --interp_steps=40 --start_interp_offset=4 --gripper_force=100

- cabinet
python evaluate_policy_real.py --run_path=locobot-learn/distillation_cabinet/h1c8advw --model_name=policy_distill_step_17 --max_path_length=300 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --total_loop_time=0.5 --interp_steps=70 --start_interp_offset=7 --gripper_force=50 --nb_open=1 --background_loop

- drawer
python evaluate_policy_real.py --run_path=locobot-learn/distillation_drawer_bigger/kjjyh92p --model_name=policy_distill_step_85 --max_path_length=300 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --background_loop --total_loop_time=0.3 --interp_steps=30 --start_interp_offset=3 --gripper_force=100 --nb_open=12

- kitchentoaster 
python evaluate_policy_real.py --run_path=locobot-learn/distillation_kitchentoaster/x6pzujl0 --model_name=policy_distill_step_15 --max_path_length=130 --cam_index=1 --extra_params=kitchentoaster,kitchentoaster_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --background_loop --total_loop_time=0.3 --interp_steps=30 --start_interp_offset=3 --gripper_force=100

# Final models
```
conda activate franka-env-new-cuda
export RDT_SOURCE_DIR=~/improbable_rdt/src/rdt
export PB_PLANNING_SOURCE_DIR=~/improbable_rdt/pybullet-planning/
```
- dishsink lab

python evaluate_policy_real.py --run_path=locobot-learn/distillation_dishsinklab/j29dk5i9 --model_name=policy_distill_step_26 --max_path_length=130 --cam_index=1 --extra_params=dishsinklab,dishsinklab_mid_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --total_loop_time=0.3 --interp_steps=30 --start_interp_offset=3 --nb_open=1 --reset_if_open --background_loop
dishsinklab imitation learning
python evaluate_policy_real.py --run_path=locobot-learn/distillation_dishsinklab/3r93t0at --model_name=policy_distill_step_10 --max_path_length=130 --cam_index=1 --extra_params=dishsinklab,dishsinklab_mid_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --total_loop_time=0.3 --interp_steps=30 --start_interp_offset=3 --nb_open=1 --reset_if_open 

- dishinrack
python evaluate_policy_real.py --run_path=locobot-learn/distillation_dishnrack/zmglvp19 --model_name=policy_distill_step_67 --max_path_length=130 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --background_loop --total_loop_time=0.3 --interp_steps=30 --start_interp_offset=3

- dishinrack real2sim2real
python evaluate_policy_real.py --run_path=locobot-learn/distillation_dishnrack/iu8i5i7h --model_name=policy_distill_step_19 --max_path_length=130 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --total_loop_time=0.3 --interp_steps=30 --start_interp_offset=3

python evaluate_policy_real.py --run_path=locobot-learn/distillation_dishnrack/ky34a1uh --model_name=policy_distill_step_38 --max_path_length=130 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --total_loop_time=0.3 --interp_steps=30 --start_interp_offset=3 --nb_open=3



- mugandshelf
python evaluate_policy_real.py --run_path=locobot-learn/distillation_mugandshelf/rne3qxrm --model_name=policy_distill_step_80 --max_path_length=300 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --background_loop --total_loop_time=0.3 --interp_steps=30 --start_interp_offset=3 --gripper_force=100

- mugandshelf real2sim2real

python evaluate_policy_real.py --run_path=locobot-learn/distillation_mugandshelf/ysknsa8g --model_name=policy_distill_step_38 --max_path_length=300 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --total_loop_time=0.3 --interp_steps=30 --start_interp_offset=3 --gripper_force=100

- mugandshelf real2sim2real v2

python evaluate_policy_real.py --run_path=locobot-learn/distillation_mugandshelf/knl3sav3 --model_name=policy_distill_step_40 --max_path_length=300 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --total_loop_time=0.3 --interp_steps=30 --start_interp_offset=3 --gripper_force=100



- booknshelf
python evaluate_policy_real.py --run_path=locobot-learn/distillation_booknshelve/chxoezd4 --model_name=policy_distill_step_24 --max_path_length=300 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --background_loop --total_loop_time=0.35 --interp_steps=40 --start_interp_offset=4 --gripper_force=100

- booknshelf real2sim

python evaluate_policy_real.py --run_path=locobot-learn/distillation_booknshelve/gixtelga --model_name=policy_distill_step_4 --max_path_length=300 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --background_loop --total_loop_time=0.35 --interp_steps=40 --start_interp_offset=4 --gripper_force=100 --background_loop

- booknshelf real2sim v2

python evaluate_policy_real.py --run_path=locobot-learn/distillation_booknshelve/y3nzkk9b --model_name=policy_distill_step_59 --max_path_length=300 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --total_loop_time=0.35 --interp_steps=40 --start_interp_offset=4 --gripper_force=100 --nb_open=3

- booknshelve imitation learning ++

python evaluate_policy_real.py --run_path=locobot-learn/distillation_booknshelve/bq3do1s7 --model_name=policy_distill_step_56 --max_path_length=300 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand,imitation_learning_plusminus --use_state --hz=2 --gripper_force=100 --total_loop_time=0.35 --interp_steps=40 --start_interp_offset=4 --gripper_force=100

- cabinet
python evaluate_policy_real.py --run_path=locobot-learn/distillation_cabinet/h1c8advw --model_name=policy_distill_step_17 --max_path_length=300 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --total_loop_time=0.5 --interp_steps=70 --start_interp_offset=7 --gripper_force=50 --nb_open=1 --background_loop

python evaluate_policy_real.py --run_path=locobot-learn/distillation_cabinet/h1c8advw --model_name=policy_distill_step_17 --max_path_length=300 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --total_loop_time=0.2 --interp_steps=100 --start_interp_offset=10 --gripper_force=50 --nb_open=1 --background_loop


- drawer
python evaluate_policy_real.py --run_path=locobot-learn/distillation_drawer_bigger/kjjyh92p --model_name=policy_distill_step_85 --max_path_length=300 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --background_loop --total_loop_time=0.3 --interp_steps=30 --start_interp_offset=3 --gripper_force=100 --nb_open=12

- drawer imitation learning ++
python evaluate_policy_real.py --run_path=locobot-learn/distillation_drawer_bigger/2mt5p1wf --model_name=policy_distill_step_74 --max_path_length=300 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --total_loop_time=0.3 --interp_steps=30 --start_interp_offset=3 --gripper_force=100 --nb_open=3

- drawer real2sim2real

<!-- python evaluate_policy_real.py --run_path=locobot-learn/distillation_drawer_bigger/070k0p1g --model_name=policy_distill_step_86 --max_path_length=300 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --background_loop --total_loop_time=0.3 --interp_steps=30 --start_interp_offset=3 --gripper_force=100 --nb_open=12

python evaluate_policy_real.py --run_path=locobot-learn/distillation_drawer_bigger/nm2exjub --model_name=policy_distill_step_40 --max_path_length=300 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --background_loop --total_loop_time=0.3 --interp_steps=30 --start_interp_offset=3 --gripper_force=100 --nb_open=12 -->

python evaluate_policy_real.py --run_path=locobot-learn/distillation_drawer_bigger/nm2exjub --model_name=policy_distill_step_35 --max_path_length=300 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100  --total_loop_time=0.3 --interp_steps=30 --start_interp_offset=3 --gripper_force=100 --nb_open=1 --background_loop

- kitchentoaster 
python evaluate_policy_real.py --run_path=locobot-learn/distillation_kitchentoaster/x6pzujl0 --model_name=policy_distill_step_15 --max_path_length=130 --cam_index=1 --extra_params=kitchentoaster,kitchentoaster_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --background_loop --total_loop_time=0.3 --interp_steps=30 --start_interp_offset=3 --gripper_force=100


- fast controller cabinet: 
python evaluate_policy_real.py --run_path=locobot-learn/distillation_cabinet/h1c8advw --model_name=policy_distill_step_17 --max_path_length=300 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --total_loop_time=0.1 --interp_steps=10 --start_interp_offset=5 --gripper_force=50 --nb_open=1 --background_loop

- fast controller drawer: 
python evaluate_policy_real.py --run_path=locobot-learn/distillation_drawer_bigger/kjjyh92p --model_name=policy_distill_step_85 --max_path_length=300 --max_path_length=300 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --total_loop_time=0.1 --interp_steps=10 --start_interp_offset=5 --gripper_force=50 --nb_open=1 --background_loop



- fast controller mug and shelf
python evaluate_policy_real.py --run_path=locobot-learn/distillation_mugandshelf/rne3qxrm --model_name=policy_distill_step_80 --max_path_length=300 --max_path_length=300 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --total_loop_time=0.1 --interp_steps=10 --start_interp_offset=5 --gripper_force=50 --nb_open=1 --background_loop

- booknshelf
python evaluate_policy_real.py --run_path=locobot-learn/distillation_booknshelve/cqcez3jj --model_name=policy_distill_step_24 --max_path_length=300 --max_path_length=300 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --total_loop_time=0.1 --interp_steps=10 --start_interp_offset=5 --gripper_force=50 --nb_open=1

- cupntrash


python evaluate_policy_real.py --run_path=locobot-learn/distillation_cupntrash/k7s5n3jx --model_name=policy_distill_step_350 --max_path_length=130 --cam_index=1 --extra_params=cupntrash,kitchentoaster_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --total_loop_time=0.1 --interp_steps=10 --start_interp_offset=3 --gripper_force=100 --nb_open=3 --background_loop --total_loop_time=0.2 --interp_steps=20 --start_interp_offset=5 --gripper_force=50 --nb_open=2 --background_loop


- cupntrash il

python evaluate_policy_real.py --run_path=locobot-learn/distillation_cupntrash/5xklng5w --model_name=policy_distill_step_12 --max_path_length=130 --cam_index=1 --extra_params=cupntrash,kitchentoaster_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --total_loop_time=0.1 --interp_steps=10 --start_interp_offset=3 --gripper_force=100 --nb_open=3 --total_loop_time=0.2 --interp_steps=20 --start_interp_offset=5 --gripper_force=50 --nb_open=2   --reset_if_open

### MIT visit days demo [bipasha]
```
conda activate franka-env-new-cuda
export RDT_SOURCE_DIR=~/improbable_rdt/src/rdt
export PB_PLANNING_SOURCE_DIR=~/improbable_rdt/pybullet-planning/
```

- fast controller drawer: 
python evaluate_policy_real.py --run_path=locobot-learn/distillation_drawer_bigger/kjjyh92p --model_name=policy_distill_step_85 --max_path_length=300 --max_path_length=300 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --total_loop_time=0.1 --interp_steps=10 --start_interp_offset=5 --gripper_force=50 --nb_open=10 --background_loop

- fast controller mug and shelf
python evaluate_policy_real.py --run_path=locobot-learn/distillation_mugandshelf/rne3qxrm --model_name=policy_distill_step_80 --max_path_length=400 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --total_loop_time=0.1 --interp_steps=10 --start_interp_offset=5 --gripper_force=50 --nb_open=10 --background_loop

- dishinrack
<!-- python evaluate_policy_real.py --run_path=locobot-learn/distillation_dishnrack/chxoezd4 --model_name=policy_distill_step_30 --max_path_length=130 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --background_loop --total_loop_time=0.3 --interp_steps=30 --start_interp_offset=3 -->


---------------------------------------------------------------------------------------------

#### Generate synthetic pcds
python generate_meshes.py --env_name=isaac-env --extra_params=booknshelve,booknshelve,booknshelve_debug_mid_randomness --usd_path=/home/marcel/USDAssets/scenes --bc_loss --bc_coef=0.1 --filename=isaac-envbooknshelvenew --datafolder=demos --num_demos=12 --ppo_batch_size=31257

### Collect teleop data in the real world
```
conda activate franka-env-new-cuda
export RDT_SOURCE_DIR=~/improbable_rdt/src/rdt
export PB_PLANNING_SOURCE_DIR=~/improbable_rdt/pybullet-planning/
```
python teleop_franka.py --demo_folder=mugandshelfreal --offset=0 --hz=2 --max_path_length=100 --extra_params=booknshelve,booknshelve_debug_low_randomness,booknshelve2

python teleop_franka.py --demo_folder=drawerbiggerreal --offset=0 --hz=2 --max_path_length=100 --extra_params=booknshelve,booknshelve_debug_low_randomness,booknshelve2 --num_demos=15

python teleop_franka.py --demo_folder=booknshelveextra --offset=15 --hz=2 --max_path_length=100 --extra_params=booknshelve,booknshelve_debug_low_randomness,booknshelve2 --num_demos=50

python teleop_franka.py --demo_folder=dishinrackreal --offset=0 --hz=2 --max_path_length=100 --extra_params=booknshelve,booknshelve_debug_low_randomness,booknshelve2 --num_demos=15

python teleop_franka.py --demo_folder=trash --offset=0 --hz=2 --max_path_length=100 --extra_params=kitchentoaster,kitchentoaster_randomness,no_action_rand --num_demos=15 --cam_index=1

python teleop_franka.py --demo_folder=cabinetrealv2 --offset=0 --hz=2 --max_path_length=100 --extra_params=cabinet,cabinet_mid_randomness,no_action_rand --num_demos=15 --cam_index=2

python teleop_franka.py --demo_folder=cupntrashreal --offset=0 --hz=2 --max_path_length=100 --extra_params=cupntrash,kitchentoaster_randomness,no_action_rand --num_demos=15 --cam_index=1

python teleop_franka.py --demo_folder=cupntrashreal --offset=0 --hz=2 --max_path_length=100 --extra_params=cupntrash,kitchentoaster_randomness,no_action_rand --num_demos=15 --cam_index=1
python teleop_franka.py --demo_folder=dishsinklabreal --offset=0 --hz=2 --max_path_length=100 --extra_params=dishsinklab,dishsinklab_randomness,no_action_rand --num_demos=15 --cam_index=1

#### Train policy from the real world

python distillation_online.py --policy_train_steps=15000 --random_augmentation --num_cameras=1 --render_images --policy_bc_epochs=1 --from_disk --extra_params=booknshelve,booknshelve_high_randomness --sampling_expert=1 --filename=demos/booknshelve --datafolder=/home/marcel/PolicyLearning/ --usd_path=/home/marcel/USDAssets/scenes/ --num_trajs_per_step=5 --num_trajs_eval=20 --num_envs=12 --store_traj --visualize_traj --pcd_randomness=high_pcd_randomness


python distillation_online.py --policy_train_steps=15000 --random_augmentation --num_cameras=1 --render_images --policy_bc_epochs=1 --from_disk --extra_params=booknshelve,booknshelve_high_randomness --sampling_expert=1 --filename=demos/booknshelve --datafolder=/home/marcel/PolicyLearning/ --usd_path=/home/marcel/USDAssets/scenes/ --num_trajs_per_step=5 --num_trajs_eval=20 --num_envs=12 --store_traj --visualize_traj --pcd_randomness=default_pcd_randomness --layers=256,256 --pool=max --lr=0.0003 --pcd_encoder_type=dense_conv --policy_batch_size=32 --num_trajs_per_step=1


#### Collect teacher demos from the real world in sim


python distillation_online.py --policy_train_steps=15000 --random_augmentation --extra_params=wooden_drawer_bigger --num_cameras=1 --render_images --policy_bc_epochs=1 --eval_freq=1 --filename=trialdaggerlittlerandomness --max_path_length=60 --extra_params=wooden_drawer_bigger,drawer_mid_randomness --sampling_expert=1 --random_config=drawer_debug_more_randomness --filename=demos/realworldfranka --datafolder=/home/marcel/PolicyLearning/ --random_config=drawer_debug_more_randomness --filename=demos/distillrealworldfranka --usd_path=/home/marcel/USDAssets/objects/ --num_trajs_per_step=5 --num_trajs_eval=20 --num_envs=12 --only_collect --run_path=locobot-learn/distillation_online/kis4qetw --model_name=policy_distill_step_439 --store_traj

python distillation_online.py --policy_train_steps=15000 --random_augmentation --num_cameras=1 --render_images --policy_bc_epochs=1 --eval_freq=1 --max_path_length=60 --sampling_expert=1 --random_config=drawer_debug_more_randomness --datafolder=/home/marcel/PolicyLearning/ --extra_params=booknshelve,booknshelve_debug_low_randomness,booknshelve2 --filename=demos/distillrealworldfrankabooknshelve --usd_path=/home/marcel/USDAssets/scenes/ --num_trajs_per_step=5 --num_trajs_eval=20 --num_envs=12 --only_collect --run_path=locobot-learn/distillation_student_from_state_booknshelve/9bpf5q7x --model_name=policy_distill_step_10000 --store_traj --pcd_randomness=extreme_pcd_randomness --layers=256,256 --pool=max --lr=0.0003 --pcd_encoder_type=dense_conv --policy_batch_size=32

python distillation_online.py --policy_train_steps=15000 --random_augmentation --num_cameras=1 --render_images --policy_bc_epochs=1 --eval_freq=1 --max_path_length=60 --sampling_expert=1 --random_config=drawer_debug_more_randomness --datafolder=/home/marcel/PolicyLearning/ --extra_params=booknshelve,booknshelve_debug_low_randomness,booknshelve2,synthetic_pcd --filename=demos/distillrealworldfrankabooknshelve --usd_path=/data/pulkitag/misc/marcel/USDAssets/scenes/ --num_trajs_per_step=5 --num_trajs_eval=20 --num_envs=12 --only_collect --run_path=locobot-learn/distillation_student_from_state_booknshelve/9bpf5q7x --model_name=policy_distill_step_10000 --store_traj --pcd_randomness=extreme_pcd_randomness --layers=256,256 --pool=max --lr=0.0003 --pcd_encoder_type=dense_conv --policy_batch_size=32 --use_synthetic_pcd

#### Distill from the real world in sim


python distillation_online.py --policy_train_steps=15000 --random_augmentation --extra_params=wooden_drawer_bigger --policy_bc_epochs=1 --eval_freq=1 --filename=trialdaggerlittlerandomness --max_path_length=60 --extra_params=wooden_drawer_bigger,drawer_mid_randomness --sampling_expert=1 --datafolder=/home/marcel/PolicyLearning/ --random_config=drawer_debug_more_randomness --filename=demos/distillrealworldfranka --usd_path=/home/marcel/USDAssets/objects/ --num_trajs_per_step=5 --num_trajs_eval=20 --num_envs=1024 --from_disk --student_from_state

python distillation_online.py --policy_train_steps=15000 --random_augmentation --extra_params=wooden_drawer_bigger --policy_bc_epochs=1 --eval_freq=1 --filename=trialdaggerlittlerandomness --max_path_length=60 --extra_params=wooden_drawer_bigger,drawer_mid_randomness --sampling_expert=1 --datafolder=/home/marcel/PolicyLearning/ --random_config=drawer_debug_more_randomness --filename=demos/distillrealworldfranka --usd_path=/home/marcel/USDAssets/objects/ --num_trajs_per_step=5 --num_trajs_eval=20 --num_envs=1024 --from_disk --student_from_state

#### [OPTIONAL] Train from sim demos in sim


python distillation.py --folder_name=demos/isaac-envbooknshelve --num_demos=23 --policy_train_steps=40000 --from_state --eval_freq=1000 --extra_params=booknshelve,booknshelve_low2_randomness --num_envs=2048

### Start PPO
 
python launch_ppo.py --n_steps=60 --max_path_length=60 --model_name=policy_distill_step_29000 --run_path=locobot-learn/distillation/c4ylw76e --extra_params=wooden_drawer_bigger,drawer_mid_randomness --num_envs=2048 --usd_name=drawerbiggerhandle.usda --usd_path=/home/marcel/USDAssets/objects/

python launch_ppo.py --num_envs=2048 --n_steps=60 --max_path_length=60 --model_name=policy_distill_step_745 --run_path=locobot-learn/distillation_student_from_state/g03jmpb5 --extra_params=wooden_drawer_bigger,drawer_debug_more_randomness

python student_evaluation.py --env_name=isaac-env --run_path=locobot-learn/drawerbiggerhandle.usdppo-finetune/esruolkr --model_name=policy_finetune_step_11 --render_images --max_path_length=50 --extra_params=wooden_drawer_bigger,drawer_debug_more_randomness --num_envs=12
python launch_ppo.py --num_envs=2048 --n_steps=80 --max_path_length=80 --model_name=policy_distill_step_745 --run_path=locobot-learn/distillation_student_from_state/g03jmpb5 --extra_params=wooden_drawer_bigger,drawer_debug_high_randomness --bc_loss --bc_coef=0.1


- book shelve

python launch_ppo.py --num_envs=2048 --n_steps=150 --max_path_length=150 --extra_params=booknshelve,booknshelve_low2_randomness --model_name=policy_distill_step_34000 --run_path=locobot-learn/distillation/xsguyshw

python launch_ppo.py --num_envs=1024 --n_steps=110 --max_path_length=110 --extra_params=booknshelve,booknshelve_low_randomness,no_action_rand --model_name=policy_distill_step_34000 --run_path=locobot-learn/distillation/xsguyshw

### Run PPO with BC loss
python launch_ppo.py --num_envs=2048 --n_steps=60 --max_path_length=60 --model_name=policy_distill_step_745 --run_path=locobot-learn/distillation_student_from_state/g03jmpb5 --extra_params=wooden_drawer_bigger,drawer_debug_more_randomnes --filename=isaac-envdistillrealworld --datafolder=demos --bc_loss --bc_coef=1

python launch_ppo.py --num_envs=4096 --n_steps=130 --max_path_length=130 --extra_params=booknshelve,booknshelve,booknshelve_debug_mid_randomness,no_action_rand  --usd_path=/home/marcel/USDAssets/scenes --bc_loss --bc_coef=0.1 --filename=isaac-envbooknshelvenew --datafolder=demos --num_demos=12 --ppo_batch_size=31257

#### Collect teacher rollout trajectories

python distillation_online.py --policy_train_steps=15000 --random_augmentation --extra_params=wooden_drawer_bigger --num_cameras=1 --render_images --num_envs=10 --num_trajs_per_step=30 --policy_bc_epochs=1 --eval_freq=1 --filename=trialdaggerlittlerandomness --max_path_length=60 --extra_params=wooden_drawer_bigger,drawer_mid_randomness --sampling_expert=1 --random_config=drawer_debug_more_randomness --filename=demos/realworldfranka --random_config=drawer_debug_more_randomness --filename=demos/distillrealworldfranka --num_trajs_per_step=5 --num_trajs_eval=20 --num_envs=12 --only_collect --run_path=locobot-learn/drawerbiggerhandle.usdppo-finetune/esruolkr --model_name=policy_finetune_step_11 --store_traj --teacher_from_state

python distillation_online.py --policy_train_steps=15000 --random_augmentation --num_cameras=1 --render_images --num_envs=10 --num_trajs_per_step=30 --policy_bc_epochs=1 --eval_freq=1 --max_path_length=60 --extra_params=wooden_drawer_bigger --sampling_expert=1 --random_config=drawer_debug_more_randomness --filename=demos/biggerdrawerclosingloop2 --num_trajs_per_step=5 --num_trajs_eval=20 --num_envs=7 --only_collect --run_path=locobot-learn/drawerbiggerhandle.usdppo-finetune/j58rpjah --model_name=policy_finetune_step_564 --store_traj --teacher_from_state --node=81 --seed=81

#### Online distillation
python distillation_online.py --policy_train_steps=15000 --random_augmentation --extra_params=wooden_drawer_bigger --num_cameras=1 --render_images --run_path=locobot-learn/drawer_wooden_final.usdppo-finetune/avtwg42u --model_name=policy_finetune_step_418

python distillation_online.py --policy_train_steps=15000 --random_augmentation --extra_params=wooden_drawer_bigger --num_cameras=1 --render_images --run_path=locobot-learn/drawer_wooden_final.usdppo-finetune/avtwg42u --model_name=policy_finetune_step_418 --num_trajs_per_step=30 --policy_bc_epochs=1 --eval_freq=1 --filename=trialdaggerlittlerandomness --render_images --max_path_length=60 --extra_params=wooden_drawer_final2 --num_envs=12 --random_config=drawer_low_randomness --store_traj --filename=rewind --from_disk --eval_freq=20 --num_trajs_eval=20

python distillation_online.py --policy_train_steps=15000 --random_augmentation --num_cameras=1 --render_images --num_trajs_per_step=30 --policy_bc_epochs=1 --render_images --max_path_length=60 --num_envs=12 --random_config=drawer_debug_more_randomness --filename=biggerdrawerclosingloop2 --from_disk --eval_freq=20 --num_trajs_eval=20 --policy_train_steps=10000000

#### Evaluation in the real world
```
conda activate franka-env-new-cuda
export RDT_SOURCE_DIR=~/improbable_rdt/src/rdt
export PB_PLANNING_SOURCE_DIR=~/improbable_rdt/pybullet-planning/
```


python evaluate_policy_real.py --run_path=locobot-learn/distillation_online/jz5u2z7t --model_name=policy_distill_step_966 --max_path_length=80 --cam_index=2 --hz=4 --extra_params=wooden_drawer_bigger,drawer_debug_more_randomness --pcd_randomness=high_pcd_randomness

- booknshelve
python evaluate_policy_real.py --run_path=locobot-learn/distillation_student_from_state_booknshelve/qy3oxm3b --model_name=policy_distill_step_270 --max_path_length=140 --cam_index=2 --hz=4 --extra_params=booknshelve,booknshelve_high_randomness,no_action_rand --pcd_randomness=default_pcd_randomness --pcd_encoder_type=dense_conv --voxel_size=0.01 --layers=256,256 --pool=max


python evaluate_policy_real.py --run_path=locobot-learn/distillation_student_from_state_booknshelve/qy3oxm3b --model_name=policy_distill_step_275 --max_path_length=140 --cam_index=2 --hz=4 --extra_params=booknshelve,booknshelve_high_randomness,no_action_rand --pcd_randomness=default_pcd_randomness --pcd_encoder_type=dense_conv --voxel_size=0.01 --layers=256,256 --pool=max --use_state --hz=2

python evaluate_policy_real.py --run_path=locobot-learn/distillation_student_from_state_booknshelve/qy3oxm3b --model_name=policy_distill_step_275 --max_path_length=140 --cam_index=2 --hz=4 --extra_params=booknshelve,booknshelve_high_randomness,no_action_rand --pcd_randomness=default_pcd_randomness --pcd_encoder_type=dense_conv --voxel_size=0.01 --layers=256,256 --pool=max --use_state --hz=1


python evaluate_policy_real.py --run_path=locobot-learn/distillation_student_from_state_mugnrack/yefl3z6i --model_name=policy_distill_step_419 --max_path_length=140 --cam_index=2 --extra_params=mugnrack,mugnrack_high_randomness,no_action_rand --pcd_randomness=default_pcd_randomness --pcd_encoder_type=dense_conv --voxel_size=0.01 --layers=256,256 --pool=max --use_state --hz=2

python evaluate_policy_real.py --run_path=locobot-learn/distillation_dishnrack/hrl2yg2v --model_name=policy_distill_step_8 --max_path_length=140 --cam_index=2 --extra_params=mugnrack,mugnrack_high_randomness,no_action_rand --pcd_randomness=default_pcd_randomness --pcd_encoder_type=dense_conv --use_state --hz=2


python evaluate_policy_real.py --model_name=policy_distill_step_26 --run_path=locobot-learn/distillation_dishnrack/vk8w4pim --max_path_length=140 --cam_index=2 --extra_params=mugnrack,mugnrack_high_randomness,no_action_rand --pcd_randomness=default_pcd_randomness --pcd_encoder_type=dense_conv --use_state --hz=2

- dishinrack sim co-training
python evaluate_policy_real.py --model_name=policy_distill_step_25 --run_path=locobot-learn/distillation_dishnrack/cxw65dfq --max_path_length=140 --cam_index=2 --extra_params=mugnrack,mugnrack_high_randomness,no_action_rand --pcd_randomness=default_pcd_randomness --pcd_encoder_type=dense_conv --use_state --hz=2


- booknshelf sim co-training
python evaluate_policy_real.py --model_name=policy_distill_step_18 --run_path=locobot-learn/distillation_booknshelve/v6bagyrg --max_path_length=140 --cam_index=2 --extra_params=mugnrack,mugnrack_high_randomness,no_action_rand --pcd_randomness=default_pcd_randomness --pcd_encoder_type=dense_conv --use_state --hz=2


- mug and shelf

python evaluate_policy_real.py --run_path=locobot-learn/distillation_mugandshelf/7l1whf4r --model_name=policy_distill_step_72 --max_path_length=80 --cam_index=2 --hz=2 --use_state --hz=2 --extra_params=mugandshelf,mugandshelf_high_rot_randomness,no_action_rand 

python evaluate_policy_real.py --run_path=locobot-learn/distillation_drawer_bigger/kjjyh92p --model_name=policy_distill_step_31 --max_path_length=80 --cam_index=2 --hz=2 --use_state --hz=2 --extra_params=mugandshelf,mugandshelf_high_rot_randomness,no_action_rand 

python evaluate_policy_real.py --run_path=locobot-learn/distillation_booknshelve/vnigo6di --model_name=policy_distill_step_49 --max_path_length=80 --cam_index=2 --hz=2 --use_state --hz=2 --extra_params=--extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --max_path_length=130


python evaluate_policy_real.py --run_path=locobot-learn/distillation_mugandshelf/rne3qxrm --model_name=policy_distill_step_99 --max_path_length=80 --cam_index=2 --hz=2 --use_state --hz=2 --extra_params=mugandshelf,mugandshelf_high_rot_randomness,no_action_rand --max_path_length=130 --gripper_force=100


- dishinrack
, 47
python evaluate_policy_real.py --run_path=locobot-learn/distillation_dishnrack/0nv1x4bi --model_name=policy_distill_step_47 --max_path_length=80 --cam_index=2 --hz=2 --use_state --hz=2 --extra_params=dishnrackv2,dishnrack_high_randomness,no_action_rand --max_path_length=130 --gripper_force=100


- simasets drawer
python evaluate_policy_real.py --run_path=locobot-learn/distillation_all_drawers/agffvjlx --model_name=policy_distill_step_114 --max_path_length=80 --cam_index=2 --hz=2 --use_state --hz=2 --extra_params=mugandshelf,mugandshelf_high_rot_randomness,no_action_rand --max_path_length=130 --gripper_force=100

- no cotraining booknshelf
python evaluate_policy_real.py --run_path=locobot-learn/distillation_booknshelve/kog6t636 --model_name=policy_distill_step_24 --max_path_length=80 --cam_index=2 --hz=2 --use_state --hz=2 --extra_params=mugandshelf,mugandshelf_high_rot_randomness,no_action_rand --max_path_length=130 --gripper_force=100 --reset_if_open

- with cotraining booknshelf
locobot-learn/distillation_booknshelve/cqcez3jj, 24
python evaluate_policy_real.py --run_path=locobot-learn/distillation_booknshelve/cqcez3jj --model_name=policy_distill_step_24 --max_path_length=80 --cam_index=2 --hz=2 --use_state --hz=2 --extra_params=mugandshelf,mugandshelf_high_rot_randomness,no_action_rand --max_path_length=130 --gripper_force=100 --reset_if_open

- dishinrack
, 
python evaluate_policy_real.py --run_path=locobot-learn/distillation_dishnrack/chxoezd4 --model_name=policy_distill_step_30 --max_path_length=80 --cam_index=2 --hz=2 --use_state --hz=2 --extra_params=mugandshelf,mugandshelf_high_rot_randomness,no_action_rand --max_path_length=130 --gripper_force=100 --reset_if_open

- drawer with sim co training
python evaluate_policy_real.py --run_path=locobot-learn/distillation_drawer_bigger/zanyktfs --model_name=policy_distill_step_13 --max_path_length=80 --cam_index=2 --hz=2 --use_state --hz=2 --extra_params=mugandshelf,mugandshelf_high_rot_randomness,no_action_rand --max_path_length=130 --gripper_force=100 --reset_if_open

###### [1.0000000e+00, 1.0000000e+00, 9.0529574e-03, 9.1591632e-01,
       2.6375528e-02, 4.0039918e-01, 4.7134092e-01, 3.0843774e-05,
       3.6080718e-01]


#### Distillation 

python distillation_online.py --policy_train_steps=15000 --random_augmentation --num_cameras=1 --render_images --num_trajs_per_step=30 --policy_bc_epochs=1 --render_images --max_path_length=60 --num_envs=12 --random_config=drawer_debug_more_randomness --filename=biggerdrawerclosingloop --from_disk --eval_freq=20 --num_trajs_eval=20 --policy_train_steps=10000000 --extra_params=wooden_drawer_bigger,drawer_debug_more_randomness --datafolder=/data/pulkitag/misc/marcel/data/

distillation_online.py --policy_train_steps=15000 --random_augmentation --num_cameras=1 --render_images --num_trajs_per_step=30 --policy_bc_epochs=1 --render_images --max_path_length=60 --num_envs=12 --random_config=drawer_debug_more_randomness --filename=biggerdrawerclosingloop2 --from_disk --eval_freq=20 --num_trajs_eval=20 --policy_train_steps=10000000 --extra_params=wooden_drawer_bigger,drawer_debug_more_randomness --datafolder=/data/pulkitag/misc/marcel/data/ --pcd_randomness=default_pcd_randomness

distillation_online.py --policy_train_steps=15000 --random_augmentation --num_cameras=1 --render_images --num_trajs_per_step=30 --policy_bc_epochs=1 --render_images --max_path_length=60 --num_envs=12 --random_config=drawer_debug_more_randomness --filename=biggerdrawerclosingloop2 --from_disk --eval_freq=20 --num_trajs_eval=20 --policy_train_steps=10000000 --extra_params=wooden_drawer_bigger,drawer_debug_more_randomness --datafolder=/data/pulkitag/misc/marcel/data/ --pcd_randomness=high_pcd_randomness


#### Evaluation in sim
eval "$(/scratch/marcel/miniconda3/bin/conda shell.bash hook)"
conda activate isaac-sim
source /scratch/marcel/isaac_sim-2022.2.1/setup_conda_env.sh

python student_evaluation.py --env_name=isaac-env --usd_name=mugandtablesite.usd --run_path=locobot-learn/distillation/myostkis --model_name=policy_distill_step_2400 --render_images --max_path_length=25

python student_evaluation.py --env_name=isaac-env --usd_name=wooden_drawer_front.usd --run_path=locobot-learn/distillation/r16ltdtr --model_name=policy_distill_step_4000 --render_images --max_path_length=50 --num_envs=7 --extra_params=wooden_drawer

python student_evaluation.py --env_name=isaac-env --usd_name=mugandtablesite.usd --run_path=locobot-learn/distillation/myostkis --model_name=policy_distill_step_2400 --render_images --max_path_length=25

python student_evaluation.py --env_name=isaac-env --run_path=locobot-learn/distillation/f26vy3sj --model_name=policy_distill_step_7000 --render_images --max_path_length=50 --extra_params=wooden_drawer_final --num_envs=7

#### Evaluate policy from the real world in sim
python student_evaluation.py --env_name=isaac-env --run_path=locobot-learn/distillation/tv10hxeq --model_name=policy_distill_step_5000 --render_images --max_path_length=50 --extra_params=wooden_drawer_bigger --num_envs=7
