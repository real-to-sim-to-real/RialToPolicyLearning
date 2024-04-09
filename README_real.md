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



install dependencies:


torchsparse (from the github repo):
```
python -c "$(curl -fsSL https://raw.githubusercontent.com/mit-han-lab/torchsparse/master/install.py)"
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

#### hyperparam sweep on booknshelve (dev)
distillation.py --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --max_path_length=130 --filename=booknshelvenewsimpcd,booknshelvenewsynthetic --run_path=locobot-learn/booknshelve.usdppo-finetune/ruk0ihdc --model_name=policy_finetune_step_210 --datafolder=/data/pulkitag/data/marcel/data/ --use_synthetic_pcd --max_demos=2500 --eval_freq=1 --use_state --visualize_traj --max_demos=5000 --policy_batch_size=20 --num_envs=100 --policy_train_steps=75 --unet_num_levels=6 --unet_f_maps=64 

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
start meshcat-server
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
python evaluate_policy_real.py --run_path=locobot-learn/distillation_drawer_bigger/2mt5p1wf --model_name=policy_distill_step_74 --max_path_length=300 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --total_loop_time=0.3 --interp_steps=30 --start_interp_offset=3 --gripper_force=100 --nb_open=3 --hz=2 --gripper_force=100 --total_loop_time=0.1 --interp_steps=10 --start_interp_offset=5 --gripper_force=50 --nb_open=5 --background_loop

- drawer real2sim2real

<!-- python evaluate_policy_real.py --run_path=locobot-learn/distillation_drawer_bigger/070k0p1g --model_name=policy_distill_step_86 --max_path_length=300 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --background_loop --total_loop_time=0.3 --interp_steps=30 --start_interp_offset=3 --gripper_force=100 --nb_open=12

python evaluate_policy_real.py --run_path=locobot-learn/distillation_drawer_bigger/nm2exjub --model_name=policy_distill_step_40 --max_path_length=300 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --background_loop --total_loop_time=0.3 --interp_steps=30 --start_interp_offset=3 --gripper_force=100 --nb_open=12 -->

python evaluate_policy_real.py --run_path=locobot-learn/distillation_drawer_bigger/nm2exjub --model_name=policy_distill_step_35 --max_path_length=300 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100  --total_loop_time=0.3 --interp_steps=30 --start_interp_offset=3 --gripper_force=100 --nb_open=1 --background_loop

- kitchentoaster 
python evaluate_policy_real.py --run_path=locobot-learn/distillation_kitchentoaster/x6pzujl0 --model_name=policy_distill_step_15 --max_path_length=130 --cam_index=1 --extra_params=kitchentoaster,kitchentoaster_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --background_loop --total_loop_time=0.3 --interp_steps=30 --start_interp_offset=3 --gripper_force=100


- fast controller cabinet: 
python evaluate_policy_real.py --run_path=locobot-learn/distillation_cabinet/h1c8advw --model_name=policy_distill_step_17 --max_path_length=300 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --total_loop_time=0.1 --interp_steps=10 --start_interp_offset=5 --gripper_force=50 --nb_open=20 --background_loop

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
 
python evaluate_policy_real.py --run_path=locobot-learn/distillation_mugandshelf/rne3qxrm --model_name=policy_distill_step_80 --max_path_length=400 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --total_loop_time=0.1 --interp_steps=10 --start_interp_offset=5 --gripper_force=50 --nb_open=10 --background_loop

- booknshelve
python evaluate_policy_real.py --run_path=locobot-learn/distillation_booknshelve/cqcez3jj --model_name=policy_distill_step_24 --max_path_length=300 --max_path_length=300 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --total_loop_time=0.1 --interp_steps=10 --start_interp_offset=5 --gripper_force=50 --nb_open=1

- dishinrack
<!-- python evaluate_policy_real.py --run_path=locobot-learn/distillation_dishnrack/chxoezd4 --model_name=policy_distill_step_30 --max_path_length=130 --cam_index=2 --extra_params=booknshelve,booknshelve,booknshelve_debug_high_rot_randomness,no_action_rand --use_state --hz=2 --gripper_force=100 --background_loop --total_loop_time=0.3 --interp_steps=30 --start_interp_offset=3 -->
