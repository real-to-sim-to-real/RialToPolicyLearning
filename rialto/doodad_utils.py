import os

import doodad as pd
import doodad.ssh as ssh
import doodad.mount as mount
from doodad.easy_sweep import hyper_sweep
import os.path as osp
import glob


PROJECT_DIR = osp.realpath(osp.join(osp.dirname(__file__),'../'))
LOCAL_OUTPUT_DIR = osp.join(PROJECT_DIR,'data')

# Set up code and output directories
REMOTE_OUTPUT_DIR = '/tmp/outputs'  # this is the directory visible to the target
REMOTE_DATA_DIR = '/tmp/data'

main_mount = mount.MountLocal(local_dir=PROJECT_DIR, pythonpath=True, filter_dir=('data','analysis','dependencies'))

# code_dependencies = [
#     item
#     for item in glob.glob(osp.join(PROJECT_DIR, "dependencies/*"))
#     if osp.isdir(item)
# ]

code_mounts = [main_mount]
code_mounts.append(mount.MountLocal(local_dir=osp.join(PROJECT_DIR, "dependencies"), pythonpath=True))
code_mounts.append(mount.MountLocal(local_dir=osp.expanduser('~/.mujoco'), mount_point='/root/.mujoco'))

instance_types = {
    'c4.large': dict(instance_type='c4.large',spot_price=0.20),
    'c4.xlarge': dict(instance_type='c4.xlarge',spot_price=0.20),
    'c4.2xlarge': dict(instance_type='c4.2xlarge',spot_price=0.50),
    'c4.4xlarge': dict(instance_type='c4.4xlarge',spot_price=0.50),
}


def launch(method, params, mode='ec2', data_dependencies=dict(), repeat=1, instance_type='c4.xlarge'):

    params['output_dir'] = [REMOTE_OUTPUT_DIR]
    params['data_dir'] = [REMOTE_DATA_DIR]

    if mode == 'local':
        doodad_mode = pd.mode.Local()
        params['output_dir'] = [LOCAL_OUTPUT_DIR]
    elif mode == 'docker':
        doodad_mode = pd.mode.LocalDocker(
            image='dibyaghosh/gcsl:0.1'
        )
    elif mode == 'ec2':
        assert instance_type in instance_types
        doodad_mode = pd.mode.EC2AutoconfigDocker(
            image='dibyaghosh/gcsl:0.1',
            region='us-west-1',  # EC2 region
            s3_log_prefix='gcsl', # Folder to store log files under
            s3_log_name='gcsl',
            terminate=True,  # Whether to terminate on finishing job
            **instance_types[instance_type]
        )

    data_mounts = [
        mount.MountLocal(local_dir=osp.realpath(directory), mount_point=osp.join(REMOTE_DATA_DIR,remote_name))
        for remote_name,directory in data_dependencies.items()
    ]

    if mode == 'local':
        output_mounts = []
    elif mode == 'docker' or mode == 'ssh':
        output_dir = osp.join(LOCAL_OUTPUT_DIR, 'docker/')
        output_mounts= [mount.MountLocal(local_dir=output_dir, mount_point=REMOTE_OUTPUT_DIR,output=True)]
    elif mode == 'ec2':
        output_mounts = [mount.MountS3(s3_path='data',mount_point=REMOTE_OUTPUT_DIR,output=True)]


    mounts = code_mounts + data_mounts + output_mounts

    hyper_sweep.run_sweep_doodad(method, params, doodad_mode, mounts, repeat=repeat)
