# ReLMM

Associated code for our paper, ReLMM: Practical RL for Learning Mobile Manipulation Skills using Only Onboard Sensors. See our [website](https://sites.google.com/view/relmm/home) for more details.

Codebase is written on top of [softlearning](https://github.com/rail-berkeley/softlearning)

## Real world setup configuration

Set up locobot as recommended by the manufacturer. We recommend replacing the pyrobot distribution with the following version
to avoid the issues with installation of Ubuntu 18.04: https://github.com/rail-berkeley/pyrobot

1. Before running the experiment, source a script similar to the following one:
```
export PYTHONPATH=/home/<user>/mobilemanipulation/softlearning:$PYTHONPATH

conda activate mobilemanipulation
export PYTHONPATH=/home/<user>/mobilemanipulation:$PYTHONPATH
export ROS_MASTER_URI=http://<locobot_address>:11311
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

2. Start the robot stack on Locobot, including the [server.py](https://github.com/charlesjsun/locobot_interface/blob/master/scripts/server.py)) 

3. Use the pretrained grasping model as described in the next section or pretrain the grasping model yourself:
```
python others/main_real.py
```

You could consider changing `min_samples_before_train` to 500 in 'others/main_real.py'.  

4. Finally, to run the experiment with navigation, use:
```
bash -c "$(python doodad_scripts/run_dual_nav_real_curriculum.py)"
```

## Pretrained models

You can use the pretrained grasping models by placing them in a new folder: 'mobilemanipulation/locobot/'

https://drive.google.com/drive/folders/1y1ZnnfPmPeh1DPH98KtaXqg67LwjK-tj?usp=sharing
