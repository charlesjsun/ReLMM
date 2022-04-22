# Softlearning

Softlearning is a deep reinforcement learning toolbox for training maximum entropy policies in continuous domains. The implementation is fairly thin and primarily optimized for our own development purposes. It utilizes the tf.keras modules for most of the model classes (e.g. policies and value functions). We use Ray for the experiment orchestration. Ray Tune and Autoscaler implement several neat features that enable us to seamlessly run the same experiment scripts that we use for local prototyping to launch large-scale experiments on any chosen cloud service (e.g. GCP or AWS), and intelligently parallelize and distribute training for effective resource allocation.

This implementation uses Tensorflow. For a PyTorch implementation of soft actor-critic, take a look at [rlkit](https://github.com/vitchyr/rlkit).

# Getting Started

## Prerequisites

The environment can be run either locally using conda or inside a docker container. For conda installation, you need to have [Conda](https://conda.io/docs/user-guide/install/index.html) installed. For docker installation you will need to have [Docker](https://docs.docker.com/engine/installation/) and [Docker Compose](https://docs.docker.com/compose/install/) installed. Also, most of our environments currently require a [MuJoCo](https://www.roboti.us/license.html) license.

## Conda Installation

1. [Download](https://www.roboti.us/index.html) and install MuJoCo 1.50 and 2.00 from the MuJoCo website. We assume that the MuJoCo files are extracted to the default location (`~/.mujoco/mjpro150` and `~/.mujoco/mujoco200_{platform}`). Unfortunately, `gym` and `dm_control` expect different paths for MuJoCo 2.00 installation, which is why you will need to have it installed both in `~/.mujoco/mujoco200_{platform}` and `~/.mujoco/mujoco200`. The easiest way is to create a symlink from `~/.mujoco/mujoco200_{plaftorm}` -> `~/.mujoco/mujoco200` with: `ln -s ~/.mujoco/mujoco200_{platform} ~/.mujoco/mujoco200`.

2. Copy your MuJoCo license key (mjkey.txt) to ~/.mujoco/mjkey.txt:

3. Clone `softlearning`
```
git clone https://github.com/rail-berkeley/softlearning.git ${SOFTLEARNING_PATH}
```

4. Create and activate conda environment, install softlearning to enable command line interface.
```
cd ${SOFTLEARNING_PATH}
conda env create -f environment.yml
conda activate softlearning
pip install -e ${SOFTLEARNING_PATH}
```

The environment should be ready to run. See examples section for examples of how to train and simulate the agents.

Finally, to deactivate and remove the conda environment:
```
conda deactivate
conda remove --name softlearning --all
```

## Docker Installation

### docker-compose
To build the image and run the container:
```
export MJKEY="$(cat ~/.mujoco/mjkey.txt)" \
    && docker-compose \
        -f ./docker/docker-compose.dev.cpu.yml \
        up \
        -d \
        --force-recreate
```

You can access the container with the typical Docker [exec](https://docs.docker.com/engine/reference/commandline/exec/)-command, i.e.

```
docker exec -it softlearning bash
```

See examples section for examples of how to train and simulate the agents.

Finally, to clean up the docker setup:
```
docker-compose \
    -f ./docker/docker-compose.dev.cpu.yml \
    down \
    --rmi all \
    --volumes
```

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

# References
The algorithms are based on the following papers:

*Soft Actor-Critic Algorithms and Applications*.</br>
Tuomas Haarnoja*, Aurick Zhou*, Kristian Hartikainen*, George Tucker, Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter Abbeel, and Sergey Levine.
arXiv preprint, 2018.</br>
[paper](https://arxiv.org/abs/1812.05905)  |  [videos](https://sites.google.com/view/sac-and-applications)

*Latent Space Policies for Hierarchical Reinforcement Learning*.</br>
Tuomas Haarnoja*, Kristian Hartikainen*, Pieter Abbeel, and Sergey Levine.
International Conference on Machine Learning (ICML), 2018.</br>
[paper](https://arxiv.org/abs/1804.02808) | [videos](https://sites.google.com/view/latent-space-deep-rl)

*Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*.</br>
Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine.
International Conference on Machine Learning (ICML), 2018.</br>
[paper](https://arxiv.org/abs/1801.01290) | [videos](https://sites.google.com/view/soft-actor-critic)

*Composable Deep Reinforcement Learning for Robotic Manipulation*.</br>
Tuomas Haarnoja, Vitchyr Pong, Aurick Zhou, Murtaza Dalal, Pieter Abbeel, Sergey Levine.
International Conference on Robotics and Automation (ICRA), 2018.</br>
[paper](https://arxiv.org/abs/1803.06773) | [videos](https://sites.google.com/view/composing-real-world-policies)

*Reinforcement Learning with Deep Energy-Based Policies*.</br>
Tuomas Haarnoja*, Haoran Tang*, Pieter Abbeel, Sergey Levine.
International Conference on Machine Learning (ICML), 2017.</br>
[paper](https://arxiv.org/abs/1702.08165) | [videos](https://sites.google.com/view/softqlearning/home)

If Softlearning helps you in your academic research, you are encouraged to cite our paper. Here is an example bibtex:
```
@techreport{haarnoja2018sacapps,
  title={Soft Actor-Critic Algorithms and Applications},
  author={Tuomas Haarnoja and Aurick Zhou and Kristian Hartikainen and George Tucker and Sehoon Ha and Jie Tan and Vikash Kumar and Henry Zhu and Abhishek Gupta and Pieter Abbeel and Sergey Levine},
  journal={arXiv preprint arXiv:1812.05905},
  year={2018}
}
```
