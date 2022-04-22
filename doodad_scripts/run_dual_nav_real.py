"""
Call in the root directory (i.e. ../ from the directory this file is in)
"""

import json
import multiprocessing

import numpy as np

from softlearning.utils.dict import deep_update


def get_command(
        task="RealNavigationGraspingDualPerturbation-v0",
        exp_name="real-nav-dual-pert",
        result_name="real_0",
        server_port=10001,
        gpu=0,
        gpu_percent=0.45,
        env_kwargs={},
        restore_path="",
        replay_pool_paths="",
):
    def process_kwargs(kwargs):
        json_str = json.dumps(kwargs)
        processed = json_str.replace("\"", "\\quote")
        return processed

    cmds = []
    cmds.append(
        f'RAY_MEMORY_MONITOR_ERROR_THRESHOLD=1.000001 CUDA_VISIBLE_DEVICES={gpu} python softlearning/scripts/console_scripts.py run_example_local examples.dual_perturbation \
--algorithm SAC \
--policy gaussian \
--universe gym \
--domain Locobot \
--task {task} \
--exp-name {exp_name} \
--result-name {result_name} \
--trial-cpus 1 \
--trial-gpus {gpu_percent} \
--max-failures 0 \
--run-eagerly False \
--server-port {server_port} \
--env-kwargs "{process_kwargs(env_kwargs)}" \
--restore-path "{restore_path}" \
--replay-pool-paths "{replay_pool_paths}"'
    )

    full_cmd = ";".join(cmds)

    return full_cmd


def base_variant(result_name, gpu, gpu_percent, server_port, restore_path, replay_pool_paths):
    return {
        "result_name": result_name,
        "gpu": gpu,
        "gpu_percent": gpu_percent,
        "server_port": server_port,
        "restore_path": restore_path,
        "replay_pool_paths": replay_pool_paths,
        "env_kwargs": {
            "grasp_perturbation": "none",
            "nav_perturbation": "none",
            "grasp_algorithm": "vacuum",
            "do_teleop": TELEOP,
        },
    }


def real_soft_q_variant(grasp_data_name=None, grasp_model_name=None, add_uncertainty_bonus=False, **kwargs):
    return deep_update(base_variant(**kwargs), {
        "exp_name": "real-soft-q",
        "env_kwargs": {
            "grasp_algorithm": "soft_q",
            "grasp_algorithm_params": {
                "grasp_data_name": grasp_data_name,
                "grasp_model_name": grasp_model_name,
                "save_frame_dir": SAVE_FRAME_DIR,
            },
            "add_uncertainty_bonus": add_uncertainty_bonus,
            "alive_penalty": 1.0,
        },
    })


def real_soft_q_uniform_variant(**kwargs):
    return deep_update(real_soft_q_variant(**kwargs), {
        "exp_name": "real-soft-q-uniform",
        "env_kwargs": {
            "grasp_perturbation": "random_uniform",
            "grasp_perturbation_params": {
                "num_steps": 5,
            },
            "nav_perturbation": "random_uniform",
            "nav_perturbation_params": {
                "num_steps": 5,
            },
            "pause_filepath": "/home/brian/realmobile/locobot_pause",
        },
    })


TELEOP = False

obstacles_2_checkpoints = [
]

SAVE_FRAME_DIR = "./grasp_frames"

variant = real_soft_q_uniform_variant(result_name="obst_2_sock2000_2",
                                      grasp_model_name="sock_2000",
                                      grasp_data_name="sock_2000",
                                      add_uncertainty_bonus=False,
                                      gpu=0, server_port=10001, gpu_percent=0.20,
                                      restore_path="",
                                      replay_pool_paths="",
                                      # restore_path=obstacles_2_checkpoints[-1],
                                      # replay_pool_paths=":".join(obstacles_2_checkpoints)
                                      )

print(get_command(**variant))

# bash -c "$(python doodad_scripts/run_dual_nav_real.py)"
