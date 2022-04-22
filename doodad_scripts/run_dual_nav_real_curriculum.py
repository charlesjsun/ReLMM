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


def real_soft_q_curriculum_uniform_variant(
            min_samples_before_train=300,
            min_samples_before_normal=2000,
            min_fails_before_stop=100,
            max_fails_until_success_before_start=10,
            do_mean_std_relabeling=False,
            **kwargs,
    ):
    return deep_update(real_soft_q_variant(**kwargs), {
        "exp_name": "real-soft-q-curriculum-uniform",
        "env_kwargs": {
            "grasp_algorithm": "soft_q_curriculum",
            "grasp_algorithm_params": {
                "min_samples_before_train": min_samples_before_train,
                "min_samples_before_normal": min_samples_before_normal,
                "min_fails_before_stop": min_fails_before_stop,
                "max_fails_until_success_before_start": max_fails_until_success_before_start,
            },
            "grasp_perturbation": "random_uniform",
            "grasp_perturbation_params": {
                "num_steps": 5,
            },
            "nav_perturbation": "random_uniform",
            "nav_perturbation_params": {
                "num_steps": 5,
            },
            "do_mean_std_relabeling": do_mean_std_relabeling,
            "pause_filepath": "/home/brian/realmobile/locobot_pause",
        },
    })


TELEOP = False

# final_checkpoints = [
#     "/home/brian/ray_results/real_trained_load_nag4_bonus_uniform_3/checkpoint_1",
#     "/home/brian/ray_results/real_trained_load_nag4_bonus_uniform_5/checkpoint_5",
#     "/home/brian/ray_results/real_trained_load_nag4_bonus_uniform_6_2020-10-20T11-30-48/checkpoint_error",
#     "/home/brian/ray_results/real_trained_load_nag4_bonus_uniform_6/checkpoint_error",
#     "/home/brian/ray_results/real_trained_load_nag4_bonus_uniform_8/checkpoint_error",
#     "/home/brian/ray_results/real_trained_load_nag4_bonus_uniform_9/checkpoint_4",
#     "/home/brian/ray_results/real_trained_load_nag4_bonus_uniform_10/checkpoint_error",
#     "/home/brian/ray_results/real_trained_load_nag4_bonus_uniform_11/checkpoint_3",
# ]

# sock2000_checkpoints = [
#     "/home/brian/ray_results/real_sock2000_3/checkpoint_error",
#     "/home/brian/ray_results/real_sock2000_4/checkpoint_error",
#     "/home/brian/ray_results/real_sock2000_5/checkpoint_2",
#     "/home/brian/ray_results/real_sock2000_7/checkpoint_error",
#     "/home/brian/ray_results/real_sock2000_8/checkpoint_error",
#     "/home/brian/ray_results/real_sock2000_9/checkpoint_error",
#     "/home/brian/ray_results/real_sock2000_10/checkpoint_3",
# ]

# obstacles_sock2000_checkpoints = [
#     "/home/brian/ray_results/real_obstacles_sock2000_4/checkpoint_error",
#     "/home/brian/ray_results/real_obstacles_sock2000_5/checkpoint_error"
# ]

# obstacles2_sock2000_checkpoints = [
#     "/home/brian/ray_results/real_obstacles2_sock2000_1/checkpoint_error",
# ]

# no_obstacles_checkpoints = [
#     "/home/brian/ray_results/no_obst_sock2000_1/checkpoint_error",
#     "/home/brian/ray_results/no_obst_sock2000_2/checkpoint_error",
#     "/home/brian/ray_results/no_obst_sock2000_3/checkpoint_error",
#     "/home/brian/ray_results/no_obst_sock2000_4/checkpoint_error",
#     "/home/brian/ray_results/no_obst_sock2000_5/checkpoint_error",
#     "/home/brian/ray_results/no_obst_sock2000_6/checkpoint_error",
#     "/home/brian/ray_results/no_obst_sock2000_7/checkpoint_error",
#     "/home/brian/ray_results/no_obst_sock2000_8/checkpoint_error",
#     "/home/brian/ray_results/no_obst_sock2000_9/checkpoint_error",
#     "/home/brian/ray_results/no_obst_sock2000_10/checkpoint_error"
# ]

# no_obstacles_2_checkpoints = [
#     "/home/brian/ray_results/no_obst_2_sock2000_1/checkpoint_error",
#     "/home/brian/ray_results/no_obst_2_sock2000_2/checkpoint_error"
# ]

# no_obstacles_3_checkpoints = [
#     "/home/brian/ray_results/no_obst_3_sock2000_2/checkpoint_error",
#     "/home/brian/ray_results/no_obst_3_sock2000_3/checkpoint_error",
#     "/home/brian/ray_results/no_obst_3_sock2000_4/checkpoint_error",
#     "/home/brian/ray_results/no_obst_3_sock2000_5/checkpoint_error"
# ]

# no_obstacles_4_checkpoints = [
#     "/home/brian/ray_results/no_obst_4_sock2000_1/checkpoint_error",
#     "/home/brian/ray_results/no_obst_4_sock2000_2/checkpoint_error",
#     "/home/brian/ray_results/no_obst_4_sock2000_3/checkpoint_error",
# ]

# obstacles_1_checkpoints = [
#     "/home/brian/ray_results/obst_1_sock2000_8/checkpoint_error",
#     "/home/brian/ray_results/obst_1_sock2000_10/checkpoint_error",
# ]

curriculum_v2_checkpoints = [
    "/home/charles/ray_results/curriculum_v2_1/checkpoint_error",
    "/home/charles/ray_results/curriculum_v2_2/checkpoint_error",
    "/home/charles/ray_results/curriculum_v2_3/checkpoint_error",
    "/home/charles/ray_results/curriculum_v2_4/checkpoint_error",
    "/home/charles/ray_results/curriculum_v2_5/checkpoint_error",
    "/home/charles/ray_results/curriculum_v2_6/checkpoint_error",
    "/home/charles/ray_results/curriculum_v2_7/checkpoint_error",
    "/home/charles/ray_results/curriculum_v2_8/checkpoint_error",
    "/home/charles/ray_results/curriculum_v2_9/checkpoint_error",
    "/home/charles/ray_results/curriculum_v2_10/checkpoint_error",
    "/home/charles/ray_results/curriculum_v2_11/checkpoint_error",
    "/home/charles/ray_results/curriculum_v2_12/checkpoint_error",
    "/home/charles/ray_results/curriculum_v2_13/checkpoint_error",
    "/home/charles/ray_results/curriculum_v2_14/checkpoint_error",
    "/home/charles/ray_results/curriculum_v2_15/checkpoint_error",
    "/home/charles/ray_results/curriculum_v2_16/checkpoint_error",
    "/home/charles/ray_results/curriculum_v2_17/checkpoint_error",
    "/home/charles/ray_results/curriculum_v2_18/checkpoint_error",
    "/home/charles/ray_results/curriculum_v2_19/checkpoint_error",
    "/home/charles/ray_results/curriculum_v2_20/checkpoint_error",
]

curriculum_diverse_obj_v1_checkpoints = [

]

SAVE_FRAME_DIR = "./grasp_frames"

# variant = real_soft_q_uniform_variant(result_name="obst_2_sock2000_2",
#                             grasp_model_name="sock_2000",
#                             grasp_data_name="sock_2000",
#                             add_uncertainty_bonus=False,
#                             gpu=0, server_port=10001, gpu_percent=0.20,
#                             restore_path="",
#                             replay_pool_paths="",
#                             # restore_path=obstacles_2_checkpoints[-1],
#                             # replay_pool_paths=":".join(obstacles_2_checkpoints)
# )

variant = real_soft_q_curriculum_uniform_variant(result_name="curriculum_diverse_obj_v1_0",
                            add_uncertainty_bonus=False,
                            max_fails_until_success_before_start=10,
                            min_fails_before_stop=50,
                            min_samples_before_normal=2000,
                            min_samples_before_train=300,
                            gpu=0, server_port=10001, gpu_percent=0.20,
                            restore_path="",
                            replay_pool_paths="",
                            # restore_path=curriculum_diverse_obj_v1_checkpoints[-1],
                            # replay_pool_paths=":".join(curriculum_diverse_obj_v1_checkpoints)
)

print(get_command(**variant))

# bash -c "$(python doodad_scripts/run_dual_nav_real_curriculum.py)"
