"""
Call in the root directory (i.e. ../ from the directory this file is in)
"""

import doodad as dd
import doodad.ssh as ssh
import doodad.mount as mount

import json
import multiprocessing

import numpy as np

from softlearning.utils.dict import deep_update

def get_command(
    task="NavigationGraspingDualPerturbationOracle-v0",
    exp_name="nav-auto-vac-oracle-reset-respawn",
    result_name="oracle_1",
    server_port=10001,
    gpu=0,
    gpu_percent=0.45,
    env_kwargs={},
    eval_env_kwargs={}):
    
    def process_kwargs(kwargs):
        json_str = json.dumps(kwargs)
        processed = json_str.replace("\"", "\\quote")
        return processed

    cmds = []
    cmds.append("source /etc/bash.bashrc")
    cmds.append("source activate softlearning")
    cmds.append("cd /home/charlesjsun/mobilemanipulation")
    cmds.append("export RAY_MEMORY_MONITOR_ERROR_THRESHOLD=1.000001")
    cmds.append(f"export CUDA_VISIBLE_DEVICES={gpu}")
    cmds.append("export PYTHONPATH=/home/charlesjsun/mobilemanipulation")
    cmds.append(
f'python softlearning/scripts/console_scripts.py run_example_local examples.dual_perturbation \
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
--eval-env-kwargs "{process_kwargs(eval_env_kwargs)}"')
    
    full_cmd = ";".join(cmds)

    return full_cmd

def base_variant(num_objects, result_name, gpu, gpu_percent, server_port, 
                room_name="single", 
                object_name="greensquareball",
                extra_room_params={}):
    return {
        "result_name": result_name,
        "gpu": gpu,
        "gpu_percent": gpu_percent,
        "server_port": server_port,
        "env_kwargs": {
            "grasp_perturbation": "none",
            "nav_perturbation": "none",
            "grasp_algorithm": "vacuum",
            "use_auto_grasp": True,
            "renders": RENDERS,
            "do_teleop": TELEOP,
            "step_duration": 0.0,
            "room_name": room_name,
            "room_params": {
                "num_objects": num_objects,
                # "object_name": ["greensquareball", "bluesquareball", "yellowsquareball", "orangesquareball"],
                "object_name": object_name,
                **extra_room_params
            }
        },
        "eval_env_kwargs": {
            "renders": RENDERS_EVAL,
            "step_duration": 0.0,
            "do_teleop": False,
        }
    }


### ORACLE VARIANTS

def oracle_base_variant(num_nearest, do_sort=True, do_cull=False, is_relative=False, **kwargs):
    return deep_update(base_variant(**kwargs), {
        "task": "NavigationGraspingDualPerturbationOracle-v0",
        "env_kwargs": {
            "grasp_perturbation": "respawn",
            "grasp_perturbation_params": {
                "respawn_radius": np.inf,
            },
            "num_nearest": num_nearest,
            "do_sort": do_sort,
            "do_cull": do_cull,
            "is_relative": is_relative,
        },
    })

def oracle_dense_variant(**kwargs):
    return deep_update(oracle_base_variant(**kwargs), {
        "exp_name": "oracle-reset-respawn-dense",
        "env_kwargs": {
            "use_dense_reward": True,
        },
    })

def oracle_sparse_variant(**kwargs):
    return deep_update(oracle_base_variant(**kwargs), {
        "exp_name": "oracle-reset-respawn-sparse",
        "env_kwargs": {
            "use_dense_reward": False,
        },
    })

def oracle_sparse_v2_variant(**kwargs):
    return deep_update(oracle_base_variant(**kwargs), {
        "exp_name": "oracle-reset-respawn-sparse",
        "env_kwargs": {
            "use_dense_reward": False,
            "alive_penalty": 1.0,
        },
    })

def oracle_dense_rf_variant(**kwargs):
    return deep_update(oracle_base_variant(**kwargs), {
        "exp_name": "oracle-reset-free-respawn-dense",
        "env_kwargs": {
            "reset_free": True,
            "use_dense_reward": True,
            "max_ep_len": np.inf,
        },
        "eval_env_kwargs": {
            "reset_free": False,
            "max_ep_len": 500,
        }
    })

def oracle_sparse_rf_variant(**kwargs):
    return deep_update(oracle_base_variant(**kwargs), {
        "exp_name": "oracle-reset-free-respawn-sparse",
        "env_kwargs": {
            "reset_free": True,
            "use_dense_reward": False,
            "max_ep_len": np.inf,
        },
        "eval_env_kwargs": {
            "reset_free": False,
            "max_ep_len": 500,
        }
    })

def oracle_single_variant(mode="zero_one", **kwargs):
    if mode == "zero_one":
        alive_penalty = 0.0
    elif mode == "neg_zero":
        alive_penalty = 1.0

    return deep_update(oracle_base_variant(**kwargs), {
        "exp_name": "oracle-single-respawn",
        "env_kwargs": {
            "use_dense_reward": False,
            "alive_penalty": alive_penalty,
            "do_single_grasp": True,
        },
        "eval_env_kwargs": {
            "alive_penalty": 0.0,
            "do_single_grasp": False,
        }
    })

def oracle_single_soft_q_variant(grasp_data_name=None, grasp_model_name=None, add_uncertainty_bonus=False, **kwargs):
    return deep_update(oracle_single_variant(mode="neg_zero", **kwargs), {
        "exp_name": "oracle-single-soft-q-respawn",
        "env_kwargs": {
            "grasp_algorithm": "soft_q",
            "grasp_algorithm_params": {
                "grasp_data_name": grasp_data_name,
                "grasp_model_name": grasp_model_name,
            },
            "add_uncertainty_bonus": add_uncertainty_bonus,
        },
        "eval_env_kwargs": {
            "do_grasp_eval": True,
        }
    })


def oracle_single_soft_q_noautograsp_variant(**kwargs):
    return deep_update(oracle_single_soft_q_variant(**kwargs), {
        "exp_name": "oracle-single-soft-q-noauto-respawn",
        "env_kwargs": {
            "use_auto_grasp": False,
        },
    })


def oracle_single_soft_q_noautograsp_uniform_variant(eval_pert="respawn", **kwargs):
    return deep_update(oracle_single_soft_q_noautograsp_variant(**kwargs), {
        "exp_name": "oracle-single-soft-q-noauto-uniform-respawn",
        "env_kwargs": {
            "grasp_perturbation": "random_uniform",
            "grasp_perturbation_params": {
                "num_steps": 25,
            },
            "nav_perturbation": "random_uniform",
            "nav_perturbation_params": {
                "num_steps": 25,
            },
        },
        "eval_env_kwargs": {
            "grasp_perturbation": eval_pert,
            "grasp_perturbation_params": {
                "respawn_radius": np.inf,
            },
            "nav_perturbation": "none",
            "nav_perturbation_params": {
            },
        }
    })


### IMAGE VARIANTS

def image_base_variant(**kwargs):
    return deep_update(base_variant(**kwargs), {
        "task": "NavigationGraspingDualPerturbation-v0",
        "env_kwargs": {
            "grasp_perturbation": "respawn",
            "grasp_perturbation_params": {
                "respawn_radius": np.inf,
            },
        },
    })

def image_dense_variant(**kwargs):
    return deep_update(image_base_variant(**kwargs), {
        "exp_name": "image-reset-respawn-dense",
        "env_kwargs": {
            "use_dense_reward": True,
        },
    })

def image_sparse_variant(**kwargs):
    return deep_update(image_base_variant(**kwargs), {
        "exp_name": "image-reset-respawn-sparse",
        "env_kwargs": {
            "use_dense_reward": False,
        },
    })

def image_sparse_v2_variant(**kwargs):
    return deep_update(image_base_variant(**kwargs), {
        "exp_name": "image-reset-respawn-sparse",
        "env_kwargs": {
            "use_dense_reward": False,
            "alive_penalty": 1.0,
        },
    })

def image_single_variant(mode="zero_one", **kwargs):
    if mode == "zero_one":
        alive_penalty = 0.0
    else:
        alive_penalty = 1.0

    return deep_update(image_base_variant(**kwargs), {
        "exp_name": "image-single-respawn",
        "env_kwargs": {
            "use_dense_reward": False,
            "alive_penalty": alive_penalty,
            "do_single_grasp": True,
        },
        "eval_env_kwargs": {
            "alive_penalty": 0.0,
            "do_single_grasp": False,
        }
    })

def image_single_soft_q_variant(grasp_data_name=None, grasp_model_name=None, add_uncertainty_bonus=False, **kwargs):
    return deep_update(image_single_variant(mode="neg_zero", **kwargs), {
        "exp_name": "image-single-soft-q-respawn",
        "env_kwargs": {
            "grasp_algorithm": "soft_q",
            "grasp_algorithm_params": {
                "grasp_data_name": grasp_data_name,
                "grasp_model_name": grasp_model_name,
            },
            "add_uncertainty_bonus": add_uncertainty_bonus,
        },
        "eval_env_kwargs": {
            "do_grasp_eval": True,
        }
    })

def image_single_soft_q_noautograsp_variant(**kwargs):
    return deep_update(image_single_soft_q_variant(**kwargs), {
        "exp_name": "image-single-soft-q-noauto-respawn",
        "env_kwargs": {
            "use_auto_grasp": False,
        },
    })

def image_single_soft_q_noautograsp_uniform_variant(**kwargs):
    return deep_update(image_single_soft_q_noautograsp_variant(**kwargs), {
        "exp_name": "image-single-soft-q-noauto-uniform-respawn",
        "env_kwargs": {
            "grasp_perturbation": "random_uniform",
            "grasp_perturbation_params": {
                "num_steps": 20,
            },
            "nav_perturbation": "random_uniform",
            "nav_perturbation_params": {
                "num_steps": 20,
            },
        },
        "eval_env_kwargs": {
            "grasp_perturbation": "respawn",
            "grasp_perturbation_params": {
                "respawn_radius": np.inf,
            },
            "nav_perturbation": "none",
            "nav_perturbation_params": {
            },
        }
    })

def image_single_soft_q_noautograsp_none_uniform_variant(**kwargs):
    return deep_update(image_single_soft_q_noautograsp_variant(**kwargs), {
        "exp_name": "image-single-soft-q-noauto-uniform-respawn",
        "env_kwargs": {
            "grasp_perturbation": "none",
            "grasp_perturbation_params": {
                "num_steps": 20,
            },
            "nav_perturbation": "random_uniform",
            "nav_perturbation_params": {
                "num_steps": 20,
            },
        },
        "eval_env_kwargs": {
            "grasp_perturbation": "respawn",
            "grasp_perturbation_params": {
                "respawn_radius": np.inf,
            },
            "nav_perturbation": "none",
            "nav_perturbation_params": {
            },
        }
    })

def image_single_soft_q_noautograsp_uncertainty_navQ_variant(**kwargs):
    return deep_update(image_single_soft_q_noautograsp_variant(**kwargs), {
        "exp_name": "image-single-soft-q-noauto-uncertainty-respawn",
        "env_kwargs": {
            "grasp_perturbation": "grasp_uncertainty",
            "grasp_perturbation_params": {
                "num_steps": 20,
            },
            "nav_perturbation": "nav_Q",
            "nav_perturbation_params": {
                "num_steps": 20,
            },
        },
        "eval_env_kwargs": {
            "grasp_perturbation": "respawn",
            "grasp_perturbation_params": {
                "respawn_radius": np.inf,
            },
            "nav_perturbation": "none",
            "nav_perturbation_params": {
            },
        }
    })

def image_single_soft_q_noautograsp_uncertainty_uniform_variant(**kwargs):
    return deep_update(image_single_soft_q_noautograsp_variant(**kwargs), {
        "exp_name": "image-single-soft-q-noauto-uncertainty-respawn",
        "env_kwargs": {
            "grasp_perturbation": "grasp_uncertainty",
            "grasp_perturbation_params": {
                "num_steps": 20,
            },
            "nav_perturbation": "random_uniform",
            "nav_perturbation_params": {
                "num_steps": 20,
            },
        },
        "eval_env_kwargs": {
            "grasp_perturbation": "respawn",
            "grasp_perturbation_params": {
                "respawn_radius": np.inf,
            },
            "nav_perturbation": "none",
            "nav_perturbation_params": {
            },
        }
    })

### FRAME STACK VARIANTS

def frame_stack_base_variant(**kwargs):
    return deep_update(base_variant(**kwargs), {
        "task": "NavigationGraspingDualPerturbationFrameStack-v0",
        "env_kwargs": {
            "grasp_perturbation": "respawn",
            "grasp_perturbation_params": {
                "respawn_radius": np.inf,
            },
        },
    })

def frame_stack_dense_variant(**kwargs):
    return deep_update(frame_stack_base_variant(**kwargs), {
        "exp_name": "frame-stack-reset-respawn-dense",
        "env_kwargs": {
            "use_dense_reward": True,
        },
    })

def frame_stack_sparse_variant(**kwargs):
    return deep_update(frame_stack_base_variant(**kwargs), {
        "exp_name": "frame-stack-reset-respawn-sparse",
        "env_kwargs": {
            "use_dense_reward": False,
        },
    })

def frame_stack_single_variant(mode="zero_one", **kwargs):
    if mode == "zero_one":
        alive_penalty = 0.0
    else:
        alive_penalty = 1.0

    return deep_update(frame_stack_base_variant(**kwargs), {
        "exp_name": "frame-stack-single-respawn",
        "env_kwargs": {
            "use_dense_reward": False,
            "alive_penalty": alive_penalty,
            "do_single_grasp": True,
        },
        "eval_env_kwargs": {
            "alive_penalty": 0.0,
            "do_single_grasp": False,
        }
    })


RENDERS = False
TELEOP = False
RENDERS_EVAL = False

variants = [
    # *[oracle_dense_variant(result_name=f"oracle_dense_o1_n1_{i}", 
    #                        num_objects=1, num_nearest=1, 
    #                        gpu=0, gpu_percent=0.1) for i in range(4)],
    # *[oracle_sparse_variant(result_name=f"oracle_sparse_o1_n1_{i}", 
    #                        num_objects=1, num_nearest=1, 
    #                        gpu=0, gpu_percent=0.1) for i in range(4)],
    # *[oracle_dense_variant(result_name=f"oracle_dense_o20_n1_{i}", 
    #                        num_objects=20, num_nearest=1, 
    #                        gpu=1, gpu_percent=0.1) for i in range(4)],
    # *[oracle_sparse_variant(result_name=f"oracle_sparse_o20_n1_{i}", 
    #                        num_objects=20, num_nearest=1, 
    #                        gpu=1, gpu_percent=0.1) for i in range(4)],
    # *[oracle_dense_variant(result_name=f"oracle_dense_o20_n3_{i}", 
    #                        num_objects=20, num_nearest=3, 
    #                        gpu=2, gpu_percent=0.1) for i in range(4)],
    # *[oracle_sparse_variant(result_name=f"oracle_sparse_o20_n3_{i}", 
    #                        num_objects=20, num_nearest=3, 
    #                        gpu=2, gpu_percent=0.1) for i in range(4)],
    # *[oracle_dense_variant(result_name=f"oracle_dense_o20_n6_{i}", 
    #                        num_objects=20, num_nearest=6, 
    #                        gpu=3, gpu_percent=0.1) for i in range(4)],
    # *[oracle_sparse_variant(result_name=f"oracle_sparse_o20_n6_{i}", 
    #                        num_objects=20, num_nearest=6, 
    #                        gpu=4, gpu_percent=0.1) for i in range(4)],
    # *[oracle_dense_variant(result_name=f"oracle_dense_o20_n10_{i}", 
    #                        num_objects=20, num_nearest=10, 
    #                        gpu=5, gpu_percent=0.1) for i in range(4)],
    # *[oracle_sparse_variant(result_name=f"oracle_sparse_o20_n10_{i}", 
    #                        num_objects=20, num_nearest=10, 
    #                        gpu=6, gpu_percent=0.1) for i in range(4)],
    
    # *[image_dense_variant(result_name=f"image_dense_o20_{i}", 
    #                        num_objects=20, 
    #                        gpu=gpu, gpu_percent=0.45) for i, gpu in ((3, 3), (4, 3), (5, 4))],
    # *[image_sparse_variant(result_name=f"image_sparse_o20_{i}", 
    #                        num_objects=20, 
    #                        gpu=gpu, gpu_percent=0.45) for i, gpu in ((3, 4), (4, 5), (5, 5))],
    # *[oracle_dense_variant(result_name=f"oracle_dense_o20_n3_{i}", 
    #                        num_objects=20, num_nearest=3, 
    #                        gpu=6, gpu_percent=0.1) for i in range(4, 7)],
    # *[oracle_sparse_variant(result_name=f"oracle_sparse_o20_n3_{i}", 
    #                        num_objects=20, num_nearest=3, 
    #                        gpu=7, gpu_percent=0.1) for i in range(4, 7)],

    # *[frame_stack_dense_variant(result_name=f"frame_stack_dense_o20_f4_d95_{i}", 
    #                        num_objects=20,
    #                        gpu=gpu, gpu_percent=0.45) for i, gpu in ((0, 0), (1, 0), (2, 1))],
    # *[frame_stack_sparse_variant(result_name=f"frame_stack_sparse_o20_f4_d95_{i}", 
    #                        num_objects=20,
    #                        gpu=gpu, gpu_percent=0.45) for i, gpu in ((0, 1), (1, 2), (2, 2))],

    # *[oracle_dense_variant(result_name=f"oracle_dense_o20_n20_ns_{i}", 
    #                        num_objects=20, num_nearest=20, do_sort=False, 
    #                        gpu=3, gpu_percent=0.1) for i in range(0, 3)],
    # *[oracle_sparse_variant(result_name=f"oracle_sparse_o20_n20_ns_{i}", 
    #                        num_objects=20, num_nearest=20, do_sort=False,
    #                        gpu=4, gpu_percent=0.1) for i in range(0, 3)],

    # *[oracle_single_variant(result_name=f"oracle_single_o20_n1_zo_{i}", 
    #                        num_objects=20, num_nearest=1, mode="zero_one",
    #                        gpu=3, gpu_percent=0.1) for i in range(3)],
    # *[oracle_single_variant(result_name=f"oracle_single_o20_n1_nz_{i}", 
    #                        num_objects=20, num_nearest=1, mode="neg_zero",
    #                        gpu=4, gpu_percent=0.1) for i in range(3)],
    # *[frame_stack_single_variant(result_name=f"frame_stack_single_o20_f4_zo_{i}", 
    #                        num_objects=20, mode="zero_one",
    #                        gpu=gpu, gpu_percent=0.45) for i, gpu in ((0, 5), (1, 5), (2, 6))],
    
    # *[frame_stack_single_variant(result_name=f"frame_stack_single_o20_f4_nz_v4_{i}", 
    #                        num_objects=20, mode="neg_zero",
    #                        gpu=gpu, gpu_percent=0.45) for i, gpu in ((0, 3), (1, 3), (2, 4))],
    # *[image_single_variant(result_name=f"image_single_o20_nz_v4_{i}", 
    #                        num_objects=20, mode="neg_zero",
    #                        gpu=gpu, gpu_percent=0.45) for i, gpu in ((0, 4), (1, 5), (2, 5))],

    # *[frame_stack_single_variant(result_name=f"frame_stack_single_o20_f4_nz_v5_{i}", 
    #                        num_objects=20, mode="neg_zero",
    #                        gpu=gpu, gpu_percent=0.45) for i, gpu in ((0, 3), (1, 4))],
    # *[image_single_variant(result_name=f"image_single_o20_nz_v4_{i}",
    #                        num_objects=20, mode="neg_zero",
    #                        gpu=gpu, gpu_percent=0.45) for i, gpu in ((0, 7), (1, 8))],
    # *[image_single_variant(result_name=f"image_single_o20_nz_v4_tr4_{i}",
    #                        num_objects=20, mode="neg_zero",
    #                        gpu=gpu, gpu_percent=0.45) for i, gpu in [(0, 0), (1, 1), (2, 2)]],

    # *[image_single_variant(result_name=f"image_single_o20_nz_v12_original_{i}",
    #                        num_objects=20, mode="neg_zero",
    #                        room_name="double_v2",
    #                        object_name="whitesquareball",
    #                        gpu=gpu, gpu_percent=0.45, server_port=server_port) 
    #                        for i, gpu, server_port in [
    #                            (0, 5, 10001), 
    #                            (1, 6, 10002),
    #                        ]],


    # *[oracle_single_soft_q_variant(result_name=f"oracle_single_soft_q_v2_o20_n1_rel_cull_{i}", 
    #                         num_objects=20, num_nearest=1, do_cull=True, is_relative=True,
    #                         gpu=gpu, server_port=server_port, gpu_percent=0.45) 
    #                         for i, gpu, server_port in [
    #                            (0, 0, 10001), 
    #                            (1, 0, 10002), 
    #                             # (2, 2, 10003),
    #                         ]],
    # *[image_single_soft_q_variant(result_name=f"image_single_soft_q_v2_load_o20_v12_{i}", 
    #                         num_objects=20, grasp_data_name="grasp_s500_f500",
    #                         gpu=gpu, server_port=server_port, gpu_percent=0.45) 
    #                         for i, gpu, server_port in [
    #                            (0, 1, 10005), 
    #                            (1, 2, 10006), 
    #                             # (2, 2, 10003),
    #                         ]],
    
    # *[image_single_soft_q_variant(result_name=f"image_single_soft_q_v2_trained_o20_v12_{i}", 
    #                         num_objects=20, grasp_model_name="alpha10min_6Q_stat_stat",
    #                         gpu=gpu, server_port=server_port, gpu_percent=0.45) 
    #                         for i, gpu, server_port in [
    #                            (0, 0, 10007),
    #                            (1, 1, 10008),
    #                         ]],
    # *[image_single_soft_q_variant(result_name=f"image_single_soft_q_v2_trained_load_o20_v12_{i}", 
    #                         num_objects=20, grasp_data_name="grasp_s500_f500", grasp_model_name="alpha10min_6Q_stat_stat",
    #                         gpu=gpu, server_port=server_port, gpu_percent=0.45) 
    #                         for i, gpu, server_port in [
    #                            (0, 3, 10009),
    #                            (1, 4, 10010),
    #                         ]],

    # *[image_single_soft_q_variant(result_name=f"image_single_soft_q_v2_load_o20_v12_bonus_{i}", 
    #                         num_objects=20, grasp_data_name="grasp_s500_f500", add_uncertainty_bonus=True,
    #                         gpu=gpu, server_port=server_port, gpu_percent=0.45) 
    #                         for i, gpu, server_port in [
    #                            (0, 1, 10016), 
    #                            (1, 2, 10017), 
    #                         ]],
    # *[oracle_single_soft_q_noautograsp_variant(result_name=f"oracle_single_soft_q_v2_trained_load_o20_n1_rel_cull_nag3_{i}", 
    #                         num_objects=20, num_nearest=1, do_cull=True, is_relative=True,
    #                         grasp_data_name="grasp_s500_f500", grasp_model_name="alpha10min_6Q_stat_stat",
    #                         gpu=gpu, server_port=server_port, gpu_percent=0.45) 
    #                         for i, gpu, server_port in [
    #                            (0, 0, 10011), 
    #                            (1, 0, 10012), 
    #                         ]],
    # *[image_single_soft_q_noautograsp_variant(result_name=f"image_single_soft_q_v2_trained_load_o20_v12_nag3_{i}", 
    #                         num_objects=20, grasp_data_name="grasp_s500_f500", grasp_model_name="alpha10min_6Q_stat_stat",
    #                         gpu=gpu, server_port=server_port, gpu_percent=0.45) 
    #                         for i, gpu, server_port in [
    #                            (0, 1, 10009),
    #                            (1, 2, 10010),
    #                         ]],
    # *[oracle_single_soft_q_noautograsp_variant(result_name=f"oracle_single_soft_q_v2_trained_load_o20_n1_rel_cull_nag3_bonus_{i}", 
    #                         num_objects=20, num_nearest=1, do_cull=True, is_relative=True,
    #                         grasp_data_name="grasp_s500_f500", grasp_model_name="alpha10min_6Q_stat_stat",
    #                         add_uncertainty_bonus=True,
    #                         gpu=gpu, server_port=server_port, gpu_percent=0.45) 
    #                         for i, gpu, server_port in [
    #                            (0, 0, 10018), 
    #                            (1, 0, 10019), 
    #                         ]],
    # *[image_single_soft_q_noautograsp_variant(result_name=f"image_single_soft_q_v2_trained_load_o20_v12_nag3_bonus_{i}", 
    #                         num_objects=20, grasp_data_name="grasp_s500_f500", grasp_model_name="alpha10min_6Q_stat_stat",
    #                         add_uncertainty_bonus=True,
    #                         gpu=gpu, server_port=server_port, gpu_percent=0.45) 
    #                         for i, gpu, server_port in [
    #                            (0, 3, 10013),
    #                            (1, 4, 10014),
    #                         ]],
    # *[image_single_soft_q_noautograsp_variant(result_name=f"image_single_soft_q_v2_load_o20_v12_nag3_bonus_{i}", 
    #                         num_objects=20, grasp_data_name="grasp_s500_f500",
    #                         add_uncertainty_bonus=True,
    #                         gpu=gpu, server_port=server_port, gpu_percent=0.45) 
    #                         for i, gpu, server_port in [
    #                            (0, 9, 10023),
    #                            (1, 8, 10022),
    #                         ]],

    # *[image_single_soft_q_noautograsp_variant(result_name=f"image_single_soft_q_v2_trained_load_o20_v12_nag4_bonus_respawn_saveeverything_{i}", 
    #                         num_objects=20,
    #                         add_uncertainty_bonus=True,
    #                         gpu=gpu, server_port=server_port, gpu_percent=0.45) 
    #                         for i, gpu, server_port in [
    #                            (0, 7, 10041),
    #                         ]],

    # *[oracle_single_soft_q_noautograsp_uniform_variant(result_name=f"oracle_single_soft_q_v2_trained_load_o20_n1_rel_cull_nag3_uniform_nr_{i}", 
    #                         num_objects=20, num_nearest=1, do_cull=True, is_relative=True,
    #                         grasp_data_name="grasp_s500_f500", grasp_model_name="alpha10min_6Q_stat_stat",
    #                         eval_pert="no_respawn",
    #                         gpu=gpu, server_port=server_port, gpu_percent=0.45) 
    #                         for i, gpu, server_port in [
    #                            (0, 5, 10004), 
    #                            (1, 5, 10005), 
    #                            (2, 5, 10006), 
    #                         ]],

    # *[image_single_soft_q_noautograsp_uniform_variant(result_name=f"image_single_soft_q_v2_trained_load_o20_v12_nag4_bonus_uniform_250eplen_2grasp_{i}", 
    #                         num_objects=20,
    #                         grasp_data_name="grasp_s500_f500",
    #                         grasp_model_name="alpha10min_6Q_stat_stat",
    #                         add_uncertainty_bonus=True,
    #                         gpu=gpu, server_port=server_port, gpu_percent=0.20) 
    #                         for i, gpu, server_port in [
    #                            (0, 0, 10010), 
    #                            (1, 1, 10011), 
    #                            (2, 2, 10012), 
    #                         ]],

    # *[image_single_soft_q_noautograsp_uniform_variant(result_name=f"image_single_soft_q_v2_trained_load_o20_v12_nag4_bonus_uniform_{i}", 
    #                         num_objects=20,
    #                         grasp_data_name="grasp_s500_f500", 
    #                         grasp_model_name="alpha10min_6Q_stat_stat",
    #                         add_uncertainty_bonus=True,
    #                         gpu=gpu, server_port=server_port, gpu_percent=0.20) 
    #                         for i, gpu, server_port in [
    #                            (0, 7, 10020), 
    #                            (1, 8, 10021), 
    #                         #    (2, 7, 10009), 
    #                         ]],

    # *[image_single_soft_q_noautograsp_none_uniform_variant(result_name=f"image_single_soft_q_v2_trained_load_o20_v12_nag4_bonus_none_uniform_{i}", 
    #                         num_objects=20,
    #                         grasp_data_name="grasp_s500_f500", 
    #                         grasp_model_name="alpha10min_6Q_stat_stat",
    #                         add_uncertainty_bonus=True,
    #                         gpu=gpu, server_port=server_port, gpu_percent=0.20) 
    #                         for i, gpu, server_port in [
    #                            (0, 3, 10001), 
    #                            (1, 4, 10002), 
    #                         ]],

    # *[image_single_soft_q_noautograsp_uncertainty_navQ_variant(result_name=f"image_single_soft_q_v2_trained_load_o20_v12_nag4_bonus_uncertainty_navQ_normalize_{i}", 
    #                         num_objects=20,
    #                         grasp_data_name="grasp_s500_f500", 
    #                         grasp_model_name="alpha10min_6Q_stat_stat",
    #                         add_uncertainty_bonus=True,
    #                         gpu=gpu, server_port=server_port, gpu_percent=0.20) 
    #                         for i, gpu, server_port in [
    #                            (0, 3, 10031), 
    #                            (1, 4, 10032), 
    #                         ]],

    # *[image_single_soft_q_noautograsp_uncertainty_uniform_variant(result_name=f"image_single_soft_q_v2_trained_load_o20_v12_nag4_bonus_uncertainty_uniform_{i}", 
    #                         num_objects=20,
    #                         grasp_data_name="grasp_s500_f500", 
    #                         grasp_model_name="alpha10min_6Q_stat_stat",
    #                         add_uncertainty_bonus=True,
    #                         gpu=gpu, server_port=server_port, gpu_percent=0.20) 
    #                         for i, gpu, server_port in [
    #                            (0, 1, 10017), 
    #                            (1, 2, 10018), 
    #                         #    (2, 7, 10009), 
    #                         ]],

    # *[image_single_soft_q_noautograsp_uniform_variant(result_name=f"image_single_soft_q_v2_trained_load_double_v2_o20_v12_nag4_bonus_uniform_{i}", 
    #                         num_objects=20,
    #                         grasp_data_name="grasp_s500_f500", 
    #                         grasp_model_name="alpha10min_6Q_stat_stat",
    #                         add_uncertainty_bonus=True,
    #                         room_name="double_v2",
    #                         gpu=gpu, server_port=server_port, gpu_percent=0.20) 
    #                         for i, gpu, server_port in [
    #                            (0, 9, 10001), 
    #                            (1, 8, 10002), 
    #                            (2, 7, 10003), 
    #                         ]],
    
    # *[image_single_soft_q_noautograsp_variant(result_name=f"image_single_soft_q_v2_trained2_o20_color_v12_nag4_bonus_{i}", 
    #                         num_objects=20, grasp_model_name="alpha10mean_beta10std_stat_rand_color",
    #                         add_uncertainty_bonus=True,
    #                         gpu=gpu, server_port=server_port, gpu_percent=0.45) 
    #                         for i, gpu, server_port in [
    #                            (0, 0, 10001),
    #                            (1, 1, 10002),
    #                            (2, 2, 10003),
    #                         ]],
    # *[image_single_soft_q_noautograsp_uniform_variant(result_name=f"image_single_soft_q_v2_trained2_o20_color_v12_nag4_bonus_uniform_{i}", 
    #                         num_objects=20, grasp_model_name="alpha10mean_beta10std_stat_rand_color",
    #                         add_uncertainty_bonus=True,
    #                         gpu=gpu, server_port=server_port, gpu_percent=0.20) 
    #                         for i, gpu, server_port in [
    #                            (0, 3, 10004), 
    #                            (1, 4, 10005), 
    #                            (2, 5, 10006), 
    #                         ]],
    # *[image_single_variant(result_name=f"image_single_o20_color_wood_nz_v12_{i}",
    #                        num_objects=20, mode="neg_zero",
    #                        gpu=gpu, server_port=server_port, gpu_percent=0.20) 
    #                        for i, gpu, server_port in [
    #                            (0, 3, 10010), 
    #                            (1, 4, 10011), 
    #                            (2, 5, 10012), 
    #                         ]],


    # Final Runs:

    # *[image_single_soft_q_noautograsp_uniform_variant(result_name=f"image_soft_q_2grasp_trained_load_o20_v12_nagmeanstd_bonus_forward_uniform_250eplen_{i}", 
    #                         num_objects=20,
    #                         grasp_data_name="grasp_s500_f500",
    #                         grasp_model_name="alpha10min_6Q_stat_stat",
    #                         add_uncertainty_bonus=True,
    #                         gpu=gpu, server_port=server_port, gpu_percent=0.20) 
    #                         for i, gpu, server_port in [
    #                            (0, 0, 10010), 
    #                            (1, 1, 10011), 
    #                            (2, 2, 10012), 
    #                         ]],

    # *[image_single_soft_q_noautograsp_uniform_variant(result_name=f"image_soft_q_2grasp_trained_load_o20_v12_nagmeanstd_nobonus_forward_uniform_250eplen_{i}", 
    #                         num_objects=20,
    #                         grasp_data_name="grasp_s500_f500",
    #                         grasp_model_name="alpha10min_6Q_stat_stat",
    #                         add_uncertainty_bonus=False,
    #                         gpu=gpu, server_port=server_port, gpu_percent=0.20) 
    #                         for i, gpu, server_port in [
    #                            (0, 3, 10013), 
    #                            (1, 4, 10014), 
    #                            (2, 5, 10015), 
    #                         ]],

    *[image_single_soft_q_noautograsp_uniform_variant(result_name=f"image_soft_q_2grasp_trained_load_o20_v12_nagmean_bonus_forward_uniform_250eplen_{i}", 
                            num_objects=20,
                            grasp_data_name="grasp_s500_f500",
                            grasp_model_name="alpha10min_6Q_stat_stat",
                            add_uncertainty_bonus=True,
                            gpu=gpu, server_port=server_port, gpu_percent=0.20) 
                            for i, gpu, server_port in [
                               (0, 6, 10016), 
                               (1, 7, 10017), 
                               (2, 8, 10018), 
                            ]],

]

def run_exp(variant):
    # Use docker mode to launch jobs on cthulhu machine
    mode_ssh = dd.mode.SSHSingularity(
        image='/home/charlesjsun/mobilemanipulation-singularity-gpu.sif',
        credentials=ssh.SSHCredentials(hostname='cthulhu2.ist.berkeley.edu',
                                    username='charlesjsun', identity_file='~/.ssh/id_rsa'),
        name_prefix=variant["result_name"] + "_",
        gpu=True,
        tmp_dir="/home/charlesjsun/.remote_tmp"
    )

    # Set up code and output directories
    mounts = [
        mount.MountLocal(local_dir='.', mount_point='/home/charlesjsun/mobilemanipulation'),
        mount.MountLocal(local_dir='~/.mujoco', mount_point='/home/charlesjsun/.mujoco'),
        mount.MountLocal(local_dir='/home/charlesjsun/ray_results', mount_point='/home/charlesjsun/ray_results', output=True),
    ]

    if RENDERS or TELEOP or RENDERS_EVAL:
        variant["gpu_percent"] = 0.0

    command = get_command(**variant)

    if RENDERS or TELEOP or RENDERS_EVAL:
        print(command.split(";")[-1])
        input()
    else:
        dd.launch_shell(
            command=command,
            mode=mode_ssh,
            mount_points=mounts,
            verbose=True
        )
    print()


if RENDERS or TELEOP or RENDERS_EVAL:
    run_exp(variants[0])
else:
    for variant in variants:
        run_exp(variant)

print("DONE!")
