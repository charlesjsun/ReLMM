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
    gpu=0,
    gpu_percent=0.45,
    env_kwargs={},
    eval_env_kwargs={}):
    
    def process_kwargs(kwargs):
        json_str = json.dumps(kwargs)
        processed = json_str.replace("\"", "\\quote")
        return processed

    cmds = []
    cmds.append("source activate softlearning")
    cmds.append("cd /root/softlearning")
    cmds.append("pip install -e .")
    cmds.append("export RAY_MEMORY_MONITOR_ERROR_THRESHOLD=1.000001")
    cmds.append(f"export CUDA_VISIBLE_DEVICES={gpu}")
    cmds.append(
f'softlearning run_example_local examples.dual_perturbation \
--algorithm SAC \
--policy gaussian \
--universe gym \
--domain Locobot \
--task {task} \
--exp-name {exp_name} \
--result-name {result_name} \
--checkpoint-frequency 10 \
--trial-cpus 1 \
--trial-gpus {gpu_percent} \
--max-failures 0 \
--run-eagerly False \
--server-port 11001 \
--env-kwargs "{process_kwargs(env_kwargs)}" \
--eval-env-kwargs "{process_kwargs(eval_env_kwargs)}"')
    
    full_cmd = ";".join(cmds)

    return full_cmd

def base_variant(num_objects, result_name, gpu, gpu_percent):
    return {
        "result_name": result_name,
        "gpu": gpu,
        "gpu_percent": gpu_percent,
        "env_kwargs": {
            "grasp_perturbation": "none",
            "nav_perturbation": "none",
            "grasp_algorithm": "vacuum",
            "use_auto_grasp": True,
            "renders": RENDERS,
            "do_teleop": TELEOP,
            "step_duration": 0.0,
            "room_name": "single",
            "room_params": {
                "num_objects": num_objects,
            }
        },
        "eval_env_kwargs": {
            "renders": False,
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

def oracle_single_soft_q_variant(**kwargs):
    return deep_update(oracle_single_variant(mode="neg_zero", **kwargs), {
        "exp_name": "oracle-single-soft-q-respawn",
        "env_kwargs": {
            "grasp_algorithm": "soft_q",
        },
        "eval_env_kwargs": {
            "do_grasp_eval": True,
        }
    })


def oracle_single_soft_q_noauto_variant(**kwargs):
    return deep_update(oracle_single_soft_q_variant(**kwargs), {
        "exp_name": "oracle-single-soft-q-noauto-respawn",
        "env_kwargs": {
            "use_auto_grasp": False,
        },
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

def image_single_soft_q_variant(**kwargs):
    return deep_update(image_single_variant(mode="neg_zero", **kwargs), {
        "exp_name": "image-single-soft-q-respawn",
        "env_kwargs": {
            "grasp_algorithm": "soft_q",
        },
        "eval_env_kwargs": {
            "do_grasp_eval": True,
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


LOCAL = False
RENDERS = False
TELEOP = False

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
    #                        gpu=gpu, gpu_percent=0.45) for i, gpu in [(0, 5), (1, 6)]],

    *[image_single_soft_q_variant(result_name=f"image_single_soft_q_v2_o20_v12_{i}", 
                           num_objects=20, 
                           gpu=gpu, gpu_percent=0.45) for i, gpu in ((0, 0), (1, 1), (2, 2))],

    # *[image_single_variant(result_name=f"image_single_o20_nz_v8_img64_{i}",
    #                        num_objects=20, mode="neg_zero",
    #                        gpu=gpu, gpu_percent=0.45) for i, gpu in [(0, 0), (1, 1), (2, 2)]],
    
    # *[oracle_single_variant(result_name=f"oracle_single_o20_n1_nz_cull_v2_rel_{i}",
    #                        num_objects=20, num_nearest=1, mode="neg_zero", do_cull=True, is_relative=True,
    #                        gpu=5, gpu_percent=0.1) for i in range(3)],
    # *[oracle_single_variant(result_name=f"oracle_single_o20_n1_nz_rel_{i}", 
    #                        num_objects=20, num_nearest=1, mode="neg_zero", do_cull=False, is_relative=True,
    #                        gpu=4, gpu_percent=0.1) for i in range(3)],

    # *[oracle_single_soft_q_variant(result_name=f"oracle_single_soft_q_l_o20_n1_rel_{i}", 
    #                        num_objects=20, num_nearest=1, do_cull=False, is_relative=True,
                        #    gpu=gpu, gpu_percent=0.45) for i, gpu in ((0, 0), (1, 0), (2, 0))],
    # *[oracle_single_soft_q_noauto_variant(result_name=f"oracle_single_soft_q_na_o20_n1_rel_{i}", 
    #                        num_objects=20, num_nearest=1, do_cull=False, is_relative=True,
    #                        gpu=gpu, gpu_percent=0.45) for i, gpu in ((0, 1), (1, 1), (2, 1))],
    

]

def run_exp(variant):
    # Use local mode to test code
    mode_local = dd.mode.LocalDocker(
        image='charlesjsun/mobilemanipulation-cpu',
        name_prefix=variant["result_name"] + "_",
    )

    # Use docker mode to launch jobs on newton machine
    mode_ssh = dd.mode.SSHDocker(
        image='charlesjsun/mobilemanipulation-gpu',
        credentials=ssh.SSHCredentials(hostname='cthulhu2.ist.berkeley.edu',
                                    username='charlesjsun', identity_file='~/.ssh/id_rsa'),
        detach=True,
        name_prefix=variant["result_name"] + "_",
        gpu=True
    )

    # Set up code and output directories
    mounts = [
        mount.MountLocal(local_dir='.', mount_point='/root/softlearning'),
        mount.MountLocal(local_dir='~/.mujoco/mjkey.txt', mount_point='/root/.mujoco/mjkey.txt'),
        mount.MountLocal(local_dir='~/ray_results', mount_point='/root/ray_results', output=True),
    ]

    if LOCAL or RENDERS or TELEOP:
        variant["gpu_percent"] = 0.0

    command = get_command(**variant)
    
    if RENDERS or TELEOP:
        print(command.split(";")[-1])
        input()
    else:
        dd.launch_shell(
            command=command,
            mode=mode_local if LOCAL else mode_ssh,
            mount_points=mounts,
            verbose=True
        )
    print()


if RENDERS or TELEOP:
    run_exp(variants[0])
elif LOCAL:
    for variant in variants:
        run_exp(variant)
else:
    for variant in variants:
        run_exp(variant)
    # pool = multiprocessing.Pool(NUM_THREADS)
    # pool.map(run_exp, variants)

print("DONE!")
