"""
Call in the root directory (i.e. ../ from the directory this file is in)
"""

try:
    import doodad as dd
    import doodad.ssh as ssh
    import doodad.mount as mount
except ImportError as e:
    print("doodad not installed.")
    dd = None

import json
import multiprocessing

import numpy as np

from softlearning.utils.dict import deep_update

def get_command(
    username='charlesjsun',
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
    cmds.append(f"cd /home/{username}/mobilemanipulation")
    cmds.append("export RAY_MEMORY_MONITOR_ERROR_THRESHOLD=1.000001")
    cmds.append(f"export CUDA_VISIBLE_DEVICES={gpu}")
    cmds.append(f"export PYTHONPATH=/home/{username}/mobilemanipulation")
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
                ):
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
                "object_name": object_name,
            }
        },
        "eval_env_kwargs": {
            "renders": RENDERS_EVAL,
            "step_duration": 0.0,
            "do_teleop": False,
            "room_params": {
                "num_objects": num_objects,
                "object_name": object_name,
            }
        }
    }

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

def image_single_soft_q_variant(grasp_data_name=None, grasp_model_name=None, add_uncertainty_bonus=False, fixed_policy=False, **kwargs):
    return deep_update(image_single_variant(mode="neg_zero", **kwargs), {
        "exp_name": "image-single-soft-q-respawn",
        "env_kwargs": {
            "grasp_algorithm": "soft_q",
            "grasp_algorithm_params": {
                "grasp_data_name": grasp_data_name,
                "grasp_model_name": grasp_model_name,
                "fixed_policy": fixed_policy,
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

def image_single_soft_q_noautograsp_none_none_variant(**kwargs):
    return deep_update(image_single_soft_q_noautograsp_variant(**kwargs), {
        "exp_name": "image-single-soft-q-noauto-none-respawn",
        "env_kwargs": {
            "grasp_perturbation": "none",
            "grasp_perturbation_params": {
                "num_steps": 20,
            },
            "nav_perturbation": "none",
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

def image_single_soft_q_noautograsp_rnd_rnd_variant(**kwargs):
    return deep_update(image_single_soft_q_noautograsp_variant(**kwargs), {
        "exp_name": "image-single-soft-q-noauto-rnd-respawn",
        "env_kwargs": {
            "grasp_perturbation": "rnd",
            "grasp_perturbation_params": {
                "num_steps": 20,
            },
            "nav_perturbation": "rnd",
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

def image_single_soft_q_noautograsp_uniform_attenuate_variant(**kwargs):
    return deep_update(image_single_soft_q_noautograsp_variant(num_objects=20, **kwargs), {
        "exp_name": "image-single-soft-q-noauto-uniform-attenuate",
        "env_kwargs": {
            "grasp_perturbation": "random_uniform",
            "grasp_perturbation_params": {
                "num_steps": 20,
            },
            "nav_perturbation": "random_uniform",
            "nav_perturbation_params": {
                "num_steps": 20,
            },
            "room_params": {
                "num_objects": 40,
                "no_spawn_radius": 0.3,
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
            "room_params": {
                "num_objects": 20,
                "no_spawn_radius": 0.5,
            },
        }
    })




RENDERS = True
TELEOP = True
RENDERS_EVAL = False

variants = [
    *[image_single_soft_q_noautograsp_uniform_variant(result_name=f"image_soft_q_2grasp_trained_load_o20_v12_nagmeanstd_bonus_forward_uniform_250eplen_{i}", 
                            num_objects=20,
                            grasp_data_name="grasp_s500_f500",
                            grasp_model_name="alpha10min_6Q_stat_stat",
                            add_uncertainty_bonus=True,
                            gpu=gpu, server_port=server_port, gpu_percent=0.20) 
                            for i, gpu, server_port in [
                               (0, 0, 10010), 
                               (1, 1, 10011), 
                               (2, 2, 10012), 
                            ]],

    *[image_single_soft_q_noautograsp_uniform_variant(result_name=f"image_soft_q_2grasp_trained_load_o20_v12_nagmeanstd_nobonus_forward_uniform_250eplen_{i}", 
                            num_objects=20,
                            grasp_data_name="grasp_s500_f500",
                            grasp_model_name="alpha10min_6Q_stat_stat",
                            add_uncertainty_bonus=False,
                            gpu=gpu, server_port=server_port, gpu_percent=0.20) 
                            for i, gpu, server_port in [
                               (0, 3, 10013), 
                               (1, 4, 10014), 
                               (2, 5, 10015), 
                            ]],

    *[image_single_soft_q_noautograsp_uniform_variant(result_name=f"image_soft_q_2grasp_trained_load_o20_v12_nagmean_nobonus_forward_uniform_250eplen_{i}", 
                            num_objects=20,
                            grasp_data_name="grasp_s500_f500",
                            grasp_model_name="alpha10min_6Q_stat_stat",
                            add_uncertainty_bonus=False,
                            gpu=gpu, server_port=server_port, gpu_percent=0.20) 
                            for i, gpu, server_port in [
                               (0, 0, 10016), 
                               (1, 1, 10017), 
                               (2, 1, 10071), 
                            ]],

    *[image_single_soft_q_noautograsp_none_none_variant(result_name=f"image_soft_q_2grasp_trained_load_o20_v12_nagmeanstd_nobonus_forward_none_none_250eplen_{i}", 
                            num_objects=20,
                            grasp_data_name="grasp_s500_f500",
                            grasp_model_name="alpha10min_6Q_stat_stat",
                            add_uncertainty_bonus=False,
                            gpu=gpu, server_port=server_port, gpu_percent=0.20) 
                            for i, gpu, server_port in [
                               (0, 6, 10011), 
                               (1, 7, 10012), 
                            ]],

    *[image_single_soft_q_noautograsp_rnd_rnd_variant(result_name=f"image_soft_q_2grasp_trained_load_o20_v12_nagmeanstd_nobonus_forward_rnd_rnd_250eplen_{i}", 
                            num_objects=20,
                            grasp_data_name="grasp_s500_f500",
                            grasp_model_name="alpha10min_6Q_stat_stat",
                            add_uncertainty_bonus=False,
                            gpu=gpu, server_port=server_port, gpu_percent=0.20) 
                            for i, gpu, server_port in [
                               (0, 8, 10013), 
                               (1, 9, 10014), 
                            ]],

    *[image_single_soft_q_noautograsp_uniform_variant(result_name=f"image_soft_q_2grasp_n0_o20_v12_nagmeanstd_nobonus_forward_uniform_250eplen_{i}", 
                            num_objects=20,
                            # grasp_data_name="n500", grasp_model_name="n500",
                            add_uncertainty_bonus=False,
                            gpu=gpu, server_port=server_port, gpu_percent=0.20) 
                            for i, gpu, server_port in [
                               (0, 0, 10018), 
                               (1, 1, 10019), 
                            ]],
    *[image_single_soft_q_noautograsp_uniform_variant(result_name=f"image_soft_q_2grasp_n500_o20_v12_nagmeanstd_nobonus_forward_uniform_250eplen_100K_{i}", 
                            num_objects=20,
                            grasp_data_name="n500", grasp_model_name="n500",
                            add_uncertainty_bonus=False,
                            gpu=gpu, server_port=server_port, gpu_percent=0.20) 
                            for i, gpu, server_port in [
                               (0, 4, 10005), 
                               (1, 5, 10006), 
                            ]],
    *[image_single_soft_q_noautograsp_uniform_variant(result_name=f"image_soft_q_2grasp_n1000_o20_v12_nagmeanstd_nobonus_forward_uniform_250eplen_{i}", 
                            num_objects=20,
                            grasp_data_name="n1000", grasp_model_name="n1000",
                            add_uncertainty_bonus=False,
                            gpu=gpu, server_port=server_port, gpu_percent=0.20) 
                            for i, gpu, server_port in [
                               (0, 0, 10010), 
                               (1, 1, 10011), 
                            ]],
    *[image_single_soft_q_noautograsp_uniform_variant(result_name=f"image_soft_q_2grasp_n1250_o20_v12_nagmeanstd_nobonus_forward_uniform_250eplen_{i}", 
                            num_objects=20,
                            grasp_data_name="n1250", grasp_model_name="n1250",
                            add_uncertainty_bonus=False,
                            gpu=gpu, server_port=server_port, gpu_percent=0.20) 
                            for i, gpu, server_port in [
                               (0, 0, 10001), 
                               (1, 1, 10002), 
                            ]],
    *[image_single_soft_q_noautograsp_uniform_variant(result_name=f"image_soft_q_2grasp_n1500_o20_v12_nagmeanstd_nobonus_forward_uniform_250eplen_{i}", 
                            num_objects=20,
                            grasp_data_name="n1500", grasp_model_name="n1500",
                            add_uncertainty_bonus=False,
                            gpu=gpu, server_port=server_port, gpu_percent=0.20) 
                            for i, gpu, server_port in [
                               (0, 2, 10012), 
                               (1, 3, 10013), 
                            ]],
    *[image_single_soft_q_noautograsp_uniform_variant(result_name=f"image_soft_q_2grasp_n2000_o20_v12_nagmeanstd_nobonus_forward_uniform_250eplen_{i}", 
                            num_objects=20,
                            grasp_data_name="n2000", grasp_model_name="n2000",
                            add_uncertainty_bonus=False,
                            gpu=gpu, server_port=server_port, gpu_percent=0.20) 
                            for i, gpu, server_port in [
                               (0, 4, 10014), 
                               (1, 5, 10015), 
                            ]],
    *[image_single_soft_q_noautograsp_uniform_variant(result_name=f"image_soft_q_2grasp_n4000_o20_v12_nagmeanstd_nobonus_forward_uniform_250eplen_{i}", 
                            num_objects=20,
                            grasp_data_name="n4000", grasp_model_name="n4000",
                            add_uncertainty_bonus=False,
                            gpu=gpu, server_port=server_port, gpu_percent=0.20) 
                            for i, gpu, server_port in [
                               (0, 6, 10016), 
                               (1, 7, 10017), 
                            ]],
    
    *[image_single_soft_q_noautograsp_uniform_attenuate_variant(result_name=f"image_soft_q_10t2grasp1000t1000_n0_o40_v12_nagmeanstd_nobonus_forward_0t20uniform1000t2000_250eplen_{i}", 
                            add_uncertainty_bonus=False,
                            gpu=gpu, server_port=server_port, gpu_percent=0.20) 
                            for i, gpu, server_port in [
                               (0, 4, 10003), 
                               (1, 5, 10004), 
                            ]],

    *[image_single_soft_q_noautograsp_uniform_variant(result_name=f"image_soft_q_2grasp_trained_load_robstacles_o30_v12_nagmeanstd_nobonus_forward_uniform_250eplen_{i}", 
                            num_objects=30, room_name="obstacles",
                            grasp_data_name="grasp_s500_f500",
                            grasp_model_name="alpha10min_6Q_stat_stat",
                            add_uncertainty_bonus=False,
                            gpu=gpu, server_port=server_port, gpu_percent=0.20) 
                            for i, gpu, server_port in [
                               (0, 0, 10081), 
                               (1, 1, 10082), 
                            ]],

    *[image_single_soft_q_noautograsp_uniform_variant(result_name=f"image_soft_q_2grasp_trained_load_o20_v12_nagmeanstd_nobonus_forward_back_uniform_250eplen_{i}", 
                            num_objects=20,
                            grasp_data_name="grasp_s500_f500",
                            grasp_model_name="alpha10min_6Q_stat_stat",
                            add_uncertainty_bonus=False,
                            gpu=gpu, server_port=server_port, gpu_percent=0.20) 
                            for i, gpu, server_port in [
                               (0, 6, 10073), 
                               (1, 7, 10074), 
                            ]],

    *[image_single_soft_q_noautograsp_uniform_variant(result_name=f"image_soft_q_2grasp_n1250_fixed_o20_v12_nagmeanstd_nobonus_forward_uniform_250eplen_{i}", 
                            num_objects=20,
                            grasp_data_name=None, grasp_model_name="n1250",
                            add_uncertainty_bonus=False, fixed_policy=True,
                            gpu=gpu, server_port=server_port, gpu_percent=0.20) 
                            for i, gpu, server_port in [
                               (0, 2, 10003), 
                               (1, 3, 10004), 
                            ]],
]

def run_exp(variant):
    if dd is None:
        command = get_command(username="username", **variant)
        print(command.split(";")[-1])
        return

    # Use singularity mode to launch jobs on ssh machine

    username = 'charlesjsun'
    hostname = 'cthulhu2.ist.berkeley.edu'
    singularity_bin = 'singularity'

    mode_ssh = dd.mode.SSHSingularity(
        image=f'/home/{username}/mobilemanipulation-singularity-gpu.sif',
        credentials=ssh.SSHCredentials(hostname=hostname,
                                    username=username, identity_file='~/.ssh/id_rsa'),
        name_prefix=variant["result_name"] + "_",
        gpu=True,
        tmp_dir=f"/home/{username}/.remote_tmp",
        singularity_bin=singularity_bin
    )

    # Set up code and output directories
    mounts = [
        mount.MountLocal(local_dir='.', mount_point=f'/home/{username}/mobilemanipulation'),
        mount.MountLocal(local_dir='~/.mujoco', mount_point=f'/home/{username}/.mujoco'),
        mount.MountLocal(local_dir='~/ray_results', mount_point=f'/home/{username}/ray_results', output=True),
    ]

    if RENDERS or TELEOP or RENDERS_EVAL:
        variant["gpu_percent"] = 0.0

    command = get_command(username=username, **variant)

    if RENDERS or TELEOP or RENDERS_EVAL:
        print(command.split(";")[-1])
    else:
        dd.launch_shell(
            command=command,
            mode=mode_ssh,
            mount_points=mounts,
            verbose=True
        )


if RENDERS or TELEOP or RENDERS_EVAL or dd is None:
    run_exp(variants[0])
else:
    for variant in variants:
        run_exp(variant)
