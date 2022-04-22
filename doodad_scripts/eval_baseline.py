import numpy as np
import time
import pprint


from softlearning.environments.utils import get_environment_from_params

environment_params = {
    "universe": "gym",
    "domain": "Locobot",
    "task": "RealNavigationGraspingDualPerturbation-v0",
    "kwargs": {
        "grasp_algorithm": "baseline",
        "grasp_perturbation": "random_uniform",
        "grasp_perturbation_params": {
            "num_steps": 10,
        },
        "nav_perturbation": "random_uniform",
        "nav_perturbation_params": {
            "num_steps": 10,
        },
        "pause_filepath": "/home/brian/realmobile/locobot_pause",
        "add_uncertainty_bonus": False,
        "alive_penalty": 0.0,
        "is_training": False,
        'reset_free': False,
        'observation_keys': ('pixels',),
        'max_ep_len': 250,
        # 'grasp_algorithm_params': {
        #     'grasp_model_name': 'sock_2000',
        # },
    }
}
env = get_environment_from_params(environment_params)


def policy(obs):
    centroids = env.grasp_algorithm.get_centroids(obs["pixels"])
    if len(centroids) == 0:
        print("random move")
        return np.random.uniform(-1.0, 1.0, size=(2,))

    positions = [env.grasp_algorithm.get_world_from_pixel(c) for c in centroids]
    closest = min(positions, key=lambda p: p[0] ** 2 + p[1] ** 2)

    radius = np.clip(np.sqrt(closest[0] ** 2 + closest[1] ** 2), 0.0, 0.1)
    theta = np.clip(np.arctan2(closest[1], closest[0]), -np.pi / 12, np.pi / 12)

    a0 = radius / 0.1 * 2.0 - 1.0
    a1 = theta / (np.pi / 12)

    return np.array([a0, a1])


try:
    # eval
    env.reset()
    print("starting in 20s")
    time.sleep(20)
    print("start")
    rewards = []
    start_time = time.time()
    success_timings = []
    i = 0
    while time.time() - start_time < 60 * 15:  # 15 minutes for each eval
        print(i)
        i += 1
        obs = env.get_observation()
        action = policy(obs)
        print(action)
        env.do_move(action)
        time.sleep(0.55)
        reward = env.do_grasp(action, {})
        print(reward)
        if reward:
            success_timings.append(time.time() - start_time)
        rewards.append(reward)
except Exception as e:
    print(e)

infos = {}
# infos["rewards"] = rewards
infos["no_respawn_eval_returns"] = sum(rewards)

pprint.pprint(infos)
print(rewards)
print(success_timings)
print('objects grasped:', len(success_timings))