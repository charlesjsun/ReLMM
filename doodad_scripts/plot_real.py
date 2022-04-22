import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import os
import glob

import seaborn as sns

from scipy.ndimage.filters import gaussian_filter1d
try:
    import natsort
    sorted = natsort.natsorted
except ImportError:
    pass


def read_values(folder, keys):
#     filename = "results/" + name + "/result.json"
    filename = os.path.join(folder, "result.json")
    values = []
    with open(filename) as f:
        for line in f:
            result = json.loads(line)
            for key in keys:
                if isinstance(key, (tuple, list)):
                    r = None
                    for k in key:
                        r = result.get(k, None)
                        if r is not None:
                            break
                    if r is None:
                        r = 0
                    result = r
                else:
                    result = result.get(key, 0)
            values.append(result)
    return values

def read_all_values(names, keys):
    return [read_values(name, keys) for name in names]

def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)

def plot_values(exp_names, keys, smooth=None, epoch_len=1000, max_epochs=None):
    plt.figure(figsize=(12, 8))
#     plt.figure(figsize=(12 // 2, 8 // 2))
    for name, indices in exp_names.items():
        plt.title("/".join(key[0] if isinstance(key, (list, tuple)) else key for key in keys))
        all_values = read_all_values([name + "_" + str(i) for i in indices], keys)
        values, error = tolerant_mean(all_values)
        if keys == ["time_total_s"]:
            values /= 3600.0
            error /= 3600.0
            plt.title("time_total_hr")
        if smooth:
            from scipy.ndimage.filters import gaussian_filter1d
            values = gaussian_filter1d(values, sigma=smooth)
            error = gaussian_filter1d(error, sigma=smooth)    
        epochs = (np.arange(len(values)) + 1) * epoch_len
        if max_epochs is not None:
            epochs = epochs[:max_epochs]
            values = values[:max_epochs]
            error = error[:max_epochs]
        plt.plot(epochs, values, label=name)
        plt.fill_between(epochs, values - error, values + error, alpha=0.4)
        plt.legend(loc='upper left', bbox_to_anchor=(0.4, 0.5))

def get_log_folders(paths):
    folders = []
    if isinstance(paths, list) and paths:
        folder = []
        for path in paths:
            folder.extend(glob.glob(path))
    else:
        folders = glob.glob(paths)
    folders = sorted(folders)
    return folders


old_runs = [
    "/home/brian/ray_results/real_trained_load_nag4_bonus_uniform_3",
    "/home/brian/ray_results/real_trained_load_nag4_bonus_uniform_5",
    "/home/brian/ray_results/real_trained_load_nag4_bonus_uniform_6_2020-10-20T11-30-48",
    "/home/brian/ray_results/real_trained_load_nag4_bonus_uniform_6",
    "/home/brian/ray_results/real_trained_load_nag4_bonus_uniform_8",
    "/home/brian/ray_results/real_trained_load_nag4_bonus_uniform_9",
    "/home/brian/ray_results/real_trained_load_nag4_bonus_uniform_10",
    "/home/brian/ray_results/real_trained_load_nag4_bonus_uniform_11",
]

new_runs = [
    "/home/brian/ray_results/real_sock2000_3",
    "/home/brian/ray_results/real_sock2000_4",
    "/home/brian/ray_results/real_sock2000_5",
    "/home/brian/ray_results/real_sock2000_7",
    #"/home/brian/ray_results/real_sock2000_8",
    #"/home/brian/ray_results/real_sock2000_9",
    #"/home/brian/ray_results/real_sock2000_10"
]

#new_runs = [                                                                                                                                                                             
#    "/home/brian/ray_results/real_obstacles2_sock2000_1",
#    "/home/brian/ray_results/real_obstacles2_sock2000_3",
#]

# new_runs = [
#     "/home/brian/ray_results/no_obst_sock2000_1",
#     "/home/brian/ray_results/no_obst_sock2000_2",
#     "/home/brian/ray_results/no_obst_sock2000_3",
#     "/home/brian/ray_results/no_obst_sock2000_4",
#     "/home/brian/ray_results/no_obst_sock2000_5",
#     "/home/brian/ray_results/no_obst_sock2000_6",
#     "/home/brian/ray_results/no_obst_sock2000_7",
#     "/home/brian/ray_results/no_obst_sock2000_8",
#     "/home/brian/ray_results/no_obst_sock2000_9",
#     "/home/brian/ray_results/no_obst_sock2000_10",
#     "/home/brian/ray_results/no_obst_sock2000_11"
# ]

# new_runs = [
#     "/home/brian/ray_results/no_obst_2_sock2000_1",
#     "/home/brian/ray_results/no_obst_2_sock2000_2"
# ]

# new_runs = [
#     "/home/brian/ray_results/no_obst_3_sock2000_2",
#     "/home/brian/ray_results/no_obst_3_sock2000_3",
#     "/home/brian/ray_results/no_obst_3_sock2000_4",
#     "/home/brian/ray_results/no_obst_3_sock2000_5",
#     "/home/brian/ray_results/no_obst_3_sock2000_6",
# ]

# last no obs that works
# new_runs = [
#     "/home/brian/ray_results/no_obst_4_sock2000_1",
#     "/home/brian/ray_results/no_obst_4_sock2000_2",
#     "/home/brian/ray_results/no_obst_4_sock2000_3",
#     "/home/brian/ray_results/no_obst_4_sock2000_4"
# ]

# new_runs = [
#     "/home/brian/ray_results/obst_1_sock2000_8",
#     "/home/brian/ray_results/obst_1_sock2000_10",
#     "/home/brian/ray_results/obst_1_sock2000_11",
# ]

no_obst_runs = [
    "/mnt/Storage/mobile_data/ray_results/real_sock2000_3",
    "/mnt/Storage/mobile_data/ray_results/real_sock2000_4",
    "/mnt/Storage/mobile_data/ray_results/real_sock2000_5",
    "/mnt/Storage/mobile_data/ray_results/real_sock2000_7",
    "/mnt/Storage/mobile_data/ray_results/real_sock2000_8",
    "/mnt/Storage/mobile_data/ray_results/real_sock2000_9",
    "/mnt/Storage/mobile_data/ray_results/real_sock2000_10"
]

obst_runs = [
    "/mnt/Storage/mobile_data/ray_results/obst_3_sock2000_5",
    "/mnt/Storage/mobile_data/ray_results/obst_3_sock2000_6",
    "/mnt/Storage/mobile_data/ray_results/obst_3_sock2000_7",
    "/mnt/Storage/mobile_data/ray_results/obst_3_sock2000_8",
    "/mnt/Storage/mobile_data/ray_results/obst_3_sock2000_9",
    "/mnt/Storage/mobile_data/ray_results/obst_3_sock2000_10",
    "/mnt/Storage/mobile_data/ray_results/obst_3_sock2000_11",
    "/mnt/Storage/mobile_data/ray_results/obst_3_sock2000_12"
]

# 
curriculum = get_log_folders("/home/charles/ray_results/curriculum_v2_*")

curriculum_runs_diverse_objects = get_log_folders("/home/charles/ray_results/curriculum_diverse_obj_v1*")

rug_runs = get_log_folders("/mnt/Storage/mobile_data/ray_results/rag2_room_*") + get_log_folders("/home/charles/ray_results/rag2_room_*")

# different_objects_no_rugs_runs = get_log_folders('/home/brian/ray_results/different_objects_no_rags3_*')

diverse_objects_small_room_runs = get_log_folders('/mnt/Storage/mobile_data/ray_results/diverse_objects_small_room1_*')
# diverse_objects_rugs_runs = get_log_folders('/mnt/Storage/mobile_data/ray_results/diverse_objects_small_room1_*') + \
diverse_objects_rugs_runs = get_log_folders('/home/charles/ray_results/diverse_rugs_bootstrapped_from_diverse_objects_*')
# curriculum = get_log_folders('/media/brian/691147a3-7d84-420f-ac7b-8cefde0b7be9/cur/*')


def plot(runs, name, title, ylabel, keys, values_multiplier=1, smooth=None, discard=0, shift_x=0):
    ax = plt.gca()

    values = []
    for run in runs:
        values.extend(read_values(run, keys))

    if smooth is not None:
        values = gaussian_filter1d(values, sigma=smooth)

    samples = []
    discard_samples = []
    samples_keys = ["sampler", "pool-size"]
    for i, run in enumerate(runs):
        if i < discard:
            discard_samples.extend(read_values(run, samples_keys))
        else:
            samples.extend(read_values(run, samples_keys))

    if discard > 0:
        print(discard_samples[-1])
        values = values[len(discard_samples):]
        samples = np.array(samples) - discard_samples[-1] + 500

    # get training time
    training_time_values = [read_values(run, ["time_total_s"]) for run in runs]
    training_times = np.array([training_times_run[-1] if training_times_run else 0 for training_times_run in training_time_values]) / 3600
    training_times_summed = []
    sum = 0
    for training_time in training_times:
        sum += training_time
        training_times_summed.append(sum)
    training_time_total = np.sum(training_times)
    print(name, 'run time:', training_time_total, 'h')
    # print(name, 'training_times_summed:', training_times_summed, 'h')

    samples = np.array(samples)
    samples_scaling = training_time_total / samples[-1]
    shift_x = shift_x * samples_scaling
    samples = samples * samples_scaling + shift_x

    # plt.figure(figsize=(12, 8))
    # plt.title("/".join(key[0] if isinstance(key, (list, tuple)) else key for key in keys))
    plt.title(title)

    plt.xlabel("training time [h]")
    plt.ylabel(ylabel)

    color = next(ax._get_lines.prop_cycler)['color']

    plt.plot(samples, values * values_multiplier, label=name, color=color)
    # print('last x:', samples[-1])
    if shift_x and values.size:
        plt.plot([0, shift_x], [0, values[0] * values_multiplier], '--', color=color)

    # plt.show()


sns.set()
sns.set_style(style='whitegrid')
sns.set_palette(sns.color_palette('colorblind'))

plt.rc('font', family='sans-serif', weight='normal')
mpl.rcParams['font.sans-serif'] = 'Helvetica'
plt.rc("figure", figsize=(6, 4))
plt.rc("lines", linewidth=3)

plt.rc("axes", titlesize=20, labelsize=16)
plt.rc("axes.formatter", use_mathtext=True)

plt.rc("xtick", labelsize=16)
plt.rc("ytick", labelsize=16)

plt.rc("legend", fontsize=12)

mpl.rcParams["figure.autolayout"] = True

smooth = 1.5

# plt.figure()
# #plot(old_runs, "run1", "Real Training Performance", "objects collected per 500 steps", ["training", "environment_infos", "success_per_500_steps"], smooth=1, discard=2)
# plot(new_runs, "run2", "Real Training Performance", "objects collected per 250 steps", ["training", "environment_infos", "success_per_500_steps"], smooth=smooth)
# plt.show()

# plt.figure()
# #plot(old_runs, "run1", "Real Steps Between Object Grasps", "Steps to object", ["training", "episode-length-mean"], smooth=1, discard=2)
# plot(new_runs, "run2", "Real Steps Between Object Grasps", "Steps to object", ["training", "episode-length-mean"], smooth=smooth)
# plt.show()

timesteps = 65

plt.figure()
plot(no_obst_runs, "no obst, StatCurr", "Real Training Performance", "objects collected unit area",
     ["training", "environment_infos", "success_per_500_steps"],
     smooth=smooth, values_multiplier=0.5 * (2 ** 2) / 20, shift_x=2000)
plot(curriculum, "no obst, AutoCurr", "Real Training Performance", "objects collected unit area",
     ["training", "environment_infos", "success_per_500_steps"],
     smooth=smooth, values_multiplier=0.5 * (2 ** 2) / 20)
plot(obst_runs, "obst, StatCurr", "Real Training Performance", "objects collected unit area",
     ["training", "environment_infos", "success_per_500_steps"],
     smooth=smooth, values_multiplier=0.5 * (3 ** 2) / 20, shift_x=2000)
plot(diverse_objects_small_room_runs, "diverse, StatCurr", "Real Training Performance", "objects collected unit area",
     ["training", "environment_infos", "success_per_500_steps"],
     smooth=smooth, values_multiplier=0.5 * (3 * 2) / 20, shift_x=2150)
plot(rug_runs, "obst+rugs, StatCurr", "Real Training Performance", "objects collected unit area",
     ["training", "environment_infos", "success_per_500_steps"],
     smooth=smooth, values_multiplier=0.5 * (3 * 3.5) / 20, shift_x=2000)
# plot(diverse_objects_rugs_runs, "diverse+rugs, StatCurr", "Real Training Performance", "objects collected unit area",
#      ["training", "environment_infos", "success_per_500_steps"],
#      smooth=smooth, values_multiplier=0.5 * (3 * 3.5) / 20, shift_x=2000)
#
# plot(different_objects_no_rugs_runs, "diff obj no rug", "Real Training Performance", "objects collected unit area", ["training", "environment_infos", "success_per_500_steps"],
#         smooth=smooth, values_multiplier=0.5 * (3 ** 2) / 30)
plt.xlim(0, timesteps)
plt.legend(loc="top left")
plt.show()

plt.figure()
plot(no_obst_runs, "no obst, StatCurr", "Real Steps Between Object Grasps", "hours to object unit area",
     ["training", "episode-length-mean"],
     smooth=smooth, values_multiplier=20.0 / (2 ** 2), shift_x=2000)
plot(curriculum, "no obst, AutoCurr", "Real Steps Between Object Grasps", "hours to object unit area",
     ["training", "episode-length-mean"],
     smooth=smooth, values_multiplier=20.0 / (2 ** 2))
plot(obst_runs, "obst, StatCurr", "Real Steps Between Object Grasps", "hours to object unit area",
     ["training", "episode-length-mean"],
     smooth=smooth, values_multiplier=20.0 / (3 ** 2), shift_x=2000)
plot(diverse_objects_small_room_runs, "diverse, StatCurr", "Real Steps Between Object Grasps",
     "hours to object unit area", ["training", "episode-length-mean"],
     smooth=smooth, values_multiplier=20.0 / (3 * 2), shift_x=2150)
plot(rug_runs, "obst+rugs, StatCurr", "Real Steps Between Object Grasps", "hours to object unit area",
     ["training", "episode-length-mean"],
     smooth=smooth, values_multiplier=20.0 / (3 * 3.5), shift_x=2000)
# plot(diverse_objects_rugs_runs, "diverse+rugs, StatCurr", "Real Steps Between Object Grasps", "hours to object unit area",
#      ["training", "episode-length-mean"],
#      smooth=smooth, values_multiplier=20.0 / (3 * 3.5), shift_x=2150)
# plot(different_objects_no_rugs_runs, "dif obj no rug", "Real Steps Between Object Grasps", "hours to object unit area", ["training", "episode-length-mean"],
#         smooth=smooth, values_multiplier=30.0 / (3 ** 2))
plt.xlim(0, timesteps)
# plt.legend(loc="upper right")
plt.show()

keys = ["time_total_s"]
values = []
for run in new_runs:
    this_values = read_values(run, keys)
    if len(values) > 0:
        this_values = [v + values[-1] for v in this_values]
    values.extend(this_values)
print(values[-1])
samples = []
samples_keys = ["sampler", "pool-size"]
for i, run in enumerate(new_runs):
    samples.extend(read_values(run, samples_keys))
print(samples[-1])
# plt.figure()
# plot(old_runs, "run1", "Grasp Success", "success rate", ["training", "environment_infos", "grasp-num_successes_per_grasp"], smooth=1, discard=2)
# plot(new_runs, "run1", "Grasp Success", "success rate", ["training", "environment_infos", "grasp-num_successes_per_grasp"], smooth=1)
# plt.show()

# plt.figure()
# plot(old_runs, "run2", "Grasp Action Success", "success rate", ["training", "environment_infos", "grasp-num_successes_per_action"], smooth=1, discard=2)
# plot(new_runs, "run2", "Grasp Action Success", "success rate", ["training", "environment_infos", "grasp-num_successes_per_action"], smooth=1)
# plt.show()

# buffer = np.load("/home/brian/realmobile/mobilemanipulation/others/logs/sock_8500/train_buffer.npy", allow_pickle=True)[()]
# rewards = buffer["rewards"][:8552]
# accuracy = []
# for i in range(0, 6000, 50):
#     r = rewards[i:i+50] 
#     accuracy.append(np.sum(r) / 50.0)
# accuracy = np.array(accuracy)
# accuracy = gaussian_filter1d(accuracy, sigma=2.0)
# plt.plot(np.arange(len(accuracy)) * 50, accuracy)
# plt.show()
