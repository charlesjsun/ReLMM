from softlearning.environments.gym.locobot.real_envs import RealLocobotGraspingEnv
import numpy as np
import time
import os

env = RealLocobotGraspingEnv()

def depth_bar(): 
    depth = env._interface.get_depth() 
    true_depth = depth[:, :, 0] + depth[:, :, 1] * 256 
    true_depth = np.clip(true_depth, 0, 1900) 
    cropped = true_depth[0:250, 200:420] 
    sec_meds = np.array([np.median(p, axis=0) for p in np.split(cropped, 5, axis=0)]) 
    # left_meds = np.median(sec_meds[:, 0:5], axis=1, keepdims=True) 
    # right_meds = np.median(sec_meds[:, -5:], axis=1, keepdims=True) 
    # meds = np.concatenate([left_meds, right_meds], axis=1) 
    # return meds
    left_meds = np.median(sec_meds[:, 0:5], axis=1) 
    right_meds = np.median(sec_meds[:, -5:], axis=1) 
    return left_meds, right_meds

def show_depth(): 
    import matplotlib.pyplot as plt 
    depth = env._interface.get_depth() 
    fig, axs = plt.subplots(1, 3, figsize=(18 * 2, 8 * 2)) 
    
    im0 = axs[0].imshow(depth[:, :, 0], cmap='hot') 
    fig.colorbar(im0, ax=axs[0]) 
    
    im1 = axs[1].imshow(depth[:, :, 1], cmap='hot') 
    fig.colorbar(im1, ax=axs[1]) 
    
    true_depth = depth[:, :, 0] + depth[:, :, 1] * 256 
    true_depth = np.clip(true_depth, 0, 1900) 
    im2 = axs[2].imshow(true_depth, cmap='hot') 
    fig.colorbar(im2, ax=axs[2])

    plt.show()

thresh = np.array([472, 516, 570, 639, 725])

def detect_wall(k=1):
    left_meds, right_meds = depth_bar()
    left_num_detect = np.sum(left_meds <= thresh)
    right_num_detect = np.sum(right_meds <= thresh)
    return left_num_detect >= k or right_num_detect >= k