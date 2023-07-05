#!/usr/bin/env python3
import glob
import os
import random
import sys

import numpy as np
import torch
from tqdm import tqdm
import datetime
from Env import CarEnv
from Agent.DDPG import DDPGAgent
from Utils.Util import abstract_data_numpy
from config import ddpg_setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    sys.path.append(glob.glob('E:/carla/CARLA_0.9.11/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


INTERVAL_NUM = ddpg_setting.INTERVAL_NUM

def test_ddpg(abstract_train=False,
              pretrain_action=None,
              pretrain_value=None,
              interval_num=INTERVAL_NUM,
              start_location="default"):
    agent = DDPGAgent(pretrain_action=pretrain_action, pretrain_value=pretrain_value)
    env = CarEnv(im_width=400, im_height=300, startlocation=start_location, action_type="continuous", reward_type=ddpg_setting.reward_type)

    current_state = env.reset()  # 像素在0-255之间
    current_state = current_state.astype(np.float32) / 255
    if abstract_train:
        current_state = abstract_data_numpy(current_state, interval_num=interval_num)
    # 将(height, width, channel)转换为(channel, height, width)
    current_state = current_state.transpose((2, 0, 1))  # (channel, 2 * height, width)
    episode_reward = 0  # 当前轮次的累计reward
    while True:
        action = agent.get_next_action(current_state, disturbance=False)
        new_state, reward, done, _ = env.step(action)
        new_state = new_state.astype(np.float32) / 255
        if abstract_train:
            new_state = abstract_data_numpy(new_state, interval_num=interval_num)
        new_state = new_state.transpose((2, 0, 1))
        episode_reward += reward
        current_state = new_state
        if done:
            break
    env.reset()

if __name__ == '__main__':
    test_ddpg(abstract_train=True,
             start_location="default",
             pretrain_action='./trained_model/DDPG_cnn_pointdefault_liner_abstract/ddpg_action_200epoch.pth',
             pretrain_value='./trained_model/DDPG_cnn_pointdefault_liner_abstract/ddpg_value_200epoch.pth',)