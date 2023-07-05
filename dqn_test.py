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
from Agent.DQN import DQNAgent
from Utils.Util import abstract_data_numpy
from config import dqn_setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    sys.path.append(glob.glob('E:/carla/CARLA_0.9.11/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


INTERVAL_NUM = dqn_setting.INTERVAL_NUM

def test_dqn(abstract_train=False, pretrain=None, model_name='xception', interval_num=INTERVAL_NUM,
             start_location="default"):
    agent = DQNAgent(model_name=model_name, pretrain=pretrain)
    env = CarEnv(im_width=400, im_height=300, startlocation=start_location)

    current_state = env.reset()  # 像素在0-255之间
    current_state = current_state.astype(np.float32) / 255
    if abstract_train:
        current_state = abstract_data_numpy(current_state, interval_num=interval_num)
    # 将(height, width, channel)转换为(channel, height, width)
    current_state = current_state.transpose((2, 0, 1))  # (channel, 2 * height, width)

    while True:
        action = torch.argmax(agent.get_qs(current_state)).item()
        new_state, reward, done, _ = env.step(action)
        new_state = new_state.astype(np.float32) / 255
        if abstract_train:
            new_state = abstract_data_numpy(new_state, interval_num=interval_num)
        new_state = new_state.transpose((2, 0, 1))
        current_state = new_state
        if done:
            break
    env.reset()

if __name__ == '__main__':
    test_dqn(abstract_train=True,
             model_name='DQN_model_64x3_CNN_abstract',
             start_location="default",
             pretrain='./trained_model/DQN_cnn_pointdefault_abstract2/dqn_1000epoch.pth')