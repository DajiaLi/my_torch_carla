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


EPISODES = ddpg_setting.EPISODES
INTERVAL_NUM = ddpg_setting.INTERVAL_NUM
WIDTH = ddpg_setting.WIDTH
HEIGHT = ddpg_setting.HEIGHT

SIGMA_DECAY = ddpg_setting.SIGMA_DECAY
MIN_SIGMA = ddpg_setting.MIN_SIGMA
def train_ddpg(episodes=EPISODES,
               interval_num=INTERVAL_NUM,
               abstract_train=False,
               txtname="ddpg_train",
               start_location="default",
               sigma=ddpg_setting.SIGMA):
    random.seed(1)
    np.random.seed(2)

    # 删除历史训练记录
    if os.path.exists(txtname):
        os.remove(txtname)

    # 创建权重保存文件夹
    dt = datetime.datetime.now()
    save_path = "./trained_model/save" + "%s-%s-%s-%s-%s-%s" % (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    os.makedirs(save_path)

    agent = DDPGAgent()
    env = CarEnv(im_width=WIDTH, im_height=HEIGHT, startlocation=start_location, action_type="continuous", reward_type=ddpg_setting.reward_type)

    for episode in tqdm(range(1, episodes + 1), unit='episodes'):
        episode_reward = 0.0  # 当前轮次的累计reward
        current_state = env.reset()  # 像素在0-255之间
        current_state = current_state.astype(np.float32) / 255
        if abstract_train:
            current_state = abstract_data_numpy(current_state,interval_num=interval_num)  # (channel, 2 * height, width)
        # 将(height, width, channel)转换为(channel, height, width)
        current_state = current_state.transpose((2, 0, 1))

        while True:
            action = agent.get_next_action(current_state, sigma=sigma, disturbance=True)
            # print("action", action)
            new_state, reward, done, _ = env.step(action)
            new_state = new_state.astype(np.float32) / 255
            if abstract_train:
                new_state = abstract_data_numpy(new_state, interval_num=interval_num)
            new_state = new_state.transpose((2, 0, 1))
            episode_reward += reward
            # if reward >= -0.99 or done is True: # 车静止时返回-1并且done=false,静止时不计入经验
            agent.update_replay_memory((current_state, action, reward, new_state, done))
            current_state = new_state
            if done:
                print("episode %d, episode_reward %f, sigma %f" % (episode, episode_reward, sigma))
                break

        with open(txtname, 'a') as f:
            f.write("episode %d, episode_reward %f, sigma %f\n" % (episode, episode_reward, sigma))

        agent.train_in_loop(train_time=ddpg_setting.TRAIN_TIME)
        # 每100次保存权重
        if episode % 50 == 0:
            torch.save(obj=agent.model_action.state_dict(),
                       f=save_path + "/ddpg_action_" + "%depoch.pth" % (episode,))
            torch.save(obj=agent.model_value.state_dict(),
                       f=save_path + "/ddpg_value_" + "%depoch.pth" % (episode,))

        sigma = max(MIN_SIGMA, sigma * SIGMA_DECAY)

if __name__ == '__main__':
    train_ddpg(abstract_train=True,
               txtname="ddpg_train_pointdefault2_abstract.txt",
               start_location="default2")



