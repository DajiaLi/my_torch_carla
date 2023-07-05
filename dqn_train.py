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

EPISODES = dqn_setting.EPISODES
EPSILON = dqn_setting.EPSILON
INTERVAL_NUM = dqn_setting.INTERVAL_NUM
EPSILON_DECAY = dqn_setting.EPSILON_DECAY
MIN_EPSILON = dqn_setting.MIN_EPSILON
MIN_REPLAY_MEMORY_SIZE = dqn_setting.MIN_REPLAY_MEMORY_SIZE
WIDTH = dqn_setting.WIDTH
HEIGHT = dqn_setting.HEIGHT
TRAIN_TIME = dqn_setting.TRAIN_TIME

def train_dqn(episodes=EPISODES, epsilon=EPSILON, interval_num=INTERVAL_NUM, abstract_train=False, txtname="dqn_train",
              start_location="default", model_name='xception', pretrain=None):
    random.seed(1)
    np.random.seed(2)

    # 删除历史训练记录
    if os.path.exists(txtname):
        os.remove(txtname)

    # 创建权重保存文件夹
    dt = datetime.datetime.now()
    save_path = "./trained_model/save" + "%s-%s-%s-%s-%s-%s" % (
        dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    os.makedirs(save_path)

    agent = DQNAgent(model_name=model_name, pretrain=pretrain)
    env = CarEnv(im_width=WIDTH, im_height=HEIGHT, startlocation=start_location)
    for episode in tqdm(range(1, episodes + 1), unit='episodes'):

        episode_reward = 0  # 当前轮次的累计reward
        current_state = env.reset()  # 像素在0-255之间
        current_state = current_state.astype(np.float32) / 255
        if abstract_train:
            current_state = abstract_data_numpy(current_state,
                                                interval_num=interval_num)  # (channel, 2 * height, width)
        # 将(height, width, channel)转换为(channel, height, width)
        current_state = current_state.transpose((2, 0, 1))
        while True:
            if np.random.random() > epsilon:
                action = torch.argmax(agent.get_qs(current_state)).item()
            else:
                action = np.random.randint(0, agent.action_dim)
            # print("a", action)
            new_state, reward, done, _ = env.step(action)
            new_state = new_state.astype(np.float32) / 255
            if abstract_train:
                new_state = abstract_data_numpy(new_state, interval_num=interval_num)
            new_state = new_state.transpose((2, 0, 1))

            episode_reward += reward
            agent.update_replay_memory((current_state, action, reward, new_state, done))

            current_state = new_state

            if done:
                print("episode %d, episode_reward %d, epsilon %f" % (episode, episode_reward, epsilon))
                break

        # for actor in env.actor_list:
        #     actor.destroy()

        # 记录训练情况
        with open(txtname, 'a') as f:
            f.write("episode %d, episode_reward %d, epsilon %f\n" % (episode, episode_reward, epsilon))

        if len(agent.replay_memory) >= MIN_REPLAY_MEMORY_SIZE:
            agent.train_in_loop(train_time=TRAIN_TIME)

        # 每100次保存权重
        if episode % 50 == 0:
            torch.save(obj=agent.model.state_dict(),
                       f=save_path + "/dqn_" + "%depoch.pth" % (episode,))

        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

    if len(env.actor_list) > 0:
        for actor in env.actor_list:
            actor.destroy()


if __name__ == '__main__':
    train_dqn(abstract_train=True,
              txtname="dqn_train_cnn_default_abstract2.txt",
              start_location="default",
              model_name='DQN_model_64x3_CNN_abstract',
              pretrain=None)
    # train_dqn(abstract_train=False, txtname="dqn_train_default2.txt", start_location="default2", model_name='model_base_64x3_CNN', pretrain=None)

