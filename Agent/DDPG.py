#!/usr/bin/env python3

import random
import numpy as np
import torch
from torch import nn, optim
from collections import deque
from Utils.Xception import xception
from Utils.CNN_models import DDPG_action2_model_64x3_CNN_abstract, DDPG_value2_model
import gc
from config import ddpg_setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MIN_REPLAY_MEMORY_SIZE = ddpg_setting.MIN_REPLAY_MEMORY_SIZE
REPLAY_MEMORY_SIZE = ddpg_setting.REPLAY_MEMORY_SIZE
MINIBATCH_SIZE = ddpg_setting.MINIBATCH_SIZE
DISCOUNT = ddpg_setting.DISCOUNT


class DDPGAgent:
    def __init__(self, pretrain_action=None, pretrain_value=None):
        self.MIN_REPLAY_MEMORY_SIZE = MIN_REPLAY_MEMORY_SIZE
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)  ## batch step

        self.model_action = self.create_model(model_name='DDPG_action2_model_64x3_CNN_abstract', pretrain=pretrain_action)
        self.model_action_next = self.create_model(model_name='DDPG_action2_model_64x3_CNN_abstract', pretrain=pretrain_action)
        self.model_action_next.load_state_dict(self.model_action.state_dict())

        self.model_value = self.create_model(model_name='DDPG_value2_model', pretrain=pretrain_value)
        self.model_value_next = self.create_model(model_name='DDPG_value2_model', pretrain=pretrain_value)
        self.model_value_next.load_state_dict(self.model_value.state_dict())

        self.optimizer_action = optim.Adam(params=self.model_action.parameters(), lr=ddpg_setting.ACTION_LR)
        self.optimizer_value = optim.Adam(params=self.model_value.parameters(), lr=ddpg_setting.VALUE_LR)
        self.loss_fn = nn.MSELoss()
        random.seed(ddpg_setting.RANDOM_SEED)

    def create_model(self, model_name, pretrain=None):
        model = None
        if model_name == 'DDPG_action2_model_64x3_CNN_abstract':
            model = DDPG_action2_model_64x3_CNN_abstract()
        elif model_name == 'DDPG_value2_model':
            model = DDPG_value2_model()
        if pretrain is not None:
            model.load_state_dict(torch.load(pretrain), strict=True)
            print("successfully load pretrained cnn, pth=" + pretrain)
        return model.to(device)

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_next_action(self, _state, sigma=0.1, disturbance=True):
        # 给动作添加噪声,增加探索
        _state = torch.tensor(_state.copy(), dtype=torch.float).unsqueeze(0).to(device)
        _action = self.model_action(_state)  # [1, 2]
        if disturbance:
            _action += random.normalvariate(mu=ddpg_setting.MU, sigma=sigma)
        _action = _action.squeeze()  # []
        _action = _action.cpu().detach().numpy()  # tensor -> numpy
        return _action

    def get_values(self, states, actions):
        # 直接评估综合了state和action的value
        # [b, channel, height, width] -> [b, 2048]
        features = self.model_action(states, get_feature=True) # [b, 22080]
        values = self.model_value(features, actions)
        return values

    def get_next_values(self, next_states, rewards):
        # 对next_state的评估需要先把它对应的动作计算出来,这里用model_action_next来计算
        # [b, channel, height, width] -> [b, 2]
        next_actions = self.model_action_next(next_states)
        # 和value的计算一样,action拼合进next_state里综合计算
        target_values = self.get_values(next_states, next_actions)
        target_values = rewards + target_values * DISCOUNT
        return target_values

    def get_loss_action(self, states):
        # 首先把动作计算出来
        # [b, channel, height, width] -> [b, 2]
        actions = self.model_action(states)
        # 使用value网络评估动作的价值,价值是越高越好
        # 因为这里是在计算loss,loss是越小越好,所以符号取反
        # [b, 4] -> [b, 1] -> [1]
        loss = -self.get_values(states, actions).mean()

        return loss

    def soft_update(self, model, model_next):
        for param, param_next in zip(model.parameters(), model_next.parameters()):
            # 以一个小的比例更新
            value = param_next.data * 0.995 + param.data * 0.005
            param_next.data.copy_(value)

    def get_sample(self, batch_size):
        minibatch = random.sample(self.replay_memory, batch_size)

        rewards = np.array([transition[2] for transition in minibatch])
        rewards = np.expand_dims(rewards, axis=-1)  # 扩充维度
        rewards = torch.tensor(rewards, dtype=torch.int64).to(device)  # (batchsize,1)
        # print("rewards", rewards)

        actions = np.array([transition[1] for transition in minibatch])
        # actions = np.expand_dims(actions, axis=-1)  # 扩充维度
        actions = torch.tensor(actions, dtype=torch.float).to(device)  # (batchsize,2)
        # print("actions:", actions)

        current_states = np.array([transition[0] for transition in minibatch])  # (batchsize, channel, height, width)
        current_states = torch.tensor(current_states, dtype=torch.float).to(device)

        new_current_states = np.array([transition[3] for transition in minibatch])
        new_current_states = torch.tensor(new_current_states, dtype=torch.float).to(device)

        return current_states, actions, rewards, new_current_states

    def train(self):
        gc.collect()
        torch.cuda.empty_cache()
        if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
            return
        current_states, actions, rewards, new_current_states = self.get_sample(MINIBATCH_SIZE)

        #计算value和target
        values = self.get_values(current_states, actions)
        next_values = self.get_next_values(new_current_states, rewards)
        #两者求差,计算loss,更新参数
        loss_value = self.loss_fn(values, next_values)

        self.optimizer_value.zero_grad()
        loss_value.backward()
        self.optimizer_value.step()

        #使用value网络评估action网络的loss,更新参数
        loss_action = self.get_loss_action(current_states)

        self.optimizer_action.zero_grad()
        loss_action.backward()
        self.optimizer_action.step()

        #以一个小的比例更新
        self.soft_update(self.model_action, self.model_action_next)
        self.soft_update(self.model_value, self.model_value_next)

    def train_in_loop(self, train_time=30):
        if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
            return
        for i in range(train_time):
            self.train()

if __name__ == '__main__':
    agent = DDPGAgent()
    # state = np.random.rand(3, 600, 400)
    # print(agent.get_next_action(state, disturbance=True))
    state = np.random.rand(10, 3, 600, 400)
    action = np.random.rand(10, 2)
    state = torch.tensor(state, dtype=torch.float).to(device)
    action = torch.tensor(action, dtype=torch.float).to(device)
    agent.get_values(state, action)
    # action = agent.model_action(state)
    # print("action ",action.shape)
    # values = agent.get_values(state, action)
    # print("values",values)
    # print("values", values.mean())
