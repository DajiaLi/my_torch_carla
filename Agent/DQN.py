#!/usr/bin/env python3

import random
import numpy as np
import torch
from torch import nn, optim
from collections import deque
from Utils.Xception import xception
import gc
from config import dqn_setting
from Utils.CNN_models import DQN_model_64x3_CNN, DQN_model_64x3_CNN_abstract

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# MIN_REPLAY_MEMORY_SIZE = 20
MIN_REPLAY_MEMORY_SIZE = dqn_setting.MIN_REPLAY_MEMORY_SIZE
REPLAY_MEMORY_SIZE = dqn_setting.REPLAY_MEMORY_SIZE
MINIBATCH_SIZE = dqn_setting.MINIBATCH_SIZE
UPDATE_TARGET_EVERY = dqn_setting.UPDATE_TARGET_EVERY
DISCOUNT = dqn_setting.DISCOUNT
LR = dqn_setting.LEARNING_RATE

class DQNAgent:
    def __init__(self, model_name='xception', pretrain=None):
        self.action_dim = 3
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)  ## batch step

        self.target_update_counter = 0  # will track when it's time to update the target model

        self.model = self.create_model(model_name=model_name, pretrain=pretrain)
        self.target_model = self.create_model(model_name=model_name, pretrain=pretrain)
        # self.model = xception(num_classes=self.action_dim, pretrain=pretrain).to(device)
        # self.target_model = xception(num_classes=self.action_dim, pretrain=pretrain).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=LR)
        self.loss = nn.MSELoss()

    def create_model(self, model_name, pretrain):
        if model_name == 'xception':
            return xception(num_classes=self.action_dim, pretrain=pretrain).to(device)
        elif model_name == 'DQN_model_64x3_CNN':
            model = DQN_model_64x3_CNN()
            if pretrain is not None:
                model.load_state_dict(torch.load(pretrain), strict=True)
                print("successfully load pretrained cnn, pth=" + pretrain)
            return model.to(device)
        elif model_name == 'DQN_model_64x3_CNN_abstract':
            model = DQN_model_64x3_CNN_abstract()
            if pretrain is not None:
                model.load_state_dict(torch.load(pretrain), strict=True)
                print("successfully load pretrained cnn, pth=" + pretrain)
            return model.to(device)

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        state = torch.tensor(state.copy(), dtype=torch.float).unsqueeze(0).to(device)
        q_out = self.model(state)  # [1, 3]
        q_out = q_out.squeeze()  # [3]
        return q_out

    def train_in_loop(self, train_time=30):
        for i in range(train_time):
            # print("now is epoch %d" % (i, ))
            self.train()


    def train(self):
        gc.collect()
        torch.cuda.empty_cache()
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        rewards = np.array([transition[2] for transition in minibatch])
        rewards = np.expand_dims(rewards, axis=-1)  # 扩充维度
        rewards = torch.tensor(rewards, dtype=torch.int64).to(device)  # (batchsize,1)
        # print("rewards", rewards)

        actions = np.array([transition[1] for transition in minibatch])
        actions = np.expand_dims(actions, axis=-1)  # 扩充维度
        actions = torch.tensor(actions, dtype=torch.int64).to(device) #(batchsize,1)
        # print("actions:", actions)

        current_states = np.array([transition[0] for transition in minibatch]) # (batchsize, channel, height, width)
        current_states = torch.tensor(current_states, dtype=torch.float).to(device)
        current_qs_list = self.model(current_states)# shape(batchsize, 3)

        # print("current_qs_list:", current_qs_list)

        current_qs_list = current_qs_list.gather(1, actions)# shape(batchsize, 1)
        # print("current_qs_list:", current_qs_list)
        new_current_states = np.array([transition[3] for transition in minibatch])
        new_current_states = torch.tensor(new_current_states, dtype=torch.float).to(device)

        with torch.no_grad():
            future_qs_list = self.target_model(new_current_states) # shape(batchsize, 3)
            # print("future_qs_list:", future_qs_list)
            # max_future_q = future_qs_list.max(1)[0].view(-1, 1)
            # for index in range(MINIBATCH_SIZE):
            #     future_qs_list[index][actions[index]] = rewards[index] + DISCOUNT * max_future_q[index][0]

            max_future_q = future_qs_list.max(1)[0].view(-1, 1)
            # print("max_future_q:", max_future_q)
            future_qs_list = rewards + DISCOUNT * max_future_q
            # print("future_qs_list:", future_qs_list)
            # exit()

        l = self.loss(current_qs_list, future_qs_list)
        self.optimizer.zero_grad()
        l.backward()
        self.optimizer.step()

        self.target_update_counter += 1
        # print("self.target_update_counter", self.target_update_counter)
        if self.target_update_counter % UPDATE_TARGET_EVERY == 0:
            self.target_model.load_state_dict(self.model.state_dict())
