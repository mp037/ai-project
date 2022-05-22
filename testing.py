# -*- coding: utf-8 -*-
"""
Created on Sun May 22 20:31:42 2022

@author: Kert PC
"""

# Based on: https://gym.openai.com/evaluations/eval_EIcM1ZBnQW2LBaFN6FY65g/
from collections import deque
import random
import math 
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time

from game import Game


class DQN(nn.Module):
    def __init__(self, input_size=12):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 16, (5,5), padding=2)
        self.conv2 = nn.Conv2d(16, 32, (5,5), padding=2)
        self.conv3 = nn.Conv2d(32, 48, (3,3), padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv4 = nn.Conv2d(48, 64, (3,3), padding=1)

        self.fc = nn.Linear(64 * input_size, 4)

        self.leaky = nn.LeakyReLU(0.1)
        
        
    def forward(self, x, crashes): 
        x = self.conv1(x)
        x = self.leaky(x)
        x = self.conv2(x)
        x = self.leaky(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.leaky(x)
        #x = self.pool(x)
        x = self.conv4(x)
        x = self.leaky(x)
        #x = torch.concat((torch.flatten(x, 1), crashes), dim=-1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    
class DQNRoomSolver:
            
    def __init__(self, map_name='bruce_lee'):
        self.map_name = 'bruce_lee'
        
        self.env = Game(visualize=True, load_map=True, map_name=self.map_name)
        m = self.env.m
        n = self.env.n
        self.input_size = ((m) // 2) * ((n) // 2)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Running on : ' + str(self.device))
        self.dqn = DQN(self.input_size)
        self.load_model()
        self.dqn.to(device=self.device)
        
    def load_model(self) :
        self.dqn.load_state_dict(torch.load('model/' + self.map_name))
        self.dqn.eval()

    def preprocess_state(self, state):
        return [torch.tensor(np.array([state[0]]), dtype=torch.float32),
                torch.tensor(np.array([state[1]]), dtype=torch.float32)]
    
    def choose_action(self, state):
        return torch.argmax(self.dqn(state[0].to(self.device), state[1].to(self.device)).cpu()).numpy()
            
    def action_to_letter(self, action) :
        if action == 0 :
            return 'w'
        if action == 1 :
            return 'a'
        if action == 2 :
            return 's'
        if action == 3 :
            return 'd'
    
    def run(self):
        actions = []
        state = self.preprocess_state(self.env.reset())
        done = False
        i = 0
        while not done:
            action = self.choose_action(state)
            next_state, reward, done = self.env.do_action(action)
            next_state = self.preprocess_state(next_state)
            actions.append(self.action_to_letter(action))
            state = next_state
            time.sleep(1)

if __name__ == '__main__':
    agent = DQNRoomSolver()
    agent.run()
    #agent.env.close()
