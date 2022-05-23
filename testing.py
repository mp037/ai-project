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
import cv2
from mss import mss
from PIL import Image
from game import Game
from matplotlib import pyplot as plt
import glob
from PIL import Image


class DQN(nn.Module):
    def __init__(self, input_size=12):
        super().__init__()
        
        self.conv1 = nn.Conv2d(4, 16, (5,5), padding=2)
        self.conv2 = nn.Conv2d(16, 32, (5,5), padding=2)
        self.conv3 = nn.Conv2d(32, 48, (3,3), padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv4 = nn.Conv2d(48, 64, (3,3), padding=1)

        self.fc = nn.Linear(64 * input_size, 4)
        """
        
        self.conv1 = nn.Conv2d(4, 8, (5,5), padding=2)
        self.conv2 = nn.Conv2d(8, 16, (5,5), padding=2)
        self.conv3 = nn.Conv2d(16, 24, (3,3), padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv4 = nn.Conv2d(24, 32, (3,3), padding=1)

        self.fc = nn.Linear(32 * input_size, 4)
        """
        """
        
        self.conv1 = nn.Conv2d(4, 16, (5,5), padding=2)
        self.conv2 = nn.Conv2d(16, 24, (5,5), padding=2)
        self.conv3 = nn.Conv2d(24, 32, (3,3), padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv4 = nn.Conv2d(32, 48, (3,3), padding=1)

        self.fc = nn.Linear(48 * input_size, 4)
        """
        
        self.leaky = nn.LeakyReLU(0.1)
        
        
    def forward(self, x): 
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
            
    def __init__(self, map_name='bruce_lee', plot=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Running on : ' + str(self.device))
        
        self.map_name = map_name
        self.plot = plot
        
        self.env = Game(visualize=True, load_map=True, map_name=self.map_name)
        m = self.env.m
        n = self.env.n
        self.input_size = ((m) // 2) * ((n) // 2)
        
        self.dqn = DQN(self.input_size)
        self.load_model()
        self.dqn.to(device=self.device)
        
    def load_model(self) :
        self.dqn.load_state_dict(torch.load('model/' + self.map_name, map_location=self.device))
        self.dqn.eval()

    def preprocess_state(self, state):
        return [torch.tensor(np.array([state[0]]), dtype=torch.float32)]
    
    def choose_action(self, state):
        return torch.argmax(self.dqn(state[0].to(self.device)).cpu()).numpy()
            
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
        time.sleep(0.1)
        screens = []
        
        if self.map_name == 'bruce_lee' :
            mon = {'left': 1310, 'top': 792, 'width': 187, 'height': 178} #bruce lee
        else :
            mon = {'left': 1310, 'top': 834, 'width': 172, 'height': 135} #empty map
            #mon = {'left': 1310, 'top': 834, 'width': 172, 'height': 135} #one enemy
            #mon = {'left': 1310, 'top': 834, 'width': 172, 'height': 135} #test map4 (one enemy)
            #mon = {'left': 1310, 'top': 834, 'width': 172, 'height': 135} #three enemies (test map 2)
        with mss() as sct:
            time.sleep(0.1)
            while not done:
                screenShot = sct.grab(mon)
                img = Image.frombytes(
                    'RGB', 
                    (screenShot.width, screenShot.height), 
                    screenShot.rgb, 
                )
                screens.append(img)
                if self.plot :
                    plt.imshow(img)
                    plt.show()

                action = self.choose_action(state)
                next_state, reward, done = self.env.do_action(action)
                next_state = self.preprocess_state(next_state)
                actions.append(self.action_to_letter(action))
                state = next_state

                time.sleep(0.5)
            
        print(actions)
        fp_out = "model/results/" + self.map_name + ".gif"
        
        img = screens[0]
        img.save(fp=fp_out, format='GIF', append_images=screens[1:],
                 save_all=True, duration=500, loop=0)
        
        

if __name__ == '__main__':
    agent = DQNRoomSolver('empty_map', plot=False)
    agent.run()
    #agent.env.close()
