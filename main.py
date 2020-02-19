import gym
import math
import numpy as numpy
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import argparse
import models.random_agent as randAgent
import models.DQN as DQN
import time
#import wandb
import hydra
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torchvision.transforms as T

'''parser=argparse.ArgumentParser()
parser.add_argument('--model',type=str,default='DQN',help='')
parser.add_argument('--env',type=str,default='CartPole-v1',help='')
parser.add_argument('--learn',type=str,default='learn',help='')
parser.add_argument('--epoch',type=int,default=100,help='')
parser.add_argument('--seed',type=int,default=100,help='')
parser.add_argument('--render',type=int,default=1,help='')

args = parser.parse_args()'''

@hydra.main(config_path="conf/config.yaml")
def hydra_set(cfg):
    global confs 
    confs= cfg
    
#set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()
confs = None

def main():
    agent = None
    main_conf = None
    env = gym.make('Breakout-v0')

    def setInitSetting():
        global agent
        global main_conf
        #print(confs)
        agent = DQN.DQN(learn=True,input_type=confs['agent']['type'],obs_size=env.observation_space.shape,\
        batch_size=confs['agent']['batchsize'],action_size=env.action_space.n,eps_min=confs['agent']['eps_min'],eps_dec=confs['agent']['eps_dec'],eps_step=confs['agent']['eps_step'],gamma=confs['agent']['gamma'],optimizer=confs['agent']['optimizer'],capacity=confs['agent']['capacity'])
        wandb.init(config=confs['agent'],project='dqn_agent')

    def setFinishedSetting():
        stepsu=1000
        global agent
        obs = env.reset()
        done = False
        total_reward=0
        n_obs = obs
        for i in range(stepsu):
            if confs['agent']['render']==0:
                time.sleep(0.1)
                env.render()
            action = agent.policy(obs,mode='test')
            n_obs,reward,done,info = env.step(action)
            total_reward+=reward
            if done or i > stepsu:
                print(total_reward,':',i)
                break       
            obs = n_obs   
        env.close()

    def doAction():
        global agent
        stepsu=10000
        for epoch in range(confs['agent']['epoch']):
            obs = env.reset()
            #print(obs.shape)
            done = False
            total_reward=0
            n_obs = obs
            for i in range(stepsu):
                action = agent.policy(obs)
                n_obs,reward,done,info = env.step(action)
                total_reward+=reward
                if confs['agent']['learn']=='learn':
                    agent.learn(obs,action,reward,n_obs)
                if done or i > stepsu:
                    print(total_reward,':',i)
                    wandb.log({'total_reward':total_reward,'epoch':epoch})
                    break       
                obs = n_obs      
    
    setInitSetting()
    doAction()
    setFinishedSetting()

if __name__=='__main__':
    hydra_set()
    main()