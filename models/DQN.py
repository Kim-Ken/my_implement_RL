import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from collections import namedtuple
import random 
import torchvision.transforms as T
import numpy as np
from PIL import Image
import wandb
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

class NoisyLayer(nn.Module):
    def __init__(self,input_feature,output_feature,sigma=0.5):
        super(NoisyLayer,self).__init__()
        self.input_feature = input_feature
        self.output_feature = output_feature
        self.sigma = 0.5

        self.weight = nn.Parameter(torch.Tensor(output_feature,input_feature))
        self.weight_noise = nn.Parameter(torch.Tensor(output_feature,input_feature))
        self.var = nn.Parameter(torch.Tensor(output_feature))
        self.var_noise = nn.Parameter(torch.Tensor(output_feature))
        self.reset_parameter()

    def reset_parameter(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv,stdv)
        self.var.data.uniform_(-stdv,stdv)

        initial_simga = stdv * self.sigma
        self.weight_noise.data.fill_(initial_simga)
        self.var_noise.data.fill_(initial_simga)
        
    def forward(self,input):
        rand_in = self._f(torch.randn(1,self.input_feature,device=device))
        rand_out = self._f(torch.randn(self.output_feature,1,device=device))
        epsilon_w = torch.matmul(rand_out,rand_in)
        epsilon_b = rand_out.squeeze()
        
        w = self.weight + self.weight_noise * epsilon_w
        b = self.var + self.var_noise * epsilon_b

        return F.linear(input,w,b)
    
    def _f(self,x):
        return torch.sign(x) * torch.sqrt(torch.abs(x))

class network(nn.Module):
        def __init__(self,input_type,obs_size,batch_size,action_size,duel,noise):
            super(network,self).__init__()
            conv_put=100
            self.batch_size = batch_size
            self.action_size = action_size
            self.put = 16
            self.duel = duel
            self.noise = noise
            if input_type == 'image':
                #print(obs_size)
                def calc_out_put_size(size,kernel_size=5,stride=2):
                    return (size - (kernel_size - 1) - 1) // stride  + 1
                c_w = calc_out_put_size(calc_out_put_size(calc_out_put_size(40)))
                c_h = calc_out_put_size(calc_out_put_size(calc_out_put_size(52)))
                self.put=c_w*c_h*32
                #print(self.put)

                self.model=nn.Sequential(nn.Conv2d(3,16,kernel_size=5,stride=2),
                nn.ReLU(),
                nn.Conv2d(16,32,kernel_size=5,stride=2),
                nn.ReLU(),
                nn.Conv2d(32,32,kernel_size=5,stride=2),
                nn.ReLU()
                )
            else:
                self.model = nn.Sequential(nn.Linear(input_size,16,kernel_size=5,stride=2),
                nn.Relu()
                )
            if noise == 'T':
                if self.duel =='T':
                    self.value_layer = NoisyLayer(self.put,1)
                    self.action_layer = NoisyLayer(self.put,self.action_size)
                else:
                    self.head = NoisyLayer(self.put,self.action_size)

            else:
                if self.duel =='T':
                    self.value_layer = nn.Linear(self.put,1)
                    self.action_layer = nn.Linear(self.put,self.action_size)
                else:
                    self.head = nn.Linear(self.put,self.action_size)

        def forward(self,state):
            x = self.model(state)
            #print(x.shape)
            if self.duel == 'T':
                a = self.action_layer(x.view(x.size(0),-1))
                v = self.value_layer(x.view(x.size(0),-1))
                return  v + a
            else:
                return self.head(x.view(x.size(0),-1))

class DQN():

    def __init__(self,learn=True,input_type='image',obs_size=(100,200),batch_size=16,action_size=2,\
        eps_min=0.1,eps_dec=0.0001,eps_step=200,gamma=0.999,update_target=2000,optimizer='RMSProp',capacity=10000,\
        duel='T',multi_step=1,noise='T',prop='T',categorical='T',priorized='T'):

        self.q_network = network(input_type,obs_size,batch_size,action_size,duel,noise).to(device)
        self.q_t_network  = network(input_type,obs_size,batch_size,action_size,duel,noise).to(device)
        self.step_num = 1
        self.eps=1.0

        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.eps_step = eps_step
        self.gamma = gamma
        self.update_target = update_target
        self.capacity = capacity

        self.optimizer = optim.RMSprop(self.q_network.parameters())
        self.DQN_memories = ReplayMemory(self.capacity)
        self.action_size = action_size
        self.batch_size = batch_size
        

        if learn:
            pass
        else:
            pass
    
    def policy(self,state,mode='learn'):
        

        if self.step_num % self.update_target == 0:
            print(self.eps)
            self.q_t_network.load_state_dict(self.q_network.state_dict())

        if self.step_num % self.eps_step == 0 and self.eps_min<self.eps:
            self.eps=self.eps-self.eps_dec
        r = random.random()

        self.step_num += 1

        if r > self.eps and mode=='learn':
            tp = np.ascontiguousarray(state.transpose(2,0,1),dtype=np.float32)/255
            tt = torch.from_numpy(tp/255)
            ts = resize(tt).unsqueeze(0).float().to(device)
            #print("a",t.shape)
            with torch.no_grad():
                return self.q_network.forward(ts).max(1)[1].view(1,1)
        else:
            return torch.tensor([[random.randrange(self.action_size)]], device=device)
        
    
    def learn(self,state,action,reward,next_state):
        
        tp = np.ascontiguousarray(state.transpose(2,0,1),dtype=np.float32)/255
        tt = torch.from_numpy(tp/255)
        ts = resize(tt).unsqueeze(0).float().to(device)
        #print(ts.shape)

        ttn = torch.from_numpy(np.ascontiguousarray(state.transpose(2,0,1),dtype=np.float32)/255)
        tsn = resize(ttn).unsqueeze(0).float().to(device)

        self.DQN_memories.push(ts,action,torch.tensor([reward], device=device),tsn)

        if len(self.DQN_memories) < self.batch_size:
            return
        transitions = self.DQN_memories.sample(self.batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
        state_action_values = self.q_network(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values[non_final_mask] = self.q_t_network(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

    # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        #print(loss)
        for param in self.q_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


Transition = namedtuple('Memory',['state','action','reward','next_state'])

class ReplayMemory(object):
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
