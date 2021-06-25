import random
import numpy as np
import os
import torch as T
from torch.utils.tensorboard import SummaryWriter

from model import *
from constants import *
from utils import ReplayBuffer

class Agent():
    def __init__(self, input_dims, n_actions, seed, agent_mode=SIMPLE, network_mode=SIMPLE, test_mode=False, batch_size=64, n_epochs=5, 
                 update_every=5, lr=0.0005, fc1_dims=64, fc2_dims=64, gamma=0.99, epsilon=1.0, eps_end=0.01, eps_dec=0.995, 
                 max_mem_size=1_00_000, tau=1e-3):
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.seed = random.seed(seed)
        
        self.agent_mode=agent_mode
        self.network_mode=network_mode
        self.test_mode=test_mode
        
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.update_every = update_every
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.mem_size = max_mem_size
        self.tau = tau
        
        # For naming purpose
        agent_ = '{}-'.format(self.agent_mode) if self.agent_mode!=SIMPLE else ''
        network_ = '{}-'.format(self.network_mode) if self.network_mode!=SIMPLE else ''
        self.agent_name = f'{agent_}{network_}DQN'.strip()
        
        if network_mode==DUELING:
            self.Q_eval = DuelingDeepQNetwork(input_dims=input_dims, n_actions=n_actions, seed=seed, lr=lr, fc1_dims=fc1_dims, fc2_dims=fc2_dims)
            if not test_mode:
                self.Q_next = DuelingDeepQNetwork(input_dims=input_dims, n_actions=n_actions, seed=seed, lr=lr, fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        else:
            self.Q_eval = DeepQNetwork(input_dims=input_dims, n_actions=n_actions, seed=seed, lr=lr, fc1_dims=fc1_dims, fc2_dims=fc2_dims)
            if not test_mode:
                self.Q_next = DeepQNetwork(input_dims=input_dims, n_actions=n_actions, seed=seed, lr=lr, fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        
        if not test_mode:
            
            self.tensorboard_step = 0
            self.tensorboard_writer = SummaryWriter(log_dir=f'logs/{self.agent_name}')
            
            self.update_cntr = 0
            self.memory = ReplayBuffer(max_mem_size, batch_size, n_actions, seed)
    
    def save_model(self):
        # Create models folder
        if not os.path.isdir(f'models/{self.agent_name}'):
            os.makedirs(f'models/{self.agent_name}')
        T.save(self.Q_eval.state_dict(), f'./models/{self.agent_name}/{self.agent_name}_EVAL.pth')
        T.save(self.Q_next.state_dict(), f'./models/{self.agent_name}/{self.agent_name}_TARGET.pth')
    
    def load_model(self):
        if os.path.exists(f'./models/{self.agent_name}/{self.agent_name}_EVAL.pth'):
            self.Q_eval.load_state_dict(T.load(f'./models/{self.agent_name}/{self.agent_name}_EVAL.pth', map_location=T.device(self.Q_eval.device)))
        if not self.test_mode:
            if os.path.exists(f'./models/{self.agent_name}/{self.agent_name}_TARGET.pth'):
                self.Q_next.load_state_dict(T.load(f'./models/{self.agent_name}/{self.agent_name}_TARGET.pth', map_location=T.device(self.Q_eval.device)))
            
    def on_epsiode_end(self, reward_avg, reward_min, reward_max, n_steps, i_steps):
        if not self.test_mode:
            self.tensorboard_writer.add_scalar('Reward Avg.', reward_avg, self.tensorboard_step)
            self.tensorboard_writer.add_scalar('Reward Min.', reward_min, self.tensorboard_step)
            self.tensorboard_writer.add_scalar('Reward Max.', reward_max, self.tensorboard_step)
            self.tensorboard_writer.add_scalar('Total Steps', n_steps, self.tensorboard_step)
            self.tensorboard_writer.add_scalar('Steps per Episode', i_steps, self.tensorboard_step)
            self.tensorboard_writer.add_scalar('Epsilon', self.epsilon, self.tensorboard_step)
        
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every update_cntr time steps.
        self.update_cntr = (self.update_cntr + 1) % self.update_every
        if self.update_cntr == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)
    
    def replace_target_network(self):
        if self.update_every != 0 and self.update_cntr % self.update_every == 0:
            # Soft Update
            for target_param, local_param in zip(self.Q_next.parameters(), self.Q_eval.parameters()):
                target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
    
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = observation[np.newaxis,:] # Add an axis to pass to model
            self.Q_eval.eval()
            with T.no_grad():
                Q = self.Q_eval.forward(T.from_numpy(state).to(self.Q_eval.device))
            self.Q_eval.train()
            action = T.argmax(Q).item()
        else:
            action = np.random.choice(np.arange(self.n_actions))

        return action
    
    def epsilon_decay(self):
        self.epsilon = max(self.epsilon*self.eps_dec, self.eps_end)
    
    def learn(self, samples):
        states, actions, rewards, next_states, dones = samples
        
        if self.agent_mode == DOUBLE:
            # Double DQN Approach
            self.Q_eval.eval()
            with T.no_grad():
                # Q_Eval over next states to fetch max action arguement to pass to q_next
                q_pred = self.Q_eval.forward(next_states).to(self.Q_eval.device)
                max_actions = T.argmax(q_pred, dim=1).long().unsqueeze(1)
                # Q_Target over next states from actions will be taken based on q_pred's max_actions
                q_next = self.Q_next.forward(next_states).to(self.Q_eval.device)
            self.Q_eval.train()
            q_target = rewards + \
                        self.gamma*q_next.gather(1, max_actions)*(1.0 - dones)
        else:
            # DQN Approach
            q_target_next = self.Q_next.forward(next_states).to(self.Q_eval.device).detach().max(dim=1)[0].unsqueeze(1)
            q_target = rewards + (self.gamma* q_target_next * (1 - dones))

        # Training
        for epoch in range(self.n_epochs):
            q_eval = self.Q_eval.forward(states).to(self.Q_eval.device).gather(1, actions)
            loss = self.Q_eval.loss(q_eval, q_target).to(self.Q_eval.device)
            self.Q_eval.optimizer.zero_grad()
            loss.backward()
            self.Q_eval.optimizer.step()

        # Replace Target Network
        self.replace_target_network() 