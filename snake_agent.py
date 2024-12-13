#snake_agent.py
import numpy as np
from collections import deque
import random as rd
import torch
import torch.nn as nn
import torch.optim as optim

class SnakeAgent(nn.Module):
    def __init__(self, state_size, action_size, matrix_size, indicator_size, gamma, learning_rate, load_model = None, device = 'cpu'):
        super(SnakeAgent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.matrix_size = matrix_size
        self.indicator_size = indicator_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.device = device

        self.memory = deque(maxlen=10_000)
        self.batch_size = 256

        self.model = self.build_model().to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        if load_model:
            print("Loading model weights...")
            self.model.load_state_dict(torch.load("Agents/trained_agent.pth"))

    def build_model(self):
        self.matrix_net = nn.Sequential(
            nn.Linear(self.matrix_size, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 32),
        ).to(self.device) 
        
        self.indicator_net = nn.Sequential(
            nn.Linear(self.indicator_size, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 32),
        ).to(self.device) 
        
        combined_size = 32 + 32
        
        final_net = nn.Sequential(
            nn.Linear(combined_size, 64),
            nn.ReLU(), 
            nn.Linear(64, 32),
            nn.ReLU(), 
            nn.Linear(32, self.action_size), 
        ).to(self.device) 

        return final_net
    
    def forward(self, state):

        matrix_part, indicator_part = state[:self.matrix_size].to(self.device), state[self.matrix_size:].to(self.device)


        matrix_out = self.matrix_net(matrix_part)
        indicator_out = self.indicator_net(indicator_part)

        combined_out = torch.cat((matrix_out, indicator_out))
        return self.model(combined_out)


    def get_exploration_options(self, state):
        is_left_viable  = int(state[49] == 0 and state[50] == 0)
        is_ahead_viable = int(state[51] == 0 and state[52] == 0)
        is_right_viable = int(state[53] == 0 and state[54] == 0)
        return [is_left_viable, is_ahead_viable, is_right_viable]

    def get_action(self, state, epsilon):
        '''
        if np.random.rand() <= epsilon:  # exploration
            viable_options = self.get_exploration_options(state)
            valid_actions = [i for i, viable in enumerate(viable_options) if viable]
            if valid_actions:
                return np.random.choice(valid_actions)
            else:
                return np.random.randint(self.action_size)  # Fallback if no valid action is found
        '''
        if np.random.rand() <= epsilon:
            return np.random.randint(self.action_size)
        state = torch.FloatTensor(state).to(self.device) 
        q_values = self.forward(state)
        
        return torch.argmax(q_values).item()
    
    def train_model(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        action = torch.LongTensor([action]).to(self.device)
        done = torch.FloatTensor([done]).to(self.device)

        q_values = self.forward(state)
        
        current_q_value = q_values.gather(0, action)

        next_q_values = self.forward(next_state)
        max_next_q_value = torch.max(next_q_values).detach()

        target_q_value = reward + (1 - done) * self.gamma * max_next_q_value

        loss = self.loss_fn(current_q_value, target_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = rd.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            self.train_model(state, action, reward, next_state, done)
