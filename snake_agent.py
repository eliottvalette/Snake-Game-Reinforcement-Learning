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
            self.model.load_state_dict(torch.load("Agents/trained_agent.pth", weights_only=True))

    def build_model(self):
        self.matrix_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0),  # (1, 11, 11) -> (32, 9, 9)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),  # (32, 9, 9) -> (64, 7, 7)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # (64, 7, 7) -> (64, 3, 3)
            nn.Flatten()
        ).to(self.device)
        
        self.indicator_net = nn.Sequential(
            nn.Linear(self.indicator_size, 64),
            nn.Linear(64, 64)
        ).to(self.device) 
        
        combined_size = 64 + 576
        
        final_net = nn.Sequential(
            nn.Linear(combined_size, 256),
            nn.ReLU(), 
            nn.Linear(256, 128),
            nn.ReLU(), 
            nn.Linear(128, 128),
            nn.ReLU(), 
            nn.Linear(128, 32),
            nn.ReLU(), 
            nn.Linear(32, self.action_size), 
        ).to(self.device) 

        return final_net
    
    def forward(self, state):

        matrix_part, indicator_part = state[:self.matrix_size].to(self.device), state[self.matrix_size:].to(self.device)
        matrix_out = self.matrix_net(matrix_part.reshape(1, 1, 11, 11)).squeeze(0)
        indicator_out = self.indicator_net(indicator_part)

        fork_1 = rd.random()
        fork_2 = rd.random()
        if fork_1 < 0.4 and fork_2 < 0.2:
            matrix_out = torch.zeros_like(matrix_out)
        elif fork_1 < 0.4 :
            indicator_out = torch.zeros_like(indicator_out)


        combined_out = torch.cat((matrix_out, indicator_out))
        return self.model(combined_out)


    def get_exploration_options(self, state):
        is_left_viable  = int(state[121] == 0 and state[122] == 0)
        is_ahead_viable = int(state[123] == 0 and state[124] == 0)
        is_right_viable = int(state[125] == 0 and state[126] == 0)
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

        # Forward pass for Q-values
        q_values = self.forward(state)
        current_q_value = q_values.gather(0, action)
        next_q_values = self.forward(next_state)
        max_next_q_value = torch.max(next_q_values).detach()
        target_q_value = reward + (1 - done) * self.gamma * max_next_q_value

        # Loss and optimization
        loss = self.loss_fn(current_q_value, target_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Collect metrics
        approx_kl = (q_values - current_q_value).mean().item()
        entropy_loss = -torch.mean(q_values * torch.log_softmax(q_values, dim=0)).item()
        value_loss = self.loss_fn(current_q_value, target_q_value).item()
        std = q_values.std().item()
        clip_fraction = torch.mean((torch.abs(current_q_value - target_q_value) > 0.2).float()).item()

        metrics = {
            "approx_kl": approx_kl,
            "entropy_loss": entropy_loss,
            "value_loss": value_loss,
            "std": std,
            "clip_fraction": clip_fraction,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "loss": loss.item(),
        }

        return metrics

    

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = rd.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            self.train_model(state, action, reward, next_state, done)
