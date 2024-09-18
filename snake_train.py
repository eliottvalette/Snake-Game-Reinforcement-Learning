#snake_train.py
import numpy as np
import random as rd
import pygame
import torch
import time
from snake_agent import SnakeAgent
from snake_game import SnakeGame
import matplotlib.pyplot as plt

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
device='cpu'
print("Using ", device)

# Hyperparameters
EPISODES = 4_000
GAMMA = 0.99
ALPHA = 0.005
GLOBAL_N = 21


rd.seed(42)
np.random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

RENDER = False

# Create the game environment
env = SnakeGame()

# Create the Q-learning agent
agent = SnakeAgent(
    state_size=63,
    matrix_size = 49,
    indicator_size = 13,
    action_size=3,
    gamma = GAMMA,
    learning_rate = ALPHA,
    load_model = True,
    device=device
)

plot_scores = []
plot_mean_scores = []
plot_mean_steps = []
plot_mean_reward = []
total_score = 0
total_steps = 0
total_reward_to_plot = 0

# Training loop
for episode in range(EPISODES):
    max_steps = 1000
    env.n = GLOBAL_N
    epsilon = max(0.01, (0.999 ** episode))
    env.epsilon = epsilon

    print(f'Randomness : {epsilon*100:.2f}%')

    length = 5 # rd.randint(0, 5)

    env.reset(max_steps = max_steps, N = env.n, length= length)
    done = False
    steps = 0
    total_reward = 0

    while not done:
        state = env.get_state()

        action = agent.get_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.train_model(state, action, reward, next_state, done)

        env.render(RENDER, total_reward, 20)

        total_reward += reward
        steps += 1
    
    score = len(env.snake)    
    plot_scores.append(score)
    total_score += score
    total_steps += steps
    total_reward_to_plot += total_reward
    mean_score = (total_score / (episode + 1))
    mean_steps = (total_steps / (episode + 1))
    mean_reward = (total_reward_to_plot / (episode + 1))
    plot_mean_scores.append(mean_score)
    plot_mean_steps.append(mean_steps)
    plot_mean_reward.append(mean_reward)
    
    if episode% 300 == 299:
        agent.replay()
        
    if episode % 2000 == 1999:
        torch.save(agent.model.state_dict(), f"Agents/trained_agent_epoch_{episode+1}.pth")

    print(f'Episode: {episode + 1}, Total Reward: {total_reward}, Steps: {steps}, Length: {len(env.snake)}')

# Plotting section
episodes = range(EPISODES)
plt.figure(figsize=(10, 5))
plt.plot(episodes, plot_scores, label='Scores')
plt.plot(episodes, plot_mean_scores, label='Mean Scores')
plt.plot(episodes, plot_mean_steps, label='Mean Steps')
plt.xlabel('Episodes')
plt.ylabel('Score')
plt.title('Training Progress')
plt.legend()
plt.savefig('Training graphs/graph.pdf')

if input("Test ? (y or n) : ") == 'y':
    agent.model.eval()
    for episode in range(10):
        # Test the agent
        max_steps = 500000
        env.n = GLOBAL_N
        epsilon = 0.0
        env.epsilon = epsilon
        print(f'Randomness : {epsilon*100:.2f}%')

        env.reset(max_steps = max_steps, N = env.n, length = 5)
        done = False
        steps = 0
        total_reward = 0

        while not done:
            state = env.get_state()

            action = agent.get_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.train_model(state, action, reward, next_state, done)

            env.render(True, total_reward, 150)

            total_reward += reward
            steps += 1
        
        print(f'Episode: {episode + 1}, Total Reward: {total_reward:7}, Steps: {steps:3}')

plt.show()