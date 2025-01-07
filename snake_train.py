#snake_train.py
import numpy as np
import random as rd
import pygame
import torch
import time
from snake_agent import SnakeAgent
from snake_game import SnakeGame
import matplotlib.pyplot as plt
from print_format import print_box
from collections import deque

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
device = 'cpu'
print("Using ", device)

# Hyperparameters
EPISODES = 0 # 6_000
GAMMA = 0.99
ALPHA = 0.003
GLOBAL_N = 11


rd.seed(42)
np.random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

RENDER = False 
LOAD_MODEL = True

# Create the game environment
env = SnakeGame()

# Create the Q-learning agent
agent = SnakeAgent(
    state_size=63,
    action_size=3,
    matrix_size = 11,
    indicator_size = 13,
    gamma = GAMMA,
    learning_rate = ALPHA,
    load_model = LOAD_MODEL,
    device=device
)

# Plotting variables
plot_scores = []
plot_steps = []
plot_mean_scores = []
plot_mean_steps = []
plot_mean_reward = []
total_score = 0
total_steps = 0
total_reward_to_plot = 0

# Tracking variables
training_metrics = {key: deque(maxlen=50) for key in ["approx_kl", "entropy_loss", "value_loss", "std", "learning_rate", "loss"]}

def moving_average(data, window_size):
    return [np.mean(data[max(0, i - window_size + 1):i + 1]) for i in range(len(data))]

# Training loop
for episode in range(EPISODES):
    max_steps = 1000
    env.n = GLOBAL_N
    epsilon = max(0.00, 0.99 ** episode)
    env.epsilon = epsilon

    # print(f'Randomness : {epsilon*100:.2f}%')

    length = 5

    env.reset(max_steps = max_steps, N = env.n, length= length)
    done = False
    steps = 0
    total_reward = 0

    while not done:
        state = env.get_state()

        action = agent.get_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)

        metrics = agent.train_model(state, action, reward, next_state, done)
        for key, value in metrics.items():
            training_metrics[key].append(value)

        env.render(RENDER, total_reward, 2000)
        total_reward += reward
        steps += 1
    
    score = len(env.snake)    
    plot_scores.append(score)
    plot_steps.append(steps)
    total_score += score
    total_steps += steps
    total_reward_to_plot += total_reward
    mean_score = (total_score / (episode + 1))
    mean_steps = (total_steps / (episode + 1))
    mean_reward = (total_reward_to_plot / (episode + 1))
    plot_mean_scores.append(mean_score)
    plot_mean_steps.append(mean_steps)
    plot_mean_reward.append(mean_reward)

    # Log metrics
    if episode % 5 == 4:
        mean_metrics = {key: np.mean([v.cpu().detach().numpy() if isinstance(v, torch.Tensor) else v for v in values]) for key, values in training_metrics.items()} 
        last_score = plot_scores[-50:]
        last_steps = plot_steps[-50:]
        print_box(episode + 1, mean_metrics, last_score, last_steps, epsilon)

    # Save model periodically    
    if episode % 1000 == 999:

        # Calculate moving averages
        window_size = 100
        plot_MAE_scores = moving_average(plot_scores, window_size)
        plot_MAE_steps = moving_average(plot_mean_steps, window_size)
        torch.save(agent.model.state_dict(), f"Agents/trained_agent_epoch_{episode+1}.pth")

        # Temporary plot
        episodes = range(len(plot_scores))
        plt.figure(figsize=(10, 5))
        plt.scatter(episodes, plot_scores, label='Scores', color='blue', s=10, alpha=0.5)
        plt.plot(episodes, plot_MAE_scores, label='Mean Scores', color='red')
        plt.plot(episodes, plot_MAE_steps, label='Mean Steps', color='green')
        plt.xlabel('Episodes')
        plt.ylabel('Score')
        plt.title('Training Progress')
        plt.legend()
        plt.savefig(f'Training graphs/graph_temp_{len(plot_scores)}.pdf')
        plt.close()


    # print(f'Episode: {episode + 1}, Total Reward: {total_reward}, Steps: {steps}, Length: {len(env.snake)}')

# Plotting section
episodes = range(EPISODES)
plt.figure(figsize=(10, 5))
plt.scatter(episodes, plot_scores, label='Scores', color='blue', s=10, alpha=0.5)
plt.plot(episodes, plot_mean_scores, label='Mean Scores', color='red')
plt.plot(episodes, plot_mean_steps, label='Mean Steps', color='green')
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
        # print(f'Randomness : {epsilon*100:.2f}%')

        env.reset(max_steps = max_steps, N = env.n, length = 5)
        done = False
        steps = 0
        total_reward = 0

        while not done:
            state = env.get_state()

            action = agent.get_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.train_model(state, action, reward, next_state, done)

            env.render(True, total_reward, 15)

            total_reward += reward
            steps += 1
        
        # print(f'Episode: {episode + 1}, Total Reward: {total_reward:7}, Steps: {steps:3}')

plt.show()