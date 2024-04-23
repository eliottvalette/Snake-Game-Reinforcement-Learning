#snake_train.py
import numpy as np
import pygame
import tensorflow as tf
from snake_agent import SnakeAgent
from snake_game import SnakeGame

# Hyperparameters
EPISODES = 500 # 1 min per 30 episodes
GAMMA = 0.95
ALPHA = 0.01

# Create the game environment
env = SnakeGame()

# Create the Q-learning agent
agent = SnakeAgent(
    state_size=11,
    action_size=3,
    gamma=GAMMA,
    learning_rate=ALPHA,
)

# Load the trained agent's weights
agent.model.load_weights("Reinforcement_Learning/trained_agent.weights.h5")

# Training loop
for episode in range(EPISODES):
    max_steps = 200
    env.n = 4 + 2*(episode//((EPISODES//3)+1)) # 4 - 6 - 8
    epsilon = max(0.01, 0.99 * (0.995 ** episode))
    print('epsilon : ', epsilon)
    env.reset(max_steps,env.n)
    done = False
    steps = 0
    total_reward = 0

    while not done :
        state=env.get_state()
        print('current state : ',state)

        # Select an action using the agent's policy
        action = agent.get_action(state,epsilon)

        # Take the action and observe the next state, reward, and done flag
        next_state, reward, done, _ = env.step(action)

        # Update the agent's Q-table
        reshaped_state = np.reshape(state, (1, agent.state_size))
        reshaped_next_state = np.reshape(next_state, (1, agent.state_size))
        agent.train_model(reshaped_state, action, reward, reshaped_next_state, done)

        # Update the state
        total_reward += reward
        steps += 1
        env.render(False,total_reward,15)
    
    print(f"Episode: {episode}, Steps: {steps}, Reward: {total_reward}")

# Save the trained agent
agent.model.save("Reinforcement_Learning/trained_agent.h5")

# Save the trained agent's weights
agent.model.save_weights("Reinforcement_Learning/trained_agent.weights.h5")

# Print the model's architecture
agent.model.summary()

if input("Test ?")=='y':
    # Test the agent
    for episode in range(50):
        env.reset(1000,8)
        state = env.get_state()
        done = False
        steps = 0
        total_reward = 0
        
        while not done:
            action = agent.get_action(state, epsilon=0)  # No exploration during testing
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
            steps += 1
            env.render(True,total_reward,12)
            
        print(f"Steps: {steps}, Total Reward: {total_reward}")

