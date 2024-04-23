#snake_agent.py
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

class SnakeAgent:
    def __init__(self, state_size, action_size, gamma, learning_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.learning_rate = learning_rate

        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(self.state_size,)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def get_action(self, state, epsilon):
        if np.random.rand() <= epsilon: # exploration
            return np.random.randint(self.action_size)
        q_values = self.model.predict(np.reshape(state, (1, self.state_size))) # exploitation
        return np.argmax(q_values[0])

    def train_model(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target += self.gamma * np.max(self.model.predict(np.reshape(next_state, (1, self.state_size)))[0]) # Bellman Equation

        target_vec = self.model.predict(np.reshape(state, (1, self.state_size)))
        target_vec[0][action] = target
        self.model.fit(np.reshape(state, (1, self.state_size)), target_vec, epochs=1, verbose=0) # Train the model