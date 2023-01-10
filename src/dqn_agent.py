
import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

class DQNAgent:
    """
    Deep Q-Network (DQN) Agent for Reinforcement Learning.
    This agent uses a neural network to approximate the Q-value function.
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000) # Replay buffer
        self.gamma = 0.95    # Discount rate
        self.epsilon = 1.0   # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        """
        Builds the neural network model for the DQN agent.
        """
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(24, activation=\'relu\'),
            Dense(24, activation=\'relu\'),
            Dense(self.action_size, activation=\'linear\')
        ])
        model.compile(loss=\'mse\', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """
        Stores a (state, action, reward, next_state, done) tuple in the replay buffer.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Chooses an action based on an epsilon-greedy policy.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        """
        Trains the model using a random sample from the replay buffer.
        """
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """
        Loads model weights from a file.
        """
        self.model.load_weights(name)

    def save(self, name):
        """
        Saves model weights to a file.
        """
        self.model.save_weights(name)

if __name__ == "__main__":
    # Example usage (simplified environment for demonstration)
    print("Initializing DQN Agent...")
    state_size = 4  # Example: position, velocity, angle, angular velocity
    action_size = 2 # Example: move left, move right
    agent = DQNAgent(state_size, action_size)

    # Simulate a simple environment interaction
    print("Simulating environment interaction...")
    episodes = 10
    for e in range(episodes):
        state = np.random.rand(1, state_size) # Initial random state
        total_reward = 0
        for time_step in range(50):
            action = agent.act(state)
            
            # Simulate next state and reward
            next_state = np.random.rand(1, state_size)
            reward = 1 if action == 0 else -1 # Dummy reward
            done = bool(time_step == 49) # Episode ends after 50 steps
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                print(f"Episode: {e+1}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
                break
        agent.replay(32) # Train after each episode

    print("DQN Agent simulation complete.")
    # agent.save("dqn_model.weights.h5") # Example of saving the model
