# Reinforcement Learning Agents

A collection of Reinforcement Learning agents implemented in Python using libraries like OpenAI Gym and PyTorch/TensorFlow. Includes implementations of Q-learning, DQN, and policy gradient methods.

## Project Overview

This repository serves as a practical exploration into the field of Reinforcement Learning (RL). It contains implementations of various fundamental RL algorithms, from classic methods like Q-learning to more advanced deep reinforcement learning techniques such as Deep Q-Networks (DQN) and Policy Gradient methods (e.g., REINFORCE, A2C). The agents are tested and demonstrated using environments from OpenAI Gym, providing a standardized platform for evaluating performance.

## Features

-   **Classic RL Algorithms**: Q-learning, SARSA, Monte Carlo methods.
-   **Deep Reinforcement Learning**: DQN, Double DQN, Dueling DQN.
-   **Policy Gradient Methods**: REINFORCE, Actor-Critic (A2C).
-   **OpenAI Gym Integration**: Agents are designed to interact with various Gym environments.
-   **Visualization**: Tools for visualizing agent performance and learning curves.

## Getting Started

### Prerequisites

-   Python 3.8+
-   PyTorch or TensorFlow
-   OpenAI Gym
-   NumPy
-   Matplotlib

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Parin1946/reinforcement-learning-agents.git
    cd reinforcement-learning-agents
    ```
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

Each algorithm typically has its own script. For example, to train a DQN agent on the CartPole environment:

```bash
python dqn_cartpole.py
```

Refer to the individual algorithm directories for specific usage instructions.

## Contributing

Contributions are welcome! Feel free to open issues, suggest improvements, or submit pull requests for new algorithms or environments.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
