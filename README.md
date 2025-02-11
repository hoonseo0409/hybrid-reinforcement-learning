# Game Automation with Hybrid Reinforcement Learning

This repository implements a hybrid reinforcement learning system for game automation that combines on-policy learning from real-time gameplay with off-policy learning from human demonstrations. The model can handle both continuous (mouse coordinates) and discrete (keyboard stroke) action spaces through a novel dual-headed architecture.

## Key Features

- Hybrid learning from both real-time gameplay and offline demonstrations
- Handles continuous and discrete action spaces simultaneously
- Temporal state modeling using frame sequences
- Combined value and Q-function learning
- Behavioral cloning from human demonstrations

## Model Architecture

### State Representation
The state is represented as a sequence of preprocessed game frames to capture temporal relationships. We use:

- Frame sequence length: 4 frames
- Frame preprocessing: Resizing to 84x84 pixels and grayscale conversion
- LSTM layers to process the temporal sequence

### Action Space
The model handles a hybrid action space consisting of:

1. Discrete action types:
   - Mouse clicks
   - Mouse drags  
   - Key combinations
   - Null action (no action)

2. Continuous action parameters:
   - Normalized mouse coordinates [-1, 1]
   - Drag start/end positions

### Neural Network Architecture

The model uses a hybrid architecture with shared feature extraction:

```
Input Frames → CNN → LSTM → Shared Features
                              ↙           ↘
                    Policy Network    Value/Q Networks
                     ↙        ↘
            Action Type    Action Parameters
```

- **CNN Layers**: Extract spatial features from frames
- **LSTM Layer**: Process temporal relationships
- **Policy Network**: Two heads for discrete and continuous actions
- **Value Network**: Estimates state values
- **Q Network**: Evaluates state-action pairs

## Learning Approach

### 1. Behavioral Cloning
We use behavioral cloning (BC) to learn from human demonstrations [1]. This provides a good initialization for the policy by mimicking expert behavior.

### 2. Value & Q-Function Learning
- Value function estimates long-term rewards for states
- Q-function evaluates specific state-action pairs
- Dual learning helps balance exploration (value) and exploitation (Q) [2]

### 3. Online Learning
- PPO (Proximal Policy Optimization) for on-policy updates [3]
- GAE (Generalized Advantage Estimation) for reduced variance
- Experience replay buffer for sample efficiency

## Implementation

### Requirements
- TensorFlow 2.x
- OpenCV
- NumPy
- PyAutoGUI

### Usage

1. Record demonstrations:
```python
from reinforcement import record_gameplay

# Define action space
action_space = {
    'keys': ['w', 'a', 's', 'd', 'space'],
    'mouse': {
        'min_pos': [0, 0],
        'max_pos': window_size,
        'actions': ['click', 'drag']
    }
}

# Start recording
recorder = record_gameplay(agent, action_space)
```

2. Train the model:
```python
from reinforcement import train_rl_model

# Implement reward function
class MyRewardFunction:
    def calculate_reward(self, current_frame, previous_frames, action):
        # Return reward based on game state
        pass

# Train model
model = train_rl_model(
    agent=game_agent,
    demo_dir="recordings",
    action_space=action_space,
    reward_function=MyRewardFunction()
)
```

### Custom Implementation Requirements

#### 1. Reward Function
Implement the `RewardFunction` class with:
```python
def calculate_reward(self, current_frame, previous_frames, action):
    # Return numerical reward
    pass
```

#### 2. Worker Class
Implement the `Worker` class to execute actions:
```python
def execute_action(self, action_type, **params):
    # Execute game inputs
    pass
```

## References

[1] Pomerleau, D. A. (1991). Efficient Training of Artificial Neural Networks for Autonomous Navigation. Neural Computation, 3(1), 88-97.

[2] Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533.

[3] Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. arXiv preprint arXiv:1707.06347.

## License
MIT License