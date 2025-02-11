import tensorflow as tf
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
import json
from collections import deque
import cv2

import time
from datetime import datetime
from threading import Thread
from pynput import mouse, keyboard
import utils

from contextlib import ExitStack

class GameStateProcessor:
    """Handles game state processing and management for RL"""
    
    def __init__(self, 
                 agent,
                 target_size: Tuple[int, int] = (84, 84),
                 sequence_length: int = 4,
                 grayscale: bool = True):
        """
        Initialize state processor
        
        Args:
            agent: Game agent instance for screen capture
            target_size: Size to resize frames to
            sequence_length: Number of frames in state sequence
            grayscale: Whether to convert frames to grayscale
        """
        self.agent = agent
        self.target_size = target_size
        self.sequence_length = sequence_length
        self.grayscale = grayscale
        
        # For frame sequence tracking
        self.frame_buffer = deque(maxlen=sequence_length)
        
        # Calculate state shape
        channels = 1 if grayscale else 3
        self.state_shape = (sequence_length, target_size[0], target_size[1], channels)
        
    def get_state_shape(self) -> tuple:
        """Get shape of processed state tensor"""
        return self.state_shape
        
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert raw frame to processed state representation
        
        Args:
            frame: Raw frame from game
            
        Returns:
            Processed frame as numpy array
        """
        # Convert to grayscale if needed
        if self.grayscale and len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
        # Resize
        frame = cv2.resize(
            frame, 
            self.target_size,
            interpolation=cv2.INTER_AREA
        )
        
        # Normalize to [0,1]
        frame = frame.astype(np.float32) / 255.0
        
        # Add channel dimension if grayscale
        if self.grayscale:
            frame = frame[..., np.newaxis]
            
        return frame
        
    def capture_frame(self) -> np.ndarray:
        """Capture current game frame"""
        frame = self.agent.screenshot(
            region=(
                self.agent.pos[0],
                self.agent.pos[1],
                self.agent.size[0],
                self.agent.size[1]
            )
        )
        return frame
        
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process frame and update frame buffer
        
        Args:
            frame: Raw game frame
            
        Returns:
            Current state tensor
        """
        # Preprocess frame
        processed = self.preprocess_frame(frame)
        
        # Add to buffer
        self.frame_buffer.append(processed)
        
        # Return current state
        return self.get_current_state()
        
    def get_current_state(self) -> np.ndarray:
        """Get current state from frame buffer"""
        # Pad buffer with zeros if needed
        while len(self.frame_buffer) < self.sequence_length:
            zeros = np.zeros_like(self.frame_buffer[0])
            self.frame_buffer.appendleft(zeros)
            
        # Stack frames into state tensor
        return np.array(self.frame_buffer)
        
    def get_initial_state(self) -> np.ndarray:
        """Get empty initial state"""
        # Clear buffer
        self.frame_buffer.clear()
        
        # Create empty frame
        channels = 1 if self.grayscale else 3
        empty_frame = np.zeros((*self.target_size, channels))
        
        # Fill buffer with empty frames
        for _ in range(self.sequence_length):
            self.frame_buffer.append(empty_frame)
            
        return self.get_current_state()
        
    def reset(self):
        """Reset processor state"""
        self.frame_buffer.clear()
        
    def add_frame(self, frame: np.ndarray):
        """Add a single frame to the state history"""
        processed = self.preprocess_frame(frame)
        self.frame_buffer.append(processed)
        
    def process_sequence(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Process a sequence of frames into a state
        
        Args:
            frames: List of raw frames
            
        Returns:
            Processed state tensor
        """
        # Reset buffer
        self.frame_buffer.clear()
        
        # Process each frame
        for frame in frames:
            processed = self.preprocess_frame(frame)
            self.frame_buffer.append(processed)
            
        return self.get_current_state()

class RewardFunction:
    """Template for custom reward function"""
    
    def __init__(self):
        pass
        
    def calculate_reward(self, 
                        current_frame: np.ndarray,
                        previous_frames: List[np.ndarray],
                        current_action: Dict,
                        ) -> float:
        """
        Calculate reward based on current and previous states/actions
        
        Args:
            current_frame: Current game frame
            previous_frames: List of previous frames
            current_action: Current action taken
            previous_actions: List of previous actions
            
        Returns:
            float: Calculated reward value
        """
        # Implement custom reward logic here
        return 0.0
    
class HybridPolicyNetwork(tf.keras.Model):
    def __init__(self, state_shape: tuple, action_space: Dict):
        super().__init__()

        # Track which layers should be frozen
        self.trainable_layers = {
            'action_type': True,
            'click': True, 
            'drag': True,
            'key_probs': True,
            'features': False  # CNN+LSTM feature extractor
        }

        self.action_space = action_space
        
        # State dimensions
        self.state_dim = state_shape[0] * state_shape[1] * state_shape[2] * state_shape[3]
        
        # Action dimensions
        self.action_dim = (4 +  # Action type one-hot
                           4 +  # Mouse position (click=2, drag=4, max is 4)
                           len(action_space['keys']))  # Key presses
        
        self.q_input_dim = self.state_dim + self.action_dim
        
        # Feature dimensions from CNN+LSTM
        feature_dim = 512  # Should match LSTM output size
        self.feature_dim = feature_dim

        # Shared feature extractor
        self.conv1 = tf.keras.layers.Conv2D(32, 8, strides=4, activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(64, 4, strides=2, activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(64, 3, strides=1, activation='relu')
        self.flatten = tf.keras.layers.Flatten()

        # LSTM and shared layers
        self.lstm = tf.keras.layers.LSTM(512)
        self.shared = tf.keras.layers.Dense(512, activation='relu')

        # Policy heads
        self.action_type = tf.keras.layers.Dense(4, activation='softmax')
        self.click_mu = tf.keras.layers.Dense(2)
        self.click_logstd = tf.keras.layers.Dense(2)
        self.drag_mu = tf.keras.layers.Dense(4)
        self.drag_logstd = tf.keras.layers.Dense(4)
        self.key_probs = tf.keras.layers.Dense(len(action_space['keys']), activation='sigmoid')

        # Q-network expects [features + action] input
        self.q_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.feature_dim + self.action_dim,)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        # Mark feature extraction layers as not trainable
        self.conv1.trainable = False
        self.conv2.trainable = False 
        self.conv3.trainable = False
        self.lstm.trainable = False
        self.shared.trainable = False

        
    def extract_features(self, state):
        batch_size = tf.shape(state)[0]
        seq_length = tf.shape(state)[1]
        
        # Reshape for conv layers
        x = tf.reshape(state, [-1, state.shape[2], state.shape[3], state.shape[4]])
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        
        # Reshape back for LSTM
        x = tf.reshape(x, [batch_size, seq_length, -1])
        x = self.lstm(x)
        x = self.shared(x)
        return x
        
    def call(self, state, training_mode='online'):
        """Forward pass with flattened output structure"""
        features = self.extract_features(state)
        
        # Scale based on training mode
        scale = 1.0 if training_mode == 'online' else 0.1
        
        # Get outputs with flattened structure for easier access
        outputs = {
            'action_probs': self.action_type(features),
            'click': {
                'mu': self.click_mu(features),
                'logstd': self.click_logstd(features)
            },
            'drag': {
                'mu': self.drag_mu(features),
                'logstd': self.drag_logstd(features)
            },
            'key_probs': tf.nn.sigmoid(self.key_probs(features))
        }
        
        # Apply scaling if in offline mode
        if training_mode == 'offline':
            outputs = tf.nest.map_structure(lambda x: x * scale, outputs)
            
        return outputs
    
        
    def get_q_value(self, state_action_input):
        batch_size = tf.shape(state_action_input)[0]
        state_dim = self.state_dim

        # Split state and action
        state = state_action_input[:, :state_dim]
        action = state_action_input[:, state_dim:]

        # Reshape state for feature extraction
        seq_length = 4
        h = w = 84
        c = 1
        reshaped_state = tf.reshape(state, [batch_size, seq_length, h, w, c])

        # Extract features
        features = self.extract_features(reshaped_state)

        # Reshape features and actions
        feature_dim = tf.shape(features)[1]
        reshaped_features = tf.reshape(features, [batch_size, feature_dim])

        reshaped_action = tf.reshape(action, [batch_size, -1])

        # Concatenate
        q_input = tf.concat([reshaped_features, reshaped_action], axis=1)

        # Q network
        q_value = self.q_net(q_input)
        return q_value

class HybridGameAutomationRL:
    """Complete hybrid RL implementation with unified value learning"""
    
    def __init__(self,
                agent,
                state_processor,
                action_space: Dict,
                reward_function,
                worker_lock,
                learning_rate=0.0001,
                gamma=0.99,
                gae_lambda=0.95,
                clip_ratio=0.2,
                bc_weight=1.0,
                value_weight=0.5,
                batch_size=32):
        
        """Initialize with explicit input sizes for networks"""
        self.agent = agent
        self.state_processor = state_processor
        self.action_space = action_space
        self.reward_function = reward_function
        self.worker_lock = worker_lock
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.bc_weight = bc_weight
        self.value_weight = value_weight
        self.batch_size = batch_size

        # Initialize with proper gradient handling
        tf.config.run_functions_eagerly(True)  # For clearer error messages
        
        # Use names that match actual layer names
        self.frozen_layers = [
            'conv1', 'conv2', 'conv3', 'lstm', 'shared'
        ]

        # Calculate total flattened state dimension
        state_shape = state_processor.get_state_shape()
        self.state_dim = state_shape[0] * state_shape[1] * state_shape[2] * state_shape[3]
        self.action_dim = (4 + 4 + len(action_space['keys']))  # type + mouse + keys
        self.q_input_dim = self.state_dim + self.action_dim

        # Calculate feature dimension after CNN+LSTM
        cnn_output_dim = 512  # This should match LSTM output size
        feature_dim = cnn_output_dim

        # Initialize policy network
        self.policy = HybridPolicyNetwork(state_shape, action_space)

        # Initialize value network
        self.value_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.state_dim,)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        # Q-network takes features + action as input
        self.q_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(feature_dim + self.action_dim,)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        # Initialize optimizers with legacy support
        self.value_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate)
        self.q_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate)
        self.policy_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate) 
        self.bc_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate)

        # Initialize model variables by running a dummy forward pass
        dummy_state = tf.zeros((1, *state_processor.get_state_shape()))
        _ = self.policy(dummy_state)  # Initialize policy variables
        _ = self.value_net(tf.zeros((1, self.state_dim)))  # Initialize value net variables
        _ = self.q_net(tf.zeros((1, feature_dim + self.action_dim)))  # Initialize Q net variables

        # Initialize buffers
        self.offline_buffer = []  # For demonstration data
        self.online_buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'log_probs': []
        }

    def _execute_action(self, worker, action, lock_here=True):
        """Execute action in game environment"""
        with self.worker_lock if lock_here else ExitStack() as workerlock:
            # print(f"Executing action: {action}")
            if action['type'] == 0:  # Click
                pos = action['mouse_pos'].numpy()[0]
                mouse_x = int((pos[0] + 1) / 2 * self.agent.size[0])
                mouse_y = int((pos[1] + 1) / 2 * self.agent.size[1])
                
                if (0 <= mouse_x < self.agent.size[0] and 
                    0 <= mouse_y < self.agent.size[1]):
                    mouse_xy = self.agent.adjust_xy(mouse_x, mouse_y)
                    worker.clickRandom(
                        x=mouse_xy['x'],
                        y=mouse_xy['y'],
                        x_rand_range=0,
                        y_rand_range=0
                    )
                    
            elif action['type'] == 1:  # Drag
                pos = action['mouse_pos'].numpy()[0]
                start_x = int((pos[0] + 1) / 2 * self.agent.size[0])
                start_y = int((pos[1] + 1) / 2 * self.agent.size[1])
                end_x = int((pos[2] + 1) / 2 * self.agent.size[0]) 
                end_y = int((pos[3] + 1) / 2 * self.agent.size[1])
                
                if (0 <= start_x < self.agent.size[0] and
                    0 <= start_y < self.agent.size[1] and
                    0 <= end_x < self.agent.size[0] and
                    0 <= end_y < self.agent.size[1]):
                    
                    start_xy = self.agent.adjust_xy(start_x, start_y)
                    end_xy = self.agent.adjust_xy(end_x, end_y)
                    
                    worker.dragFromToRandom(
                        x1=start_xy['x'], y1=start_xy['y'],
                        x2=end_xy['x'], y2=end_xy['y'],
                        x1_rand_range=0, y1_rand_range=0,
                        x2_rand_range=0, y2_rand_range=0
                    )
                    
            elif action['type'] == 2:  # Key presses
                key_presses = action['key_presses'].numpy()[0]
                pressed_keys = []
                for i, pressed in enumerate(key_presses):
                    if pressed:
                        key = self.action_space['keys'][i]
                        pressed_keys.append(key)
                        
                for key in pressed_keys:
                    self.agent.bring_window_forward()
                    worker.press(
                        key=key,
                        if_special_key='Key.' in key
                    )

    def _store_transition(self, state, action, reward, next_state, action_probs):
        """Store transition with correct output structure"""
        outputs = {
            'action_probs': action_probs,
            'mouse': {
                'click': {'mu': None, 'logstd': None},
                'drag': {'mu': None, 'logstd': None}
            },
            'key_probs': None
        }
        
        # Calculate log probability
        log_prob = self._get_log_prob(action, outputs)
        
        # Store transition in buffer
        self.online_buffer['states'].append(state)
        self.online_buffer['actions'].append(action)
        self.online_buffer['rewards'].append(reward)
        self.online_buffer['next_states'].append(next_state)
        self.online_buffer['log_probs'].append(log_prob)

    def _get_log_prob(self, action, outputs):
        """Calculate log probability of action under current policy with proper None handling"""
        # Get action type probability 
        action_type = int(action['type'].numpy()[0])
        log_prob = tf.math.log(outputs['action_probs'][0, action_type])
        
        if action_type == 0:  # Click
            if action['mouse_pos'] is not None and 'click' in outputs and outputs['click']['mu'] is not None:
                mu = outputs['click']['mu']
                std = tf.exp(outputs['click']['logstd'])
                log_prob += tf.reduce_sum(
                    -0.5 * tf.math.log(2 * np.pi * std**2) \
                    - 0.5 * ((action['mouse_pos'] - mu) / std)**2
                )
                
        elif action_type == 1:  # Drag
            if action['mouse_pos'] is not None and 'drag' in outputs and outputs['drag']['mu'] is not None:
                mu = outputs['drag']['mu']
                std = tf.exp(outputs['drag']['logstd'])
                log_prob += tf.reduce_sum(
                    -0.5 * tf.math.log(2 * np.pi * std**2) \
                    - 0.5 * ((action['mouse_pos'] - mu) / std)**2
                )
                
        elif action_type == 2:  # Keys
            if action['key_presses'] is not None and 'key_probs' in outputs and outputs['key_probs'] is not None:
                # Convert key_presses to tensor if needed
                if not isinstance(action['key_presses'], tf.Tensor):
                    key_presses = tf.convert_to_tensor(action['key_presses'], dtype=tf.float32)
                else:
                    key_presses = action['key_presses']
                    
                # Clip probabilities to avoid numerical issues
                probs = tf.clip_by_value(outputs['key_probs'], 1e-10, 1.0)
                
                # Calculate log probability for binary events
                log_prob += tf.reduce_sum(
                    key_presses * tf.math.log(probs) + \
                    (1 - key_presses) * tf.math.log(1 - probs)
                )
        
        return log_prob

    def train_value_functions(self, combined_batch):
        """Train value and Q functions with proper tensor shape handling
        
        Args:
            combined_batch: Dictionary containing states, actions, rewards, next_states
            
        Returns:
            Dictionary containing value and Q losses
        """
        # Get states in correct shape for value network
        states = tf.cast(combined_batch['states'], tf.float32)
        rewards = tf.cast(combined_batch['rewards'], tf.float32)
        next_states = tf.cast(combined_batch['next_states'], tf.float32)
        
        # Get batch size and dimensions
        batch_size = tf.shape(states)[0]
        seq_length = tf.shape(states)[1]
        h = tf.shape(states)[2]
        w = tf.shape(states)[3] 
        c = tf.shape(states)[4]
        
        # Flatten states for value network
        flattened_states = tf.reshape(states, [batch_size, -1])
        flattened_next_states = tf.reshape(next_states, [batch_size, -1])
        
        # Encode actions    
        encoded_actions = []
        for action in combined_batch['actions']:
            action_vec = self._encode_action(action)
            encoded_actions.append(action_vec)
        
        # Convert actions to tensor with proper shape
        actions = tf.cast(encoded_actions, tf.float32)
        
        # Extract state features using shared CNN+LSTM layers
        state_features = self.policy.extract_features(states)
        next_state_features = self.policy.extract_features(next_states)
        
        # Create Q-network input by concatenating features with actions
        q_input = tf.concat([state_features, actions], axis=1)
        
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            # Only watch value and Q network variables
            tape.watch(self.value_net.trainable_variables)
            tape.watch(self.q_net.trainable_variables)

            # Value predictions
            current_v = self.value_net(flattened_states)
            next_v = self.value_net(flattened_next_states)
            
            # Q-value predictions
            current_q = self.q_net(q_input)
            
            # Ensure reward shape matches
            rewards = tf.reshape(rewards, [batch_size, 1])
            
            # Compute TD targets
            v_targets = rewards + self.gamma * next_v
            q_targets = rewards + self.gamma * tf.stop_gradient(next_v)
            
            # Value losses
            v_loss = tf.reduce_mean(tf.square(v_targets - current_v))
            q_loss = tf.reduce_mean(tf.square(q_targets - current_q))
        
        # Get gradients only for trainable variables
        v_grads = tape.gradient(v_loss, self.value_net.trainable_variables)
        q_grads = tape.gradient(q_loss, self.q_net.trainable_variables)
        
        # Filter out None gradients
        v_grads = [g if g is not None else tf.zeros_like(v) 
                for g, v in zip(v_grads, self.value_net.trainable_variables)]
        q_grads = [g if g is not None else tf.zeros_like(v)
                for g, v in zip(q_grads, self.q_net.trainable_variables)]
                
        del tape

        # Apply non-None gradients
        self.value_optimizer.apply_gradients(
            [(g, v) for g, v in zip(v_grads, self.value_net.trainable_variables)
            if g is not None]
        )
        self.q_optimizer.apply_gradients(
            [(g, v) for g, v in zip(q_grads, self.q_net.trainable_variables)
            if g is not None]
        )
        
        return {
            'v_loss': v_loss,
            'q_loss': q_loss
        }

    def compute_advantages(self, states, actions, rewards, next_states):
        """Compute advantages using both V and Q values with proper tensor reshaping"""
        
        # Convert inputs to tensors if needed
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        
        # Get batch size and original dimensions
        batch_size = tf.shape(states)[0]
        seq_length = tf.shape(states)[1]
        h = tf.shape(states)[2]
        w = tf.shape(states)[3]
        c = tf.shape(states)[4]
        
        # Flatten states for value network
        flattened_states = tf.reshape(states, [batch_size, seq_length * h * w * c])
        flattened_next_states = tf.reshape(next_states, [batch_size, seq_length * h * w * c])
        
        # Get state features using policy's feature extractor
        state_features = self.policy.extract_features(states)
        
        # Encode actions
        encoded_actions = []
        for action in actions:
            action_vec = self._encode_action(action)
            encoded_actions.append(action_vec)
        actions_tensor = tf.convert_to_tensor(encoded_actions, dtype=tf.float32)
        
        # Create Q-network input
        q_input = tf.concat([state_features, actions_tensor], axis=1)
        
        # Get value estimates using flattened states
        v_values = self.value_net(flattened_states)
        q_values = self.q_net(q_input)
        next_v = self.value_net(flattened_next_states)
        
        # Ensure rewards have right shape
        rewards = tf.reshape(rewards, [-1, 1])
        
        # Compute TD errors
        td_errors = rewards + self.gamma * next_v - v_values
        
        # GAE calculation
        advantages = []
        gae = 0
        for delta in tf.squeeze(tf.unstack(td_errors))[::-1]:  # Reverse order
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.append(gae)
        advantages = tf.stack(advantages[::-1])  # Reverse back
        advantages = tf.expand_dims(advantages, -1)
        
        # Add Q-value information
        q_advantages = q_values - v_values
        
        # Combine advantages
        final_advantages = advantages + self.value_weight * q_advantages
        
        # Normalize advantages
        final_advantages = (final_advantages - tf.reduce_mean(final_advantages)) / \
                        (tf.math.reduce_std(final_advantages) + 1e-8)
        
        return final_advantages

    def train_from_gameplay(self, worker, num_episodes: int):
        """Train through actual gameplay using PPO"""
        for episode in range(num_episodes):
            state = self.state_processor.get_initial_state()
            done = False
            episode_reward = 0
            
            while not done:
                # Sample and execute action
                action, outputs = self.select_action(state)
                self._execute_action(worker, action)
                
                # Observe next state and reward
                next_frame = self.state_processor.capture_frame()
                next_state = self.state_processor.process_frame(next_frame)
                reward = self.reward_function.calculate_reward(
                    next_frame, state, action
                )
                episode_reward += reward
                
                # Store transition
                self._store_transition(
                    state, action, reward, next_state,
                    outputs['action_probs']
                )
                
                state = next_state
                
                # Train if buffer is full
                if len(self.online_buffer['states']) >= self.batch_size:
                    self.train_iteration(
                        self._sample_demonstration_batch(self.offline_buffer),
                        self._get_online_batch()
                    )
                    
            print(f"Episode {episode} reward: {episode_reward}")

    def _encode_action(self, action):
        """Encode action into a flat vector with a fixed length of 34"""
        action_vec = []

        # Action type (one-hot encoding, fixed to 4 values)
        action_type = [0.0] * 4
        
        # Handle action type based on different input formats
        if isinstance(action['type'], tf.Tensor):
            action_idx = int(action['type'].numpy()[0] if len(action['type'].shape) > 0 else action['type'].numpy())
        elif isinstance(action['type'], np.ndarray):
            action_idx = int(action['type'].item() if action['type'].size == 1 else action['type'][0])
        else:
            action_idx = int(action['type'])
            
        action_type[action_idx] = 1.0
        action_vec.extend(action_type)

        # Mouse position (click=2D, drag=4D), always 4 elements
        mouse_pos = [0.0] * 4  # Default 4D vector (ensuring fixed size)
        if action['mouse_pos'] is not None:
            if isinstance(action['mouse_pos'], tf.Tensor):
                mouse_pos = action['mouse_pos'].numpy().flatten()[:4].tolist()
            elif isinstance(action['mouse_pos'], np.ndarray):
                mouse_pos = action['mouse_pos'].flatten()[:4].tolist()
            else:
                mouse_pos = list(action['mouse_pos'])[:4]
            mouse_pos = [float(x) for x in mouse_pos]

            # If it's a click (2D), pad to 4D
            while len(mouse_pos) < 4:
                mouse_pos.append(0.0)
        action_vec.extend(mouse_pos)

        # Key presses (variable length, should match `len(action_space['keys'])`)
        key_presses = [0.0] * len(self.action_space['keys'])  # Fixed size key vector
        if action['key_presses'] is not None:
            if isinstance(action['key_presses'], tf.Tensor):
                key_presses_tensor = action['key_presses'].numpy()
                if len(key_presses_tensor.shape) > 1:
                    key_presses_tensor = key_presses_tensor[0]  # Extract first batch item 
                key_presses = key_presses_tensor.tolist()
            elif isinstance(action['key_presses'], np.ndarray):
                key_presses = action['key_presses'].flatten()[:len(self.action_space['keys'])].tolist()
            elif isinstance(action['key_presses'], list):
                key_presses = action['key_presses']
            key_presses = [float(x) for x in key_presses]  # Convert to float

        action_vec.extend(key_presses)

        # Ensure vector has correct length
        expected_length = 34
        if len(action_vec) < expected_length:
            action_vec.extend([0.0] * (expected_length - len(action_vec)))
        elif len(action_vec) > expected_length:
            action_vec = action_vec[:expected_length]

        return action_vec

    def select_action(self, state):
        """Action selection using both policy and Q-values"""
        state = tf.cast(state, tf.float32)
        state = tf.expand_dims(state, 0)
        
        # Get policy distribution
        policy_output = self.policy(state)
        
        # Sample multiple actions from policy
        action_samples = []
        for _ in range(5):  # Sample 5 candidate actions
            action = self._sample_action(policy_output)
            action_samples.append(action)
        
        # Score with Q-values
        q_values = []
        
        # Calculate dimensions
        batch_size = tf.shape(state)[0]
        seq_length = tf.shape(state)[1]
        h = tf.shape(state)[2]
        w = tf.shape(state)[3]
        c = tf.shape(state)[4]
        
        # Flatten state for Q-network input
        flattened_state = tf.reshape(state, [batch_size, seq_length * h * w * c])
        flattened_state_f32 = tf.cast(flattened_state, tf.float32)
        
        for action in action_samples:
            # Encode action
            action_vec = self._encode_action(action)
            
            # Convert to tensor with explicit float32 dtype
            action_tensor = tf.convert_to_tensor([action_vec], dtype=tf.float32)
            
            # Concatenate state and action
            q_input = tf.concat([flattened_state_f32, action_tensor], axis=1)
            
            # Get Q-value using policy's q_net directly
            q = self.policy.get_q_value(q_input)
            q_values.append(q[0])
        
        # Select best action
        best_idx = tf.argmax(q_values)
        selected_action = action_samples[best_idx.numpy()[0]]
        
        return selected_action, policy_output

    def _sample_action(self, outputs):
        """Sample action from current policy with consistent dtypes"""
        # Sample action type
        action_type = tf.random.categorical(
            tf.math.log(outputs['action_probs']), 1
        )[0]
        
        action = {
            'type': tf.cast(action_type, tf.float32),
            'mouse_pos': None,
            'key_presses': None
        }
        
        if action_type == 0:  # Click
            mu = tf.cast(outputs['click']['mu'], tf.float32)
            std = tf.cast(tf.exp(outputs['click']['logstd']), tf.float32)
            action['mouse_pos'] = tf.random.normal(mu.shape, mu, std)
            
        elif action_type == 1:  # Drag
            mu = tf.cast(outputs['drag']['mu'], tf.float32)
            std = tf.cast(tf.exp(outputs['drag']['logstd']), tf.float32)
            action['mouse_pos'] = tf.random.normal(mu.shape, mu, std)
            
        elif action_type == 2:  # Keys
            probs = tf.cast(outputs['key_probs'], tf.float32)
            probs = tf.clip_by_value(probs, 1e-10, 1.0)
            action['key_presses'] = tf.cast(
                tf.random.uniform(probs.shape) < probs,
                tf.float32
            )
            
        return action

    def _get_online_batch(self):
        """Get current online buffer as batch"""
        return {
            'states': np.array(self.online_buffer['states']),
            'actions': self.online_buffer['actions'],
            'rewards': np.array(self.online_buffer['rewards']),
            'next_states': np.array(self.online_buffer['next_states']),
            'log_probs': np.array(self.online_buffer['log_probs'])
        }

    def _sample_demonstration_batch(self, demonstrations):
        """Sample a batch from demonstrations"""
        # Use min to handle case where we have fewer demonstrations than batch_size
        actual_batch_size = min(self.batch_size, len(demonstrations))
        indices = np.random.choice(
            len(demonstrations), 
            actual_batch_size,
            replace=len(demonstrations) < self.batch_size
        )
        
        # Initialize batch with empty lists
        batch = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'action_types': [],
            'mouse_positions': [],
            'key_presses': []
        }
        
        # Fill batch
        for idx in indices:
            demo = demonstrations[idx]
            batch['states'].append(demo['state'])
            batch['next_states'].append(demo['next_state'])
            batch['rewards'].append(demo['reward'])
            
            # Process action components
            action = demo['action']
            action_type = np.zeros(4, dtype=np.float32)
            if isinstance(action['type'], np.ndarray):
                action_idx = np.argmax(action['type'])
            else:
                action_idx = action['type']
            action_type[action_idx] = 1
            
            batch['action_types'].append(action_type)
            batch['actions'].append(action)
            
            # Handle mouse positions - ensure 4D vector (for both click and drag)
            if action['mouse_pos'] is not None:
                if len(action['mouse_pos']) == 2:  # Click
                    mouse_pos = np.zeros(4, dtype=np.float32)
                    mouse_pos[:2] = action['mouse_pos']
                else:  # Drag
                    mouse_pos = action['mouse_pos']
                batch['mouse_positions'].append(mouse_pos)
            else:
                batch['mouse_positions'].append(np.zeros(4, dtype=np.float32))
                
            # Handle key presses
            if action['key_presses'] is not None:
                batch['key_presses'].append(action['key_presses'])
            else:
                batch['key_presses'].append(
                    np.zeros(len(self.action_space['keys']), dtype=np.float32)
                )
        
        # Convert to numpy arrays
        return {
            'states': np.array(batch['states'], dtype=np.float32),
            'actions': batch['actions'],  # Keep as list for complex actions
            'rewards': np.array(batch['rewards'], dtype=np.float32),
            'next_states': np.array(batch['next_states'], dtype=np.float32),
            'action_types': np.array(batch['action_types'], dtype=np.float32),
            'mouse_positions': np.array(batch['mouse_positions'], dtype=np.float32),
            'key_presses': np.array(batch['key_presses'], dtype=np.float32)
        }

    def train_iteration(self, demo_data, online_data):
        """Training iteration with empty demo data handling"""
        
        # Convert online data tensors
        online_states = tf.convert_to_tensor(online_data['states'], dtype=tf.float32)
        
        # Calculate flattened dimensions
        batch_size = tf.shape(online_states)[0]
        seq_length = tf.shape(online_states)[1]
        h = tf.shape(online_states)[2]
        w = tf.shape(online_states)[3]
        c = tf.shape(online_states)[4]
        
        # Flatten states for value network
        flattened_online_states = tf.reshape(online_states, [batch_size, seq_length * h * w * c])
        
        # Check if demo data is empty
        has_demo_data = len(demo_data['states']) > 0 if isinstance(demo_data['states'], list) else demo_data['states'].size > 0
        
        if has_demo_data:
            # Convert demo data tensors
            demo_states = tf.convert_to_tensor(demo_data['states'], dtype=tf.float32)
            
            # Combine data
            combined_batch = {
                'states': tf.concat([demo_states, online_states], axis=0),
                'actions': demo_data['actions'] + online_data['actions'],
                'rewards': tf.concat([
                    tf.convert_to_tensor(demo_data['rewards'], dtype=tf.float32),
                    tf.convert_to_tensor(online_data['rewards'], dtype=tf.float32)
                ], axis=0),
                'next_states': tf.concat([
                    tf.convert_to_tensor(demo_data['next_states'], dtype=tf.float32),
                    tf.convert_to_tensor(online_data['next_states'], dtype=tf.float32)
                ], axis=0)
            }
        else:
            # Use only online data
            combined_batch = {
                'states': online_states,
                'actions': online_data['actions'],
                'rewards': tf.convert_to_tensor(online_data['rewards'], dtype=tf.float32),
                'next_states': tf.convert_to_tensor(online_data['next_states'], dtype=tf.float32)
            }
        
        # Train value functions
        value_losses = self.train_value_functions(combined_batch)
        
        # Compute advantages for PPO using online data
        advantages = self.compute_advantages(
            online_data['states'],
            online_data['actions'],
            online_data['rewards'],
            online_data['next_states']
        )
        
        # PPO update
        with tf.GradientTape() as tape:
            current_policy_output = self.policy(online_states)
            
            # Get current action probabilities
            current_log_probs = tf.stack([
                self._get_log_prob(action, current_policy_output)
                for action in online_data['actions']
            ])
            
            # Calculate probability ratio
            ratio = tf.exp(current_log_probs - online_data['log_probs'])
            
            # Clipped objective
            clip_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantages, clip_ratio * advantages)
            )
            
            # Value function loss
            value_pred = self.value_net(flattened_online_states)
            value_targets = advantages + value_pred
            value_loss = tf.reduce_mean(tf.square(value_pred - value_targets))
            
            # Combined loss
            total_loss = policy_loss + 0.5 * value_loss
        
        # Get trainable variables, excluding frozen layers
        trainable_vars = []
        for var in self.policy.trainable_variables:
            if any(name in var.name for name in self.frozen_layers):
                continue
            trainable_vars.append(var)
        
        # Update policy
        grads = tape.gradient(total_loss, trainable_vars)
        grads = [g if g is not None else tf.zeros_like(v) for g, v in zip(grads, trainable_vars)]
        self.policy_optimizer.apply_gradients(zip(grads, trainable_vars))
        
        return {
            'value_losses': value_losses,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'total_loss': total_loss
        }

    def _behavioral_cloning_step(self, batch):
        """Behavior cloning update on demonstration batch with correct output structure"""
        states = tf.convert_to_tensor(batch['states'])
        action_types = tf.convert_to_tensor(batch['action_types'])
        mouse_positions = tf.convert_to_tensor(batch['mouse_positions'])
        key_presses = tf.convert_to_tensor(batch['key_presses'])
        
        # Get trainable variables before gradient tape
        trainable_vars = [v for v in self.policy.trainable_variables 
                        if not any(name in v.name for name in self.frozen_layers)]
        
        with tf.GradientTape() as tape:
            outputs = self.policy(states)
            
            # Action type loss
            action_loss = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(
                    action_types,
                    outputs['action_probs']
                )
            )
            
            # Mouse position losses
            click_mask = action_types[:, 0]
            drag_mask = action_types[:, 1]
            
            click_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.square(mouse_positions[:, :2] - outputs['click']['mu']),
                    axis=1
                ) * click_mask
            )
            
            drag_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.square(mouse_positions - outputs['drag']['mu']),
                    axis=1
                ) * drag_mask
            )
            
            # Key press loss
            key_mask = action_types[:, 2]
            key_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(
                    key_presses,
                    outputs['key_probs']
                ) * key_mask
            )
            
            # Total BC loss
            total_loss = action_loss + click_loss + drag_loss + key_loss
        
        # Compute gradients only for trainable variables
        grads = tape.gradient(total_loss, trainable_vars)
        
        # Filter and process gradients
        processed_grads = []
        for g, v in zip(grads, trainable_vars):
            if g is not None:
                # Clip gradients to avoid exploding gradients
                g = tf.clip_by_norm(g, 1.0)
                processed_grads.append((g, v))
        
        # Apply processed gradients
        if processed_grads:
            self.bc_optimizer.apply_gradients(processed_grads)
        
        return total_loss

    def train_on_demonstrations(self, demo_dir: str, num_epochs: int = 100):
        """Pre-train using demonstrations"""
        # Load demonstrations
        self.demonstrations = self._load_demonstrations(demo_dir)
        
        # Initial dummy pass to build model
        dummy_batch = self._sample_demonstration_batch(self.demonstrations)
        _ = self._behavioral_cloning_step(dummy_batch)
        
        for epoch in range(num_epochs):
            # Sample batch
            batch = self._sample_demonstration_batch(self.demonstrations)
            
            # Train value functions
            value_losses = self.train_value_functions(batch)
            
            # Behavior cloning
            bc_loss = self._behavioral_cloning_step(batch)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}")
                print(f"Value Loss: {value_losses['v_loss']:.4f}")
                print(f"Q Loss: {value_losses['q_loss']:.4f}") 
                print(f"BC Loss: {bc_loss:.4f}")

    def _load_demonstrations(self, demo_dir):
        """Load demonstration data"""
        demonstrations = []
        
        for session_dir in os.listdir(demo_dir):
            demo_path = os.path.join(demo_dir, session_dir, "demonstration.json")
            if not os.path.exists(demo_path):
                continue
                
            with open(demo_path, 'r') as f:
                demo_data = json.load(f)
                
            processed_demo = self._process_demonstration(demo_data)
            demonstrations.extend(processed_demo)
            
        return demonstrations

    def _process_demonstration(self, demo_data):
        """Process raw demonstration data into training format"""
        processed = []
        state_sequence = []
        
        for frame_data in demo_data:
            # Process frame
            frame = cv2.imread(frame_data['frame'])
            processed_frame = self.state_processor.preprocess_frame(frame)
            
            state_sequence.append(processed_frame)
            if len(state_sequence) < self.state_processor.sequence_length:
                continue
                
            # Create state from sequence
            state = np.array(state_sequence[-self.state_processor.sequence_length:])
            next_state = np.array(state_sequence[-self.state_processor.sequence_length+1:] + 
                                [processed_frame])
            
            # Convert action data
            action = self._format_action(frame_data)
            
            processed.append({
                'state': state,
                'action': action,
                'next_state': next_state,
                'reward': 1.0  # Assume demonstrated actions are good
            })
            
        return processed

    def _format_action(self, action_data):
        """Convert raw action data to model format"""
        action = {
            'type': np.zeros(4),  # One-hot action type
            'mouse_pos': None,
            'key_presses': np.zeros(len(self.action_space['keys']))
        }
        
        if action_data['action_type'] == 'mouse_click':
            action['type'][0] = 1
            pos = np.array(action_data['position'])
            action['mouse_pos'] = pos / self.agent.size
            
        elif action_data['action_type'] == 'mouse_drag':
            action['type'][1] = 1
            start = np.array(action_data['start_position'])
            end = np.array(action_data['end_position'])
            action['mouse_pos'] = np.concatenate([
                start / self.agent.size,
                end / self.agent.size
            ])
            
        elif action_data['action_type'] == 'key_combination':
            action['type'][2] = 1
            for key in action_data['keys']:
                if key in self.action_space['keys']:
                    idx = self.action_space['keys'].index(key)
                    action['key_presses'][idx] = 1
        else:
            action['type'][3] = 1  # Null action
            
        return action

def train_rl_model(agent,
                         demo_dir: str,
                         action_space: Dict,
                         worker,
                         reward_function,
                         worker_lock,
                         num_demo_epochs: int = 50,
                         num_episodes: int = 100,
                         save_dir: str = "model"):
    """Train hybrid model using both demonstrations and gameplay"""
    
    # Initialize state processor
    state_processor = GameStateProcessor(
        agent=agent,
        target_size=(84, 84),
        sequence_length=4
    )
    
    # Create model
    model = HybridGameAutomationRL(
        agent=agent,
        state_processor=state_processor,
        action_space=action_space,
        reward_function=reward_function,
        worker_lock=worker_lock,
    )
    
    # First train on demonstrations
    print("Training on demonstrations...")
    model.train_on_demonstrations(demo_dir, num_epochs=num_demo_epochs)
    
    # Then train through gameplay
    print("Training through gameplay...")
    model.train_from_gameplay(worker, num_episodes)
    
    # Save trained model
    model.save_model(save_dir)
    
    return model

class GameRecorder:
    """Records gameplay demonstrations for RL training"""
    
    def __init__(self, agent, action_space, save_dir="recordings", 
                 capt_interval=1.0, sequence_length=4):
        
        self.agent = agent
        self.dt_num = int(self.agent.eng_name[-1])
        self.window_pos = agent.pos
        self.window_size = agent.size
        self.save_dir = save_dir
        self.capt_interval = capt_interval
        self.sequence_length = sequence_length
        self.action_space = action_space
        
        self.recording = False
        self.last_capture_time = 0
        self.current_session = None
        self.demonstration_buffer = []
        
        # Frame sequence for state construction
        self.frame_sequence = []
        
        # Track pressed keys
        self.pressed_keys = set()
        self.key_press_times = {}
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Setup input listeners
        self.mouse_listener = mouse.Listener(
            on_click=self._on_click,
            suppress=False)
        self.keyboard_listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
            suppress=False)
        
        # Mouse tracking for drags
        self.mouse_press_pos = None
        self.mouse_press_time = None
        self.mouse_click_threshold = action_space['mouse_click_threshold']
        
        # Track currently pressed keys for combinations
        self.current_key_combination = set()
            
    def start_recording(self):
        """Start a new recording session"""
        if self.recording:
            return

        print("Start recording.")
        self.recording = True
        self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(os.path.join(self.save_dir, self.current_session), exist_ok=True)
        
        self.last_capture_time = time.time()
        self.demonstration_buffer = []
        self.frame_sequence = []
        
        # Start input listeners
        self.mouse_listener.start()
        self.keyboard_listener.start()
        
        # Start capture thread
        self.capture_thread = Thread(target=self._capture_loop)
        self.capture_thread.start()
        
    def stop_recording(self):
        """Stop recording and save the session"""
        if not self.recording:
            return

        print("Stop recording.")  
        self.recording = False
        self.capture_thread.join()
        
        # Save demonstration buffer
        if self.demonstration_buffer:
            buffer_path = os.path.join(self.save_dir, self.current_session, 
                                     "demonstration.json")
            with open(buffer_path, 'w') as f:
                json.dump(self.demonstration_buffer, f)
                
        self.mouse_listener.stop()
        self.keyboard_listener.stop()
        
    def _capture_loop(self):
        """Main capture loop that runs in separate thread"""
        while self.recording:
            current_time = time.time()
            
            if current_time - self.last_capture_time >= self.capt_interval:
                self._capture_frame(action_type="null", action_data={})
                self.last_capture_time = current_time
                
            time.sleep(0.01)
            
    def _capture_frame(self, action_type, action_data):
        """Capture current frame with action annotation"""
        frame_path = os.path.join(
            self.save_dir, 
            self.current_session,
            f"frame_{len(self.demonstration_buffer):06d}.png"
        )

        frame = utils.screenshot(region=(
            self.window_pos[0], self.window_pos[1],
            self.window_size[0], self.window_size[1]
        ), path=frame_path)
        
        # Add to frame sequence
        self.frame_sequence.append(frame_path)
        if len(self.frame_sequence) > self.sequence_length:
            self.frame_sequence.pop(0)
                
        # Save frame info and action
        frame_data = {
            "frame": frame_path,
            "timestamp": time.time(),
            "action_type": action_type,
            "sequence": self.frame_sequence.copy(),
            **action_data
        }
        
        self.demonstration_buffer.append(frame_data)
        self.last_capture_time = time.time()
            
    def _on_click(self, x, y, button, pressed):
        """Handle mouse clicks and drags"""
        if not self.recording:
            return
            
        # Convert global coordinates to window coordinates
        window_x, window_y = self.agent.adjust_xy_inv(x, y, dt_num=1)["x"], self.agent.adjust_xy_inv(x, y, dt_num=1)["y"]
        
        # Convert to normalized coordinates for RL
        norm_x = window_x / self.window_size[0] * 2 - 1
        norm_y = window_y / self.window_size[1] * 2 - 1
        
        # Only process if within window
        if not (0 <= window_x < self.window_size[0] and 0 <= window_y < self.window_size[1]):
            return
            
        if pressed:
            # Record start of potential drag
            self.mouse_press_pos = (norm_x, norm_y)
            self.mouse_press_time = time.time()
        else:
            # Mouse released - determine if it was a click or drag
            if self.mouse_press_pos is None:
                return
                
            press_duration = time.time() - self.mouse_press_time
            if press_duration < self.mouse_click_threshold:
                # Short press - it's a click
                self._capture_frame(
                    action_type="mouse_click",
                    action_data={"position": [norm_x, norm_y]}
                )
            else:
                # Long press - it's a drag
                self._capture_frame(
                    action_type="mouse_drag",
                    action_data={
                        "start_position": list(self.mouse_press_pos),
                        "end_position": [norm_x, norm_y]
                    }
                )
                
            self.mouse_press_pos = None
            self.mouse_press_time = None
            
    def _on_press(self, key):
        """Handle key press"""
        if not self.recording:
            return
            
        # Check for Page Down key
        if key == keyboard.Key.page_down:
            print("Page Down pressed - stopping recording...")
            self.stop_recording()
            return
            
        try:
            key_char = key.char
        except AttributeError:
            key_char = str(key)
            
        if key_char not in self.pressed_keys:
            self.pressed_keys.add(key_char)
            self.key_press_times[key_char] = time.time()
            
            # Update current key combination
            if key_char in self.action_space['keys']:
                self.current_key_combination.add(key_char)
            
            # Record the key combination
            if self.current_key_combination:
                self._capture_frame(
                    action_type="key_combination",
                    action_data={"keys": list(self.current_key_combination)}
                )
            
    def _on_release(self, key):
        """Handle key release"""
        if not self.recording:
            return
            
        try:
            key_char = key.char
        except AttributeError:
            key_char = str(key)
            
        if key_char in self.pressed_keys:
            self.pressed_keys.remove(key_char)
            if key_char in self.current_key_combination:
                self.current_key_combination.remove(key_char)
            
            # Update frame with new key combination
            if self.current_key_combination:
                self._capture_frame(
                    action_type="key_combination",
                    action_data={"keys": list(self.current_key_combination)}
                )

def record_gameplay(agent, action_space, save_dir="recordings", capt_interval=1.0):
    """
    Record gameplay demonstrations for RL training
    
    Args:
        agent: Game agent instance 
        action_space: Dictionary defining possible actions
        save_dir: Directory to save recordings
        capt_interval: Interval between frame captures in seconds
        
    Returns:
        GameRecorder instance
    """
    print("Starting gameplay recording...")
    print("Controls:")
    print("- Mouse clicks and drags will be recorded")
    print("- Key presses from action space will be recorded:")
    print(f"  {', '.join(action_space['keys'])}")
    print("- Press Page Down to stop recording")
    
    # Create recorder with same sequence length as RL model
    recorder = GameRecorder(
        agent=agent,
        action_space=action_space,
        save_dir=save_dir,
        capt_interval=capt_interval,
        sequence_length=4  # Match RL state processor sequence length
    )
    
    # Start recording
    recorder.start_recording()
    
    return recorder