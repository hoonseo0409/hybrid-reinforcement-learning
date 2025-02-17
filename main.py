import reinforcement

rl_recording_agent = "dt_1"
rl_target_agent = next(agent for agent in list_agents if agent.eng_name == rl_recording_agent)
gtype = rl_target_agent.gtype
if gtype == "ms":
    action_space = {
        'keys': [
            # Arrow keys
            'Key.up', 'Key.down', 'Key.left', 'Key.right',
            # Modifier keys
            'Key.ctrl', 'Key.alt', 'esc', 'space',
            # Letter keys
            'z', 'x', 'c', 'u', 'q', 'w', 'e', 'r',
            'a', 's', 'd', 'f',
            # Number keys
            '1', '2', '3', '4', '5', '6'
        ],
        'mouse': {
            'min_pos': [0, 0],
            'max_pos': rl_target_agent.size,  # Adjust based on window size
            'actions': ['click', 'drag']
        },
        'mouse_click_threshold': 0.3  # Seconds threshold between click and drag
    }
elif gtype == "nc":
    action_space = {
        'keys': [
            # Modifier keys
            'space',
            # Letter keys
            'w', 'a', 's', 'd',
            # Number keys
            '1', '2', '3', '4', '5', '6', '7', '8'
        ],
        'mouse': {
            'min_pos': [0, 0],
            'max_pos': rl_target_agent.size,  # Adjust based on window size
            'actions': ['click', 'drag']
        },
        'mouse_click_threshold': 0.3  # Seconds threshold between click and drag
    }
else:
    raise Exception(NotImplementedError)
if 0:
    reinforcement.record_gameplay(agent = rl_target_agent, action_space= action_space, save_dir= f"./syncing/RL/recordings/{gtype}")
else:
    reinforcement.train_rl_model(agent = rl_target_agent, demo_dir = f"./syncing/RL/recordings/{gtype}", action_space = action_space, reward_function= reinforcement.RewardFunction())