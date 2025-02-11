import reinforcement

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

if True:
    reinforcement.record_gameplay(agent = rl_target_agent, action_space= action_space)
else:
    reinforcement.train_rl_model(agent = rl_target_agent, demo_dir = "./recordings", action_space = action_space, worker = worker_obj, reward_function= reinforcement.RewardFunction(), worker_lock=worker_lock)