import reinforcement

rl_recording_agent = "dt_1"

import boto3
from sagemaker.remote_function import remote
from datetime import datetime

imput_img_shape = (384, 216) # (1536, 864)
def upload_recordings_to_s3(local_dir, bucket_name, s3_prefix, aws_region="us-east-1"):
    """Upload local gameplay recordings to S3 bucket"""
    s3_client = boto3.client('s3', region_name=aws_region)
    
    # Create bucket if it doesn't exist
    try:
        s3_client.head_bucket(Bucket=bucket_name)
    except:
        s3_client.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={'LocationConstraint': aws_region}
        )
    
    # Upload files recursively
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            rel_path = os.path.relpath(local_path, local_dir)
            s3_path = os.path.join(s3_prefix, rel_path).replace("\\", "/")
            
            print(f"Uploading {local_path} to s3://{bucket_name}/{s3_path}")
            s3_client.upload_file(local_path, bucket_name, s3_path)
    
    return f"s3://{bucket_name}/{s3_prefix}"

@remote(
    instance_type="ml.c5.xlarge",
    role="arn:aws:iam::746669199361:role/SageMakerExecutionRole",
    use_spot_instances=True,
    max_runtime_in_seconds=7200,
    max_wait_time_in_seconds=9000,
    environment_variables={
        "TF_FORCE_GPU_ALLOW_GROWTH": "true",
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python"
    },
    dependencies="./sagemaker/requirements.txt",
    include_local_workdir=True,
    image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-tensorflow-trcomp-training:2.6.3-transformers4.17.0-gpu-py38-cu112-ubuntu20.04",
)
def train_rl_model_remote(
    demo_dir_s3,  # S3 path to demo directory
    action_space, 
    game_type,
    num_demo_epochs=50,
    save_dir_s3=None,  # S3 path to save the model
):
    """Remote training function to train RL model on AWS"""
    import os
    import json
    import boto3
    import numpy as np
    import tensorflow as tf
    import cv2
    from reinforcement import GameStateProcessor, HybridGameAutomationRL, RewardFunction
    from pathlib import Path
    
    # Set better error reporting
    tf.debugging.set_log_device_placement(True)
    
    print(f"Starting training for game type {game_type}")
    print(f"Demo directory S3 path: {demo_dir_s3}")
    
    # Parse S3 path components
    s3_parts = demo_dir_s3.split('/', 3)
    bucket_name = s3_parts[2]
    s3_prefix = s3_parts[3] if len(s3_parts) > 3 else ""
    
    print(f"Bucket name: {bucket_name}")
    print(f"S3 prefix: {s3_prefix}")
    
    # Create local directories
    local_demo_dir = f"/tmp/{game_type}_recordings"
    os.makedirs(local_demo_dir, exist_ok=True)
    
    # Initialize S3 client
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    
    # List objects in bucket to debug
    print("Listing objects in bucket with prefix:")
    all_objects = list(bucket.objects.filter(Prefix=s3_prefix))
    print(f"Found {len(all_objects)} objects with prefix {s3_prefix}")
    
    # Print all objects to debug
    print("All objects found:")
    for obj in all_objects:
        print(f"  {obj.key}")
    
    # Only download if we found objects
    if all_objects:
        print("Downloading all objects...")
        for obj in all_objects:
            # Skip objects that are directories
            if obj.key.endswith('/'):
                continue
                
            # Get the relative path from the prefix
            if s3_prefix:
                rel_path = obj.key[len(s3_prefix):].lstrip('/')
            else:
                rel_path = obj.key
                
            # Handle demonstration.json specially
            if os.path.basename(obj.key) == 'demonstration.json':
                # Create the session directory
                session_dir = os.path.dirname(obj.key)
                local_session_dir = os.path.join(local_demo_dir, os.path.basename(session_dir))
                os.makedirs(local_session_dir, exist_ok=True)
                
                # Save to the session directory
                local_path = os.path.join(local_session_dir, 'demonstration.json')
            else:
                # For frames, save directly to the demo dir
                local_path = os.path.join(local_demo_dir, os.path.basename(obj.key))
            
            # Create parent directories
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            print(f"Downloading {obj.key} to {local_path}")
            bucket.download_file(obj.key, local_path)
    else:
        print(f"WARNING: No objects found in bucket {bucket_name} with prefix {s3_prefix}")
    
    # Check what was downloaded
    print("Contents of local demo directory:")
    for root, dirs, files in os.walk(local_demo_dir):
        for file in files:
            print(os.path.join(root, file))
    
    # Define a function to fix frame paths in demonstration.json
    def fix_frame_paths_in_demo(demo_path):
        """Fix frame paths to match local structure"""
        try:
            with open(demo_path, 'r') as f:
                demo_data = json.load(f)
            
            # Directory containing the demo file
            demo_dir = os.path.dirname(demo_path)
            
            # Fix all frame paths
            for i, frame_data in enumerate(demo_data):
                if 'frame' in frame_data:
                    # Get the basename of the frame file
                    frame_basename = os.path.basename(frame_data['frame'].replace('\\', '/'))
                    
                    # Create new path relative to demo dir
                    new_path = os.path.join(demo_dir, frame_basename)
                    
                    # Update the path
                    demo_data[i]['frame'] = new_path
            
            # Save the updated demo file
            with open(demo_path, 'w') as f:
                json.dump(demo_data, f)
                
            print(f"Fixed frame paths in {demo_path}")
            return True
        except Exception as e:
            print(f"Error fixing frame paths in {demo_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    # Fix all demonstration.json files
    for root, dirs, files in os.walk(local_demo_dir):
        for file in files:
            if file == 'demonstration.json':
                fix_frame_paths_in_demo(os.path.join(root, file))
                    
    # Define a custom load_demonstrations function for our structure
    def load_demonstrations(demo_dir):
        """Load demonstration data from downloaded structure"""
        demonstrations = []
        
        print(f"Loading demonstrations from: {demo_dir}")
        if not os.path.exists(demo_dir):
            print(f"WARNING: Demo directory does not exist: {demo_dir}")
            return []
        
        # Look for all demonstration.json files
        for root, dirs, files in os.walk(demo_dir):
            if 'demonstration.json' in files:
                demo_path = os.path.join(root, 'demonstration.json')
                print(f"Found demonstration file: {demo_path}")
                
                try:
                    with open(demo_path, 'r') as f:
                        demo_data = json.load(f)
                    
                    print(f"Loaded {len(demo_data)} frames from {demo_path}")
                    
                    # Process each frame in the demonstration
                    processed_frames = []
                    
                    for frame_data in demo_data:
                        # Get local frame path
                        local_frame_path = frame_data['frame']
                        
                        # Check if frame exists
                        if not os.path.exists(local_frame_path):
                            print(f"Warning: Frame file not found: {local_frame_path}")
                            # Try the basename approach as fallback
                            frame_basename = os.path.basename(local_frame_path)
                            fallback_path = os.path.join(demo_dir, frame_basename)
                            
                            if os.path.exists(fallback_path):
                                print(f"Found frame at fallback path: {fallback_path}")
                                local_frame_path = fallback_path
                            else:
                                print(f"Frame not found at fallback path either: {fallback_path}")
                                continue
                        
                        # Load frame
                        frame = cv2.imread(local_frame_path)
                        if frame is None:
                            print(f"Warning: Failed to load frame: {local_frame_path}")
                            continue
                        
                        # Process frame to create state imput_img_shape
                        processed_frame = cv2.resize(frame, imput_img_shape)
                        if len(processed_frame.shape) == 3:  # Color image
                            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
                        
                        # Create state and next_state
                        processed_frame = processed_frame / 255.0  # Normalize
                        processed_frame = processed_frame[..., np.newaxis]  # Add channel dimension
                        
                        state = np.repeat(processed_frame[np.newaxis, ...], 4, axis=0)
                        next_state = state.copy()  # Simple duplicate for demo
                        
                        # Convert action data to model format
                        action = {
                            'type': np.array([0]),  # Default to click
                            'mouse_pos': None,
                            'key_presses': None
                        }
                        
                        if frame_data['action_type'] == 'mouse_click':
                            action['type'] = np.array([0])
                            action['mouse_pos'] = np.array(frame_data['position'], dtype=np.float32)
                        elif frame_data['action_type'] == 'mouse_drag':
                            action['type'] = np.array([1])
                            start_pos = np.array(frame_data['start_position'], dtype=np.float32)
                            end_pos = np.array(frame_data['end_position'], dtype=np.float32)
                            action['mouse_pos'] = np.concatenate([start_pos, end_pos])
                        elif frame_data['action_type'] == 'key_combination':
                            action['type'] = np.array([2])
                            key_presses = np.zeros(len(action_space['keys']), dtype=np.float32)
                            for key in frame_data.get('keys', []):
                                if key in action_space['keys']:
                                    idx = action_space['keys'].index(key)
                                    key_presses[idx] = 1.0
                            action['key_presses'] = key_presses
                        else:  # null action
                            action['type'] = np.array([3])
                        
                        processed_frames.append({
                            'state': state,
                            'action': action,
                            'next_state': next_state,
                            'reward': 1.0  # Demonstrations are assumed to be good
                        })
                    
                    demonstrations.extend(processed_frames)
                    print(f"Processed {len(processed_frames)} frames from {demo_path}")
                    
                except Exception as e:
                    import traceback
                    print(f"Error processing demonstration at {demo_path}: {str(e)}")
                    traceback.print_exc()
        
        print(f"Total processed demonstrations: {len(demonstrations)}")
        return demonstrations
    
    # Set local save directory
    local_save_dir = f"/tmp/{game_type}_models"
    os.makedirs(local_save_dir, exist_ok=True)
    
    # Initialize state processor with dummy agent
    state_processor = GameStateProcessor(
        agent=None,
        target_size=imput_img_shape,
        sequence_length=4
    )
    
    # Initialize model
    model = HybridGameAutomationRL(
        state_processor=state_processor,
        action_space=action_space,
        reward_function=RewardFunction(),
        agent=None,
        worker_lock=None
    )
    
    # Override the model's _load_demonstrations function
    model._load_demonstrations = load_demonstrations
    
    # Load demonstrations manually first to check if there are any
    demos = load_demonstrations(local_demo_dir)
    print(f"local_demo_dir: {local_demo_dir}")
    if not demos:
        raise ValueError("No demonstrations found.")
        
    # Train on demonstrations
    print("Training on demonstrations...")
    model.train_on_demonstrations(local_demo_dir, num_epochs=num_demo_epochs)
    
    # Save model
    model_path = os.path.join(local_save_dir, f"{game_type}_model")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Set up S3 output path
    if not save_dir_s3:
        save_dir_s3 = f"s3://{bucket_name}/{game_type}/models"
    
    # Parse S3 output path
    s3_out_parts = save_dir_s3.split('/', 3)
    out_bucket = s3_out_parts[2]
    out_prefix = s3_out_parts[3] if len(s3_out_parts) > 3 else ""
    
    # Upload saved model to S3
    print(f"Uploading model to {save_dir_s3}")
    s3_client = boto3.client('s3')
    
    # Upload all model files
    model_dir_files = os.listdir(local_save_dir)
    for file in model_dir_files:
        if file.startswith(f"{game_type}_model"):
            local_file = os.path.join(local_save_dir, file)
            s3_key = f"{out_prefix}/{file}" if out_prefix else file
            print(f"Uploading {local_file} to s3://{out_bucket}/{s3_key}")
            s3_client.upload_file(local_file, out_bucket, s3_key)
    
    return {
        "status": "completed",
        "game_type": game_type,
        "epochs": num_demo_epochs,
        "model_path": f"{save_dir_s3}/{game_type}_model"
    }

def train_model_on_aws(
    game_type,
    action_space,
    instance_type="ml.c5.xlarge",
    use_spot=True,
    aws_region="us-east-1",
    bucket_name="game-rl-training",
    hyperparameters=None,
    aws_access_key_id=None,
    aws_secret_access_key=None
):
    """Train RL model on AWS using local recordings"""
    # Set AWS credentials if provided
    if aws_access_key_id and aws_secret_access_key:
        boto3.setup_default_session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region
        )
    
    # Create a SageMaker execution role if one doesn't exist
    iam_client = boto3.client('iam')
    role_name = "SageMakerExecutionRole"
    
    try:
        # Check if role exists
        iam_client.get_role(RoleName=role_name)
        print(f"Using existing IAM role: {role_name}")
    except iam_client.exceptions.NoSuchEntityException:
        # Create a new role
        print(f"Creating new IAM role: {role_name}")
        assume_role_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "sagemaker.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        try:
            create_role_response = iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(assume_role_policy_document),
                Description="SageMaker execution role created for RL training"
            )
            
            # Attach required policies
            iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
            )
            iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn="arn:aws:iam::aws:policy/AmazonS3FullAccess"
            )
            
            # Wait for role to be ready
            import time
            time.sleep(10)
        except Exception as e:
            print(f"Failed to create role: {e}")
            # If we can't create a role, we'll need to use an existing one or provide instructions
            print("Please create a SageMaker execution role in the AWS console and try again.")
            return None, None
    
    # Get role ARN
    role_response = iam_client.get_role(RoleName=role_name)
    role_arn = role_response['Role']['Arn']
    
    # Configure remote function with the role
    # train_rl_model_remote.remote_params['role'] = role_arn
    
    # Default hyperparameters if not provided
    if hyperparameters is None:
        hyperparameters = {
            'game-type': game_type,
            'num-epochs': '50',
            'batch-size': '32',
            'learning-rate': '0.0001'
        }
    
    # Local recordings directory
    local_recordings_dir = f"./syncing/RL/recordings/{game_type}"
    
    # Upload recordings to S3
    s3_recordings_prefix = f"{game_type}/recordings/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    s3_recordings_path = upload_recordings_to_s3(
        local_recordings_dir, 
        bucket_name, 
        s3_recordings_prefix,
        aws_region
    )
    
    # S3 path for saving models
    s3_models_prefix = f"{game_type}/models"
    s3_models_path = f"s3://{bucket_name}/{s3_models_prefix}"
    
    # Start remote training job
    print(f"Starting remote training job for {game_type}...")
    training_job = train_rl_model_remote(
        demo_dir_s3=s3_recordings_path,
        action_space=action_space,
        game_type=game_type,
        num_demo_epochs=int(hyperparameters['num-epochs']),
        save_dir_s3=s3_models_path
    )
    
    # Return job information and trainer object
    job_info = {
        'training_job': training_job,
        'game_type': game_type,
        'instance_type': instance_type,
        'spot_instance': use_spot,
        'bucket': bucket_name,
        'recordings_path': s3_recordings_path,
        'models_path': s3_models_path,
        'start_time': datetime.now().isoformat(),
        'hyperparameters': hyperparameters,
        'role_arn': role_arn
    }
    
    return job_info, training_job

def download_trained_model(model_path, game_type, aws_region="us-east-1", bucket_name="game-rl-training"):
    """
    Download trained model from S3 using model_path
    
    Args:
        model_path: S3 path to the model (e.g., 's3://game-rl-training/nc/models/nc_model')
        game_type: Game type ('nc', 'ms', etc.)
        aws_region: AWS region
        bucket_name: S3 bucket name (used as fallback if not in model_path)
        
    Returns:
        Path to downloaded model
    """
    import os
    import boto3
    
    # Local models directory
    local_models_dir = f"./syncing/RL/models/{game_type}"
    os.makedirs(local_models_dir, exist_ok=True)
    
    # Local model path
    local_model_path = os.path.join(local_models_dir, f"{game_type}_model")
    
    # Parse S3 path
    if model_path.startswith('s3://'):
        parts = model_path.replace('s3://', '').split('/')
        s3_bucket = parts[0]
        s3_key_prefix = '/'.join(parts[1:])
    else:
        # Fallback if model_path is not a full S3 URI
        s3_bucket = bucket_name
        s3_key_prefix = f"{game_type}/models/{game_type}_model"
    
    # Initialize S3 client
    s3 = boto3.client('s3', region_name=aws_region)
    
    # Get model components (network weights, config, etc.)
    try:
        # List objects with the model prefix
        response = s3.list_objects_v2(
            Bucket=s3_bucket,
            Prefix=s3_key_prefix
        )
        
        if 'Contents' not in response:
            print(f"Warning: No model files found at {model_path}")
            return None
            
        # Download each model file
        for obj in response['Contents']:
            key = obj['Key']
            filename = os.path.basename(key)
            local_file = os.path.join(local_models_dir, filename)
            
            print(f"Downloading {key} to {local_file}")
            s3.download_file(s3_bucket, key, local_file)
        
        print(f"Model downloaded to {local_models_dir}")
        return local_model_path.replace('\\', '/')
        
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if rl_recording_agent is None: ## Agent Game Play
    runner.run(list_agents)
else: ## RL Game Play
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
                'space', 'tab',
                # Letter keys
                'w', 'a', 's', 'd', 'f', 'z', 'g',
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
    if False:
        # To record game screeoshot and actions locally:
        reinforcement.record_gameplay(agent = rl_target_agent, action_space= action_space, save_dir= os.path.join('.', 'syncing', 'RL', 'recordings', gtype))
    elif False:
        # Train on local mahcine:
        reinforcement.train_rl_model(agent = rl_target_agent, demo_dir = os.path.join('.', 'syncing', 'RL', 'recordings', gtype), action_space = action_space, worker = worker_obj, reward_function= reinforcement.RewardFunction(), worker_lock=worker_lock, gtype= gtype, target_size= imput_img_shape)
    elif False:
        # Load trained model and run gameplay:
        # Define simple reward function
        class CustomRewardFunction(reinforcement.RewardFunction):
            def __init__(self, agent):
                super().__init__()
                self.agent = agent
                self.last_hp = None
                
            def calculate_reward(self, current_frame, previous_frames, current_action):
                reward = 0.5  # Base reward
                
                # Reward based on HP changes
                current_hp = self.agent.get_hp()
                if self.last_hp is not None:
                    hp_diff = current_hp - self.last_hp
                    reward += hp_diff * 5.0  # Reward for HP gain, penalty for loss
                self.last_hp = current_hp
                
                # Reward for hunt assist being on
                if self.agent.is_hunt_assist_on():
                    reward += 0.2
                    
                return reward
                
        # Define end condition
        def end_condition(state):
            return (rl_target_agent.is_dead() or 
                    rl_target_agent.is_disconnected() or 
                    rl_target_agent.get_hp() < 0.2)
        
        # Define agent action condition
        def agent_action_condition(state):
            if rl_target_agent.is_dead():
                return rl_target_agent.revive_and_comeback_to_hunt
            if rl_target_agent.is_disconnected():
                return rl_target_agent.wait_until_reconnected
            if rl_target_agent.get_hp() < 0.25:
                return lambda: rl_target_agent.return_buy_comeback(return_from_pk=True)
            return False
        
        # Create controller with existing model
        controller = reinforcement.RLGameplayController(
            agent=rl_target_agent,
            action_space=action_space,
            reward_function=CustomRewardFunction(rl_target_agent),
            end_condition=end_condition,
            agent_action_condition=agent_action_condition,
            action_threshold=0.7
        )
        
        if False:
            # Run gameplay
            try:
                print(f"Starting RL gameplay for {rl_target_agent.eng_name}")
                episode_reward = controller.run_gameplay()
                print(f"RL gameplay completed with reward: {episode_reward}")
            except Exception as e:
                print(f"Error during RL gameplay: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # Try to return to town if something went wrong
                try:
                    rl_target_agent.return_ber()
                except:
                    pass
        tf.keras.backend.clear_session()
        del controller
        gc.collect()
        print(f"RL gameplay completed with reward: ")
    elif secures.if_remote_work:
        # To train on AWS or cloud platform:

        job_info, trainer = train_model_on_aws(
            game_type=gtype,
            aws_access_key_id=secures.aws_access_key_id,
            aws_secret_access_key=secures.aws_secret_access_key,
            action_space=action_space,
        )

        # To check job status later:
        # job_status = check_training_job(trainer=trainer)

        # If job completed, download model:
        if job_info['training_job']['status'] == 'completed':
            # If job completed, download model:
            if 'model_path' in trainer.keys():
                model_path = download_trained_model(
                    model_path=trainer['model_path'],
                    game_type=gtype
                )
                print(f"Training was successful, Model downloaded to: {model_path}")
            else:
                print("Model path not found in job info. Training may not have completed successfully.")
        else:
            print(f"Training job failed with job_info: {job_info}")
    else:
        print("No work to do, check setting.")