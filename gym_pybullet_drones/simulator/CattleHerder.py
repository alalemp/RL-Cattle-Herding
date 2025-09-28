"""
-------
Setup:

    pip install tensorboard pyyaml

Run:
    $ conda activate drones
    $ python CattleHerder.py

Monitor logs:

    $ tensorboard --logdir models/model-v9-0/tb

    tensorboard --logdir /home/ben/ros_ws/src/RL-Cattle-Herding/gym_pybullet_drones/simulator/models/model-v3-1
-------
"""

import os
import time
import argparse
import numpy as np
import torch
import gymnasium as gym
import multiprocessing
from datetime import datetime

# Force PyTorch to use CPU to avoid GPU performance warnings
num_cores = max(1, multiprocessing.cpu_count() - 2)
torch.set_default_device('cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
#os.environ['OMP_NUM_THREADS'] = str(num_cores)
#torch.set_num_threads(num_cores)

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.CattleAviary import CattleAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType, DroneModel

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'models'
DEFAULT_COLAB = False
TARGET_REWARD = 99999   # reward threshold to stop training
LOAD_FILE = "model-alen-v03-0/best_model.zip"  # Phase 2: Load Phase 1 best model and add containment rewards
# LOAD_FILE = None  # Commented out - was used for fresh Phase 1 start

DEFAULT_OBS = ObservationType('cokin') # collaborative kinematics
DEFAULT_ACT = ActionType('vel')        # 'rpm' | 'pid' | 'vel' | 'one_d_rpm' | 'one_d_pid'
DEFAULT_NUM_ENVS = 32 # 32
DEFAULT_DRONES = 6
DEFAULT_CATTLE = 16

MAX_TIMESTEPS = 1000000
EVAL_FILE = None
EVALUATION_FREQUENCY = 2048

EVALUATE_ONLY = False   # skip training, run evaluation only
EVAL_EPISODE_LENGTH = 60  # Episode length in seconds for evaluation (default: 60s)
NUM_EVALUATION_EPS = 50 # 50  # Increase episodes for better statistics

def evaluate_policy_with_progress(model, env, n_eval_episodes=10, deterministic=True):
    """
    Custom evaluation function with progress indicators
    """
    all_episode_rewards = []
    all_episode_lengths = []
    
    print(f"[INFO] Starting evaluation with {n_eval_episodes} episodes...")
    print("Progress: ", end="", flush=True)
    
    for episode in range(n_eval_episodes):
        episode_reward = 0.0
        episode_length = 0
        obs, _ = env.reset()
        
        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            # Debug info for first episode
            if episode == 0 and episode_length % 100 == 0:
                try:
                    # Get drone and herd positions for debugging
                    drone_states = [env.unwrapped._getDroneStateVector(i) for i in range(env.unwrapped.NUM_DRONES)]
                    herd_center = env.unwrapped.HerdCentroid()
                    drone_center = np.mean([d[:2] for d in drone_states], axis=0)
                    distance_to_herd = np.linalg.norm(drone_center - herd_center[:2])
                    print(f"\n  Step {episode_length}: Distance to herd: {distance_to_herd:.2f}m, Reward: {reward:.2f}")
                    print(f"  Action magnitude: {np.linalg.norm(action):.3f}")
                except:
                    pass
            
            if terminated or truncated:
                break
        
        all_episode_rewards.append(episode_reward)
        all_episode_lengths.append(episode_length)
        
        # Progress indicator
        if (episode + 1) % max(1, n_eval_episodes // 20) == 0 or episode == n_eval_episodes - 1:
            print(f"█", end="", flush=True)
        
        # Show intermediate stats every 10 episodes
        if (episode + 1) % 10 == 0:
            current_mean = np.mean(all_episode_rewards)
            current_std = np.std(all_episode_rewards)
            mean_length = np.mean(all_episode_lengths)
            print(f"\n  Episode {episode + 1:2d}: Mean Reward = {current_mean:7.1f} ± {current_std:5.1f}, Mean Length = {mean_length:4.0f}")
            print("Progress: ", end="", flush=True)
    
    mean_reward = np.mean(all_episode_rewards)
    std_reward = np.std(all_episode_rewards)
    mean_length = np.mean(all_episode_lengths)
    
    print(f"\n\n[EVALUATION COMPLETE]")
    print(f"Episodes: {n_eval_episodes}")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean Episode Length: {mean_length:.1f}")
    print(f"Min/Max Reward: {np.min(all_episode_rewards):.1f} / {np.max(all_episode_rewards):.1f}")
    
    return mean_reward, std_reward

#Main Runner
def run(
        target_reward=TARGET_REWARD,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        gui=DEFAULT_GUI, 
        plot=False, 
        colab=DEFAULT_COLAB, 
        record_video=DEFAULT_RECORD_VIDEO, 
        local=True,
        drone=DroneModel.CF2X,
        num_drones=DEFAULT_DRONES,
        num_cattle=DEFAULT_CATTLE,
        obs=DEFAULT_OBS,
        act=DEFAULT_ACT,
        eval_only=EVALUATE_ONLY,
        eval_episode_length=EVAL_EPISODE_LENGTH):

    # Debug: Print the eval_only parameter to see what we received
    print(f"[DEBUG] eval_only parameter: {eval_only} (type: {type(eval_only)})")
    
    # Set model directory - use different directories for different reward structures
    if eval_only:
        model_dir = os.path.join(output_folder, 'model-alen-v02-2-0')  # Your trained model
    else:
        model_dir = os.path.join(output_folder, 'model-alen-v04-0')  # Phase 2: Containment rewards with winding number algorithm
    os.makedirs(model_dir, exist_ok=True)


    # Environment setup
    env_kwargs = dict(
        drone_model=drone,
        num_drones=num_drones,
        num_cattle=num_cattle,
        neighbourhood_radius=np.inf,
        obs=obs,
        act=act,
        gui=gui,
        record=record_video,
        pyb_freq=240,
        ctrl_freq=240
    )
    
    # Enable training mode for herd-relative positioning
    class CurriculumCattleAviary(CattleAviary):
        def __init__(self, eval_episode_length=None, **kwargs):
            super().__init__(**kwargs)
            self.training_mode = True  # Enable randomized herd positions
            self.eval_episode_length = eval_episode_length  # Custom evaluation episode length
            self.episode_length_schedule = {
                0: 500,      # Start with shorter episodes
                50000: 1000,  # Increase episode length
                150000: 1500, # Full length episodes
            }
        
        def reset(self, seed=None, options=None):
            # Use custom evaluation episode length if set and not in training mode
            if hasattr(self, 'training_mode') and not self.training_mode and self.eval_episode_length:
                self.EPISODE_LEN_SEC = self.eval_episode_length
            elif hasattr(self, '_total_steps'):
                # Adjust episode length based on training progress
                for step_threshold in sorted(self.episode_length_schedule.keys(), reverse=True):
                    if self._total_steps >= step_threshold:
                        self.EPISODE_LEN_SEC = self.episode_length_schedule[step_threshold] / self.CTRL_FREQ
                        break
            return super().reset(seed, options)
    
    # Only create training environments if not in evaluation-only mode
    if not eval_only:
        # Update environment creation to use curriculum
        def make_env():
            return CurriculumCattleAviary(**env_kwargs)

        train_env = make_vec_env(
            make_env, 
            n_envs=DEFAULT_NUM_ENVS, 
            vec_env_cls=SubprocVecEnv, 
            seed=0
        )

        # Create evaluation environment that matches training
        def make_eval_env():
            env = CurriculumCattleAviary(**env_kwargs)
            env.training_mode = False  # Disable randomization for consistent eval
            return env
        
        eval_env = DummyVecEnv([lambda: Monitor(make_eval_env())])
    else:
        # For evaluation only, we'll create environments later to avoid conflicts
        train_env = None
        eval_env = None

    # Load existing model or create new one
    if LOAD_FILE is not None and os.path.exists(os.path.join(output_folder, LOAD_FILE)):
        model_path = os.path.join(output_folder, LOAD_FILE)
        # Load model with appropriate environment
        if eval_only:
            model = PPO.load(model_path)
        else:
            model = PPO.load(model_path, env=train_env)
        print(f"[LOG] Loaded existing PPO model from {model_path}")
        
        # If not eval_only, we will continue training from the loaded model
        if not eval_only:
            print(f"[LOG] Will continue training from loaded model")
            
    elif not eval_only:
        model = PPO(
            "MlpPolicy", 
            train_env,
            learning_rate=lambda progress: 3e-5 * (1 - 0.8 * progress),  # Decaying learning rate - WORKING WELL!
            n_steps=4096,          # More steps per update for better sampling
            batch_size=128,        # Larger batch for more stable updates
            n_epochs=15,           # More epochs for better policy improvement
            gamma=0.995,           # Higher discount for long-term formation maintenance
            gae_lambda=0.98,       # Higher GAE for better advantage estimation
            clip_range=0.2,        # Standard clip range
            ent_coef=0.05,         # Lower entropy for more focused exploration
            vf_coef=0.5,           # Standard value function coefficient
            max_grad_norm=0.5,
            tensorboard_log=os.path.join(model_dir, "tb"),
            policy_kwargs=dict(
                log_std_init=-0.5,     # Higher initial action variance for exploration
                ortho_init=True,       # Orthogonal initialization for better training
                net_arch=[dict(pi=[256, 256, 128], vf=[256, 256, 128])]  # Larger network for complex coordination
            ),
            verbose=1,
            stats_window_size=100  # Track rollout stats over 100 episodes
        )
        print("[LOG] Created new PPO model")
    else:
        # In evaluation-only mode, we must have a model file
        raise ValueError("Evaluation-only mode requires LOAD_FILE to be set and exist")

    # Configure logger (only needed for training)
    if not eval_only:
        tb_log_dir = os.path.join(model_dir, "tb")
        print(f"[LOG] TensorBoard logs will be saved to: {tb_log_dir}")
        os.makedirs(tb_log_dir, exist_ok=True)
        
        new_logger = configure(tb_log_dir, ["stdout", "tensorboard"])
        model.set_logger(new_logger)
        
        # Force an initial log entry
        model.logger.record("setup/initialized", 1.0)
        model.logger.dump()

    # Custom callback for formation learning metrics
    class FormationLoggingCallback(BaseCallback):
        def __init__(self, verbose=0):
            super(FormationLoggingCallback, self).__init__(verbose)
            self.step_count = 0
            self.episode_rewards = []
            self.episode_lengths = []
            
        def _on_step(self) -> bool:
            self.step_count += 1
            
            # Log episode-level metrics when episodes complete
            if self.locals.get('dones') is not None:
                dones = self.locals['dones']
                if any(dones):
                    infos = self.locals.get('infos', [])
                    for i, done in enumerate(dones):
                        if done and i < len(infos) and 'episode' in infos[i]:
                            ep_info = infos[i]['episode']
                            self.episode_rewards.append(ep_info['r'])
                            self.episode_lengths.append(ep_info['l'])
            
            # Log training metrics every 100 episodes or 1000 steps
            log_condition = (len(self.episode_rewards) > 0 and len(self.episode_rewards) % 100 == 0) or (self.step_count % 1000 == 0)
            
            if log_condition:
                try:
                    # Log training episode statistics
                    if len(self.episode_rewards) > 0:
                        recent_rewards = self.episode_rewards[-100:]  # Last 100 episodes
                        recent_lengths = self.episode_lengths[-100:]
                        
                        self.logger.record("rollout/ep_rew_mean", np.mean(recent_rewards))
                        self.logger.record("rollout/ep_len_mean", np.mean(recent_lengths))
                        if len(recent_rewards) > 1:
                            self.logger.record("rollout/ep_rew_std", np.std(recent_rewards))
                    
                    # Get current environment state
                    env_instance = self.training_env.envs[0] if hasattr(self.training_env, 'envs') else self.training_env
                    
                    if hasattr(env_instance, 'unwrapped'):
                        env_instance = env_instance.unwrapped
                    
                    # Log formation metrics if available
                    if hasattr(env_instance, '_getDroneStateVector') and env_instance.NUM_DRONES > 0:
                        drone_states = np.array([env_instance._getDroneStateVector(i) for i in range(env_instance.NUM_DRONES)])
                        drones_poses = drone_states[:, :2]
                        
                        # Formation spacing
                        if len(drones_poses) > 1:
                            spacings = []
                            for i in range(len(drones_poses)):
                                for j in range(i + 1, len(drones_poses)):
                                    dist = np.linalg.norm(drones_poses[i] - drones_poses[j])
                                    spacings.append(dist)
                            
                            avg_spacing = np.mean(spacings)
                            spacing_std = np.std(spacings)
                            
                            # Log to tensorboard
                            self.logger.record("formation/avg_spacing", avg_spacing)
                            self.logger.record("formation/spacing_consistency", 1.0 / (1.0 + spacing_std))
                        
                        # Formation-herd distance
                        if hasattr(env_instance, 'HerdCentroid'):
                            herd_center = env_instance.HerdCentroid()[:2]
                            drone_center = np.mean(drones_poses, axis=0)
                            herd_distance = np.linalg.norm(drone_center - herd_center)
                            self.logger.record("formation/herd_distance", herd_distance)
                        
                        # Formation quality
                        if hasattr(env_instance, '_evaluate_formation_quality'):
                            formation_quality = env_instance._evaluate_formation_quality(drones_poses)
                            self.logger.record("formation/quality_score", formation_quality)
                            
                except Exception as e:
                    if self.verbose > 0:
                        print(f"Formation logging error: {e}")
            
            return True

    # Training
    if not eval_only:
        # Early stopping if performance degrades
        callback_on_best = StopTrainingOnRewardThreshold(
            reward_threshold=target_reward, verbose=1
        )
        
        # Custom callback to stop if performance collapses
        class EarlyStoppingCallback(BaseCallback):
            def __init__(self, patience=5, min_reward_threshold=10.0):
                super().__init__()
                self.patience = patience
                self.min_reward_threshold = min_reward_threshold
                self.best_reward = float('-inf')
                self.patience_counter = 0
                
            def _on_step(self):
                try:
                    # Check if ep_info_buffer exists and has data
                    if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                        # Handle different buffer types safely
                        buffer = self.model.ep_info_buffer
                        if hasattr(buffer, '__getitem__'):
                            # Get last 10 episodes, or all if fewer than 10
                            num_episodes = min(10, len(buffer))
                            recent_episodes = [buffer[i] for i in range(len(buffer) - num_episodes, len(buffer))]
                            recent_rewards = [ep_info['r'] for ep_info in recent_episodes if 'r' in ep_info]
                            
                            if recent_rewards:
                                current_reward = np.mean(recent_rewards)
                                
                                if current_reward > self.best_reward:
                                    self.best_reward = current_reward
                                    self.patience_counter = 0
                                elif current_reward < self.min_reward_threshold:
                                    self.patience_counter += 1
                                    if self.patience_counter >= self.patience:
                                        print(f"STOPPING: Performance collapsed below {self.min_reward_threshold}")
                                        return False
                except Exception as e:
                    # Silently continue if buffer access fails
                    pass
                return True
        
        # Disable early stopping for initial learning with new reward structure
        # early_stopping = EarlyStoppingCallback(patience=5, min_reward_threshold=-10.0)
        eval_callback = EvalCallback(
            eval_env,
            callback_on_new_best=callback_on_best,
            verbose=1,
            best_model_save_path=model_dir,
            log_path=model_dir,
            eval_freq=EVALUATION_FREQUENCY,
            deterministic=True,
            render=False
        )
        
        # Add formation logging
        formation_logger = FormationLoggingCallback(verbose=1)

        # Train without early stopping to allow learning new reward structure
        model.learn(total_timesteps=MAX_TIMESTEPS, callback=[eval_callback, formation_logger], log_interval=100)
        model.save(os.path.join(model_dir, "final_model.zip"))
        print(f"[LOG] Training finished. Model saved in {model_dir}")

        # Load best model for evaluation
        best_model_path = os.path.join(model_dir, "best_model.zip")
    else:
        # Evaluation only mode - use the same model directory
        # model_dir is already set above, so we just need to find the best model
        best_model_path = os.path.join(model_dir, "best_model.zip")

    if os.path.isfile(best_model_path):
        model = PPO.load(best_model_path)
        print(f"[LOG] Loaded best model from {best_model_path}")
    else:
        print(f"[ERROR] No best model found at {best_model_path}, using final_model.zip")
        model = PPO.load(os.path.join(model_dir, "final_model.zip"))

    # Create evaluation environment function
    def make_single_eval_env(use_gui=False):
        eval_kwargs = env_kwargs.copy()
        eval_kwargs['gui'] = use_gui
        eval_kwargs['eval_episode_length'] = EVAL_EPISODE_LENGTH
        env = CurriculumCattleAviary(**eval_kwargs)
        env.training_mode = False
        return env
    
    # Create ONLY ONE environment to avoid PyBullet multiple connection error
    test_env = Monitor(make_single_eval_env(use_gui=gui))
    test_env.is_evaluating = True
    if gui:
        test_env.record = record_video

    print("\n" + "="*60)
    print("🚁 CATTLE HERDING EVALUATION MODE")
    print("="*60)
    print(f"Model: {model_dir}")
    print(f"Episodes: {NUM_EVALUATION_EPS}")
    print(f"Episode Length: {EVAL_EPISODE_LENGTH} seconds")
    print(f"Single-core usage is NORMAL during evaluation")
    print("Evaluation runs sequentially through episodes...")
    print("="*60)

    # Evaluate trained policy with progress indicators
    mean_reward, std_reward = evaluate_policy_with_progress(model, test_env, n_eval_episodes=NUM_EVALUATION_EPS)
    if hasattr(test_env, 'evaluation_save'):
        test_env.evaluation_save()

    # Only prompt for input if not in eval_only mode
    if local and not eval_only:
        input("Press Enter to continue...")

    # Rollout for logging
    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                    num_drones=DEFAULT_DRONES, 
                    output_folder=output_folder,
                    colab=colab)
    
    obs, info = test_env.reset(seed=42, options={})
    start = time.time()

    for i in range((test_env.EPISODE_LEN_SEC + 2) * test_env.CTRL_FREQ):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        print(f"reward recieved: {reward}, terminated: {terminated}, trunacted: {truncated}")

        obs2 = np.atleast_2d(obs)
        act2 = np.atleast_2d(action)

        for d in range(DEFAULT_DRONES):
            logger.log(
                drone=d,
                timestamp=i / test_env.CTRL_FREQ,
                state=np.hstack([obs2[d][0:3], np.zeros(4), obs2[d][3:15], act2[d]]),
                control=np.zeros(12)
            )

        #test_env.render()
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated or truncated:
            obs, _ = test_env.reset(seed=42, options={})
            
    test_env.close()
    if plot:
        logger.plot()


# ---------------- CLI ---------------- #
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-drone cattle herding with PPO")
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool, help="Enable PyBullet GUI (default: True)")
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VIDEO, type=str2bool, help="Record a video (default: False)")
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str, help='Output folder for logs/models')
    parser.add_argument('--colab', default=DEFAULT_COLAB, type=bool, help="Notebook mode (default: False)")
    parser.add_argument('--drone', default=DroneModel.CF2X, type=DroneModel, help='Drone model (default: CF2X)')
    parser.add_argument('--num_drones', default=DEFAULT_DRONES, type=int, help=f'Number of drones (default: {DEFAULT_DRONES})')
    parser.add_argument('--num_cattle', default=DEFAULT_CATTLE, type=int, help=f'Number of cattle (default: {DEFAULT_CATTLE})')
    parser.add_argument('--obs', default=DEFAULT_OBS, type=ObservationType, help='Observation type (default: cokin)')
    parser.add_argument('--act', default=DEFAULT_ACT, type=ActionType, help='Action type (default: vel)')
    parser.add_argument('--eval_only', default=EVALUATE_ONLY, type=str2bool, help='Run evaluation only, no training')
    parser.add_argument('--eval_episode_length', default=EVAL_EPISODE_LENGTH, type=int, help=f'Episode length in seconds for evaluation (default: {EVAL_EPISODE_LENGTH})')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
