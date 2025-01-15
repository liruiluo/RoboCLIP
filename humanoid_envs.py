from gymnasium import Env, spaces
import numpy as np
from stable_baselines3 import PPO
import torch as th
from s3dg import S3D
from gymnasium.wrappers.time_limit import TimeLimit
# from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from PIL import Image, ImageSequence
import torch as th
from s3dg import S3D
import numpy as np
from PIL import Image, ImageSequence
import cv2
import gif2numpy
import PIL
import os
import seaborn as sns
import matplotlib.pylab as plt

from typing import Any, Dict

import gymnasium as gym
from gymnasium.spaces import Box
import torch as th

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video
import os
from stable_baselines3.common.monitor import Monitor
from memory_profiler import profile
import argparse
# from stable_baselines3.common.callbacks import EvalCallback
from callbacks import EvalCallback

# from kitchen_env_wrappers import readGif
from matplotlib import animation
import matplotlib.pyplot as plt
from prompts import TASKS, TASKS_TARGET
# from humanoid_bench import TASKS
import henv
import wandb
from callbacks import ProgressBarCallback

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--algo', type=str, default='ppo')
    parser.add_argument('--text-string', type=str, default='robot opening sliding door')
    parser.add_argument('--dir-add', type=str, default='')
    parser.add_argument('--env-id', type=str, default='h1hand-run-customized-v0')
    parser.add_argument('--env-type', type=str, default='h1hand-run-customized-v0')
    parser.add_argument('--total-time-steps', type=int, default=10000000)
    parser.add_argument('--n-envs', type=int, default=8)
    parser.add_argument('--n-steps', type=int, default=128)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)


    args = parser.parse_args()
    return args
class MetaworldSparse(Env):
    def __init__(self, env_id, text_string=None, time=False, video_path=None, rank=0, human=True):
        super(MetaworldSparse,self)
        # door_open_goal_hidden_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[env_id]
        # env = door_open_goal_hidden_cls(seed=rank)
        env= gym.make(env_id)
        env.action_space.seed(rank)
        self.env = TimeLimit(env, max_episode_steps=128)
        self.time = time
        if not self.time:
            self.observation_space = self.env.observation_space
        else:
            self.observation_space = Box(low=-8.0, high=8.0, shape=(self.env.observation_space.shape[0]+1,), dtype=np.float32)
        self.action_space = self.env.action_space
        self.past_observations = []
        self.window_length = 16
        self.net = S3D('s3d_dict.npy', 512)

        # Load the model weights
        self.net.load_state_dict(th.load('s3d_howto100m.pth'))
        # Evaluation mode
        self.net = self.net.eval()
        self.target_embedding = None
        if text_string:
            text_output = self.net.text_module([text_string])
            self.target_embedding = text_output['text_embedding']
        if video_path:
            frames = readGif(video_path)
            
            if human:
                frames = self.preprocess_human_demo(frames)
            else:
                frames = self.preprocess_metaworld(frames)
            if frames.shape[1]>3:
                frames = frames[:,:3]
            video = th.from_numpy(frames)
            video_output = self.net(video.float())
            self.target_embedding = video_output['video_embedding']
        assert self.target_embedding is not None

        self.counter = 0

    def get_obs(self):
        return self.baseEnv._get_obs(self.baseEnv.prev_time_step)

    def preprocess_human_demo(self, frames):
        frames = np.array(frames)
        frames = frames[None, :,:,:,:]
        frames = frames.transpose(0, 4, 1, 2, 3)
        return frames

    # def preprocess_metaworld(self, frames, shorten=True):
    #     center = 240, 320
    #     h, w = (250, 250)
    #     x = int(center[1] - w/2)
    #     y = int(center[0] - h/2)
    #     frames = np.array([cv2.resize(frame, dsize=(250, 250), interpolation=cv2.INTER_CUBIC) for frame in frames])
    #     frames = np.array([frame[y:y+h, x:x+w] for frame in frames])
    #     a = frames
    #     frames = frames[None, :,:,:,:]
    #     frames = frames.transpose(0, 4, 1, 2, 3)
    #     if shorten:
    #         frames = frames[:, :,::4,:,:]
    #     # frames = frames/255
    #     print(frames.shape)
    #     exit()
    #     return frames
    def preprocess_metaworld(self, frames):
        # frames = np.array(frames) # (35, 256, 256, 3)
        frames = np.array([cv2.resize(frame, dsize=(224, 224), interpolation=cv2.INTER_CUBIC) for frame in frames])
        frames = frames[None, :,:,:,:]
        frames = frames.transpose(0, 4, 1, 2, 3)
        frames = frames[:, :,::4,:,:] # (1, 3, 9, 256, 256)
        return frames
    
    def render(self):
        frame = self.env.render()
        # center = 240, 320
        # h, w = (250, 250)
        # x = int(center[1] - w/2)
        # y = int(center[0] - h/2)
        # frame = frame[y:y+h, x:x+w]
        return frame


    def step(self, action):
        obs, task_reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.past_observations.append(self.env.render())
        self.counter += 1
        t = self.counter/128
        if self.time:
            obs = np.concatenate([obs, np.array([t])])
        if done:
            frames = self.preprocess_metaworld(self.past_observations)
            
        
        
            video = th.from_numpy(frames) # (1, 3, 9, 256, 256)

            video_output = self.net(video.float())

            video_embedding = video_output['video_embedding']
            similarity_matrix = th.matmul(self.target_embedding, video_embedding.t())

            reward = similarity_matrix.detach().numpy()[0][0]
            return obs, reward*0.001+task_reward, terminated, truncated, info
        return obs, task_reward, terminated, truncated, info

    def reset(self):
        self.past_observations = []
        self.counter = 0
        if not self.time:
            return self.env.reset()
        return np.concatenate([self.env.reset()[0], np.array([0.0])]),self.env.reset()[1]


class MetaworldDense(Env):
    def __init__(self, env_id, time=False, rank=0):
        super(MetaworldDense,self)
        # door_open_goal_hidden_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[env_id]
        # env = door_open_goal_hidden_cls(seed=rank)
        env = gym.make(env_id)
        self.env = TimeLimit(env, max_episode_steps=128)
        self.time = time
        if not self.time:
            self.observation_space = self.env.observation_space
        else:
            self.observation_space = Box(low=-8.0, high=8.0, shape=(self.env.observation_space.shape[0]+1,), dtype=np.float32)
        self.action_space = self.env.action_space
        self.past_observations = []
        
        self.counter = 0

    def get_obs(self):
        return self.baseEnv._get_obs(self.baseEnv.prev_time_step)
        
    
    def render(self):
        # camera_name="topview"
        frame = self.env.render()
        return frame


    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # self.past_observations.append(self.env.render())
        self.counter += 1
        t = self.counter/128
        if self.time:
            obs = np.concatenate([obs, np.array([t])])
        return obs, reward, terminated, truncated, info
        
    def reset(self):
        self.counter = 0
        if not self.time:
            return self.env.reset()
        # return np.concatenate([self.env.reset(), np.array([0.0])])
        return np.concatenate([self.env.reset()[0], np.array([0.0])]),self.env.reset()[1]




def make_env(env_type, env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        # env = KitchenMicrowaveHingeSlideV0()
        if env_type == "sparse_learnt":
            env = MetaworldSparse(env_id=env_id, text_string=TASKS[env_id], time=True, rank=rank)
            # env = MetaworldSparse(env_id=env_id, video_path="./gifs/human_opening_door.gif", time=True, rank=rank, human=True)
        
        elif env_type == "sparse_original":
            env = KitchenEnvSparseOriginalReward(time=True)
        else:
            env = MetaworldDense(env_id=env_id, time=True, rank=seed + rank)
        env = Monitor(env, os.path.join(log_dir, str(rank)))
        # env.env.seed(seed + rank)
        return env
    # set_global_seeds(seed)
    return _init

class CustomWandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _on_step(self) -> bool:
        # Get the current info from the environment
        for info in self.locals['infos']:
            if 'episode' in info.keys():
                episode_info = info['episode']
                wandb.log({
                    'train/episode_reward': episode_info['r'],
                    'train/episode_length': episode_info['l']
                }, step=self.num_timesteps)
                
                self.episode_rewards.append(episode_info['r'])
                self.episode_lengths.append(episode_info['l'])

        # Log training metrics
        if self.n_calls % 1000 == 0:  # Log every 1000 steps
            wandb.log({
                'train/mean_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
                'train/mean_length': np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0,
                'train/timesteps': self.num_timesteps,
                'train/policy_loss': self.locals['self'].logger.name_to_value['train/policy_loss'],
                'train/value_loss': self.locals['self'].logger.name_to_value['train/value_loss'],
                'train/entropy_loss': self.locals['self'].logger.name_to_value.get('train/entropy_loss', 0)
            }, step=self.num_timesteps)
        
        return True
from wandb.integration.sb3 import WandbCallback
def main():
    global args
    global log_dir
    args = get_args()
    log_dir = f"output/humanoid/{args.env_id}_{args.env_type}{args.seed}"
    run = wandb.init(
        project="s3dg",
        name=f"{args.env_id}_{args.seed}",
        config={
            "algo": args.algo,
            "text_string": args.text_string,
            "env_id": args.env_id,
            "env_type": args.env_type,
            "total_time_steps": args.total_time_steps,
            "n_envs": args.n_envs,
            "n_steps": args.n_steps,
            "pretrained": args.pretrained,
            "seed": args.seed
        },
        sync_tensorboard=True,
        dir=log_dir,
    )

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    envs = SubprocVecEnv([make_env(args.env_type, args.env_id, args.seed + i) for i in range(args.n_envs)])

    if not args.pretrained:
        model = PPO("MlpPolicy", envs, verbose=1, tensorboard_log=log_dir, n_steps=args.n_steps, batch_size=args.n_steps*args.n_envs, n_epochs=1, ent_coef=0.5)
    else:
        model = PPO.load(args.pretrained, env=envs, tensorboard_log=log_dir)

    eval_env = SubprocVecEnv([make_env("dense_original", args.env_id, args.seed + i) for i in range(10, 10+args.n_envs)])#KitchenEnvDenseOriginalReward(time=True)
    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                                 log_path=log_dir, eval_freq=5000,
                                 deterministic=True, render=False)
    wandb_callback_1 = WandbCallback(verbose=2)
    progress_callback = ProgressBarCallback(total_timesteps=args.total_time_steps)

    model.learn(total_timesteps=int(args.total_time_steps), callback=[eval_callback,wandb_callback_1, progress_callback])
    model.save(f"{log_dir}/trained")



if __name__ == '__main__':
    main()
