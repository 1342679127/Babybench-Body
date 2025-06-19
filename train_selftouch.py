# -*- coding: utf-8 -*-
"""
Created on Fri May 30 01:32:06 2025

@author: Administrator
"""

# 安装必要依赖 (运行前执行)
# pip install gymnasium stable-baselines3 pyyaml numpy babybench2025

import os
import yaml
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from babybench.utils import make_env

# 定义环境包装器 - 添加内在奖励
class SelfTouchWrapper(gym.Wrapper):
    """环境包装器，为自我触摸行为计算内在奖励"""
    
    def __init__(self, env):
        super().__init__(env)
        # 记录最大触摸传感器数（用于归一化）
        self.max_touch_sensors = env.observation_space['touch'].shape[0]
        
    def compute_intrinsic_reward(self, obs):
        """计算内在奖励：激活的触摸传感器比例"""
        # 统计激活的触摸传感器数量（阈值>1e-6）
        active_sensors = np.sum(obs['touch'] > 1e-6)
        # 归一化为[0,1]范围
        return active_sensors / self.max_touch_sensors
    
    def step(self, action):
        """重写step方法，添加内在奖励"""
        obs, extrinsic_reward, terminated, truncated, info = self.env.step(action)
        intrinsic_reward = self.compute_intrinsic_reward(obs)
        # 组合奖励（外部奖励始终为0）
        total_reward = intrinsic_reward + extrinsic_reward
        return obs, total_reward, terminated, truncated, info

# 主训练函数
def train_selftouch():
    """训练自我触摸行为的强化学习智能体"""
    
    # 加载配置文件
    config_path = 'examples/config_selftouch.yml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 创建环境
    env = make_env(config, training=True)
    # 应用包装器添加内在奖励
    wrapped_env = SelfTouchWrapper(env)
    
    # 创建PPO模型（最简单的可靠RL算法）
    model = PPO(
        policy="MultiInputPolicy",
        env=wrapped_env,
        learning_rate=3e-4,
        n_steps=2048,        # 每次更新收集的步数
        batch_size=64,       # 训练批大小
        gamma=0.99,          # 折扣因子
        gae_lambda=0.95,     # GAE参数
        ent_coef=0.01,       # 熵系数（鼓励探索）
        verbose=1            # 显示训练日志
    )
    
    # 训练模型
    total_timesteps = 100000  # 总训练步数
    model.learn(total_timesteps=total_timesteps)
    
    # 保存模型
    model.save("selftouch_ppo_model")
    print(f"训练完成！模型已保存至: selftouch_ppo_model.zip")
    
    return model

# 评估函数
def evaluate_model(model):
    """评估训练好的模型"""
    
    # 加载配置文件（使用相同配置）
    config_path = 'examples/config_selftouch.yml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 创建评估环境（关闭训练日志）
    config["save_logs_every"] = 0
    env = make_env(config, training=False)
    wrapped_env = SelfTouchWrapper(env)
    
    # 运行10个测试episode
    total_rewards = []
    for ep in range(10):
        obs, _ = wrapped_env.reset()
        episode_reward = 0
        terminated = truncated = False
        
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = wrapped_env.step(action)
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        print(f"Episode {ep+1}: 总奖励={episode_reward:.2f}")
    
    print(f"\n平均奖励: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")

if __name__ == "__main__":
    # 训练模型
    trained_model = train_selftouch()
    
    # 评估模型性能
    print("\n开始评估模型...")
    evaluate_model(trained_model)