# -*- coding: utf-8 -*-
"""
自我触摸训练脚本 - Self-Touch Training with Itch Motivation
修复版：解决'touch_sensors'属性不存在的问题
"""

import yaml
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from babybench.utils import make_env
import matplotlib.pyplot as plt

# 修复版瘙痒包装器
class ItchWrapper(gym.Wrapper):
    def __init__(self, env):
        """
        初始化瘙痒包装器
        :param env: BabyBench环境实例
        """
        super().__init__(env)
        
        # 获取身体部位信息
        self.body_parts = list(env.touch.sensor_positions.keys())
        self.num_parts = len(self.body_parts)
        
        # 创建传感器到身体部位的映射
        self.sensor_to_body = {}
        for body_part, positions in env.touch.sensor_positions.items():
            for i in range(len(positions)):
                # 为每个传感器分配唯一的索引
                sensor_idx = len(self.sensor_to_body)
                self.sensor_to_body[sensor_idx] = body_part
        
        print(f"传感器总数: {len(self.sensor_to_body)}")
        print(f"身体部位: {self.body_parts}")
        
        # 初始化瘙痒状态和身体图式
        self.current_itch = None
        self.body_schema = {part: 0 for part in self.body_parts}
        self.reset_itch()
        
        print(f"初始身体图式: {self.body_schema}")

    def reset_itch(self):
        """随机生成新的瘙痒部位"""
        self.current_itch = np.random.choice(self.body_parts)
        print(f"新瘙痒部位: {self.current_itch}")

    def compute_intrinsic_reward(self, obs):
        """
        计算内在奖励
        :param obs: 环境观察值
        :return: 内在奖励值
        """
        # 获取触摸传感器数据
        touch_obs = obs['touch']
        
        # 检查当前瘙痒部位是否有触摸活动
        itch_active = False
        
        # 遍历所有传感器
        for sensor_idx in range(len(touch_obs)):
            # 获取此传感器对应的身体部位
            body_part = self.sensor_to_body.get(sensor_idx, None)
            
            # 如果传感器属于瘙痒部位且被激活
            if body_part == self.current_itch and touch_obs[sensor_idx] > 1e-6:
                itch_active = True
                break
        
        # 如果瘙痒被触摸到
        if itch_active:
            # 更新身体图式
            self.body_schema[self.current_itch] += 1
            
            # 生成新瘙痒
            self.reset_itch()
            
            # 返回正奖励
            return 1.0
        
        # 没有触摸到瘙痒
        return 0.0

    def step(self, action):
        """
        环境步进函数
        :param action: 动作向量
        :return: 更新后的观察值、奖励、终止标志等
        """
        # 执行动作
        obs, extrinsic_reward, terminated, truncated, info = self.env.step(action)
        
        # 计算内在奖励
        intrinsic_reward = self.compute_intrinsic_reward(obs)
        
        # 组合奖励 (外部奖励始终为0)
        total_reward = intrinsic_reward + extrinsic_reward
        
        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        """重置环境并生成初始瘙痒"""
        obs = super().reset(**kwargs)
        self.reset_itch()
        return obs

    def plot_body_schema(self, save_path="body_schema.png"):
        """
        绘制身体图式
        :param save_path: 图像保存路径
        """
        # 提取身体部位和触摸次数
        parts = list(self.body_schema.keys())
        counts = list(self.body_schema.values())
        
        # 创建图表
        plt.figure(figsize=(12, 8))
        
        # 创建水平条形图
        plt.barh(
            [str(p) for p in parts],  # 确保所有部位名称都是字符串
            counts, 
            color='skyblue'
        )
        
        # 添加标签和标题
        plt.xlabel('触摸次数', fontsize=14)
        plt.ylabel('身体部位ID', fontsize=14)
        plt.title('身体图式 - 各部位触摸频率', fontsize=16)
        
        # 添加网格线
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # 调整布局并保存
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"身体图式已保存至: {save_path}")

def main():
    # 加载配置文件
    config_path = "examples/config_selftouch.yml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 配置修改
    config["training"] = True
    config["save_logs_every"] = 100  # 每100步保存日志
    config["max_episode_steps"] = 500  # 每episode最大步数
    config["save_dir"] = "results/itch_training"  # 明确设置保存目录
    
    # 创建环境
    env = make_env(config)
    
    # 包装环境
    wrapped_env = ItchWrapper(env)
    
    # 创建矢量化环境
    vec_env = DummyVecEnv([lambda: wrapped_env])
    
    # 初始化PPO模型
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        device="auto"
    )
    
    # 训练模型
    total_timesteps = 50000
    model.learn(total_timesteps=total_timesteps)
    
    # 保存模型
    model.save("itch_touch_model")
    print("模型已保存至: itch_touch_model.zip")
    
    # 绘制身体图式
    wrapped_env.plot_body_schema()

if __name__ == "__main__":
    main()