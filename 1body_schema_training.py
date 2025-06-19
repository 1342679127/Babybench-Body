# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 00:21:36 2025

@author: Administrator
"""

"""
自我触摸身体图式构建 - Self-Touch Body Schema Construction
基于BabyBench环境实现内在动机强化学习
婴儿通过主动触摸身体部位构建全身地图
"""

import yaml
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from babybench.utils import make_env
import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 身体图式构建包装器
class BodySchemaWrapper(gym.Wrapper):
    def __init__(self, env):
        """
        初始化身体图式包装器
        Initialize the body schema wrapper
        :param env: BabyBench环境实例
        :param env: BabyBench environment instance
        """
        super().__init__(env)
        
        # 获取身体部位信息
        # Get body parts information
        self.body_parts = list(env.touch.sensor_positions.keys())
        self.num_parts = len(self.body_parts)
        
        # 创建传感器到身体部位的映射
        # Create mapping from sensors to body parts
        self.sensor_to_body = {}
        for body_part, positions in env.touch.sensor_positions.items():
            for i in range(len(positions)):
                sensor_idx = len(self.sensor_to_body)
                self.sensor_to_body[sensor_idx] = body_part
        
        print(f"传感器总数/Total sensors: {len(self.sensor_to_body)}")
        print(f"身体部位/Body parts: {self.body_parts}")
        
        # 初始化身体图式状态
        # Initialize body schema state
        self.body_schema = {part: {
            'touch_count': 0,           # 总触摸次数
            'target_touches': 0,        # 目标触摸次数
            'completed': False,         # 是否完成探索
            'positions': positions[0]   # 位置信息（取第一个传感器）
        } for part, positions in env.touch.sensor_positions.items()}
        
        # 初始化当前目标部位
        # Initialize current target body part
        self.current_target = None
        self.reset_target()
        
        print(f"初始身体图式/Initial body schema created")

    def reset_target(self):
        """选择新的目标身体部位并设置随机触摸次数"""
        """Select a new target body part and set random touch count"""
        # 获取尚未完成探索的身体部位
        # Get body parts that haven't been completed
        incomplete_parts = [part for part, data in self.body_schema.items() if not data['completed']]
        
        if not incomplete_parts:
            # 所有部位都已完成探索
            # All parts have been explored
            print("所有身体部位探索完成！/All body parts explored!")
            self.current_target = None
            return
        
        # 随机选择一个未完成的部位
        # Randomly select an incomplete part
        self.current_target = np.random.choice(incomplete_parts)
        
        # 设置随机触摸次数 (2-5次)
        # Set random touch count (2-5 times)
        self.body_schema[self.current_target]['target_touches'] = np.random.randint(2, 6)
        self.body_schema[self.current_target]['touch_count'] = 0
        
        print(f"新目标/New target: {self.current_target}, 需要触摸/Required touches: {self.body_schema[self.current_target]['target_touches']}")

    def compute_intrinsic_reward(self, obs):
        """
        计算内在奖励
        Calculate intrinsic reward
        :param obs: 环境观察值
        :param obs: Environment observation
        :return: 内在奖励值
        :return: Intrinsic reward value
        """
        if self.current_target is None:
            return 0.0
        
        # 获取触摸传感器数据
        # Get touch sensor data
        touch_obs = obs['touch']
        reward = 0.0
        
        # 检查当前目标部位是否有触摸活动
        # Check for touch activity on the current target part
        target_active = False
        
        # 遍历所有传感器
        # Iterate through all sensors
        for sensor_idx in range(len(touch_obs)):
            # 获取此传感器对应的身体部位
            # Get body part corresponding to this sensor
            body_part = self.sensor_to_body.get(sensor_idx, None)
            
            # 如果传感器属于目标部位且被激活
            # If sensor belongs to target part and is activated
            if body_part == self.current_target and touch_obs[sensor_idx] > 1e-6:
                target_active = True
                break
        
        # 如果目标部位被触摸到
        # If target part is touched
        if target_active:
            # 增加触摸计数
            # Increment touch count
            self.body_schema[self.current_target]['touch_count'] += 1
            
            # 给予小奖励
            # Give small reward
            reward += 0.1
            
            print(f"部位/Part {self.current_target} 被触摸/touched ({self.body_schema[self.current_target]['touch_count']}/{self.body_schema[self.current_target]['target_touches']})")
            
            # 检查是否达到目标触摸次数
            # Check if target touch count reached
            if self.body_schema[self.current_target]['touch_count'] >= self.body_schema[self.current_target]['target_touches']:
                # 标记该部位为已完成
                # Mark part as completed
                self.body_schema[self.current_target]['completed'] = True
                
                # 给予大奖励
                # Give large reward
                reward += 1.0
                
                print(f"部位/Part {self.current_target} 完成探索/exploration completed!")
                
                # 选择新目标
                # Select new target
                self.reset_target()
        
        return reward

    def step(self, action):
        """
        环境步进函数
        Environment step function
        :param action: 动作向量
        :param action: Action vector
        :return: 更新后的观察值、奖励、终止标志等
        :return: Updated observation, reward, termination flag, etc.
        """
        # 执行动作
        # Execute action
        obs, extrinsic_reward, terminated, truncated, info = self.env.step(action)
        
        # 计算内在奖励
        # Calculate intrinsic reward
        intrinsic_reward = self.compute_intrinsic_reward(obs)
        
        # 组合奖励 (外部奖励始终为0)
        # Combine rewards (extrinsic reward is always 0)
        total_reward = intrinsic_reward + extrinsic_reward
        
        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        """重置环境并初始化目标"""
        """Reset environment and initialize target"""
        obs = super().reset(**kwargs)
        
        # 重置所有身体部位状态
        # Reset all body part states
        for part in self.body_parts:
            self.body_schema[part]['touch_count'] = 0
            self.body_schema[part]['target_touches'] = 0
            self.body_schema[part]['completed'] = False
        
        # 设置初始目标
        # Set initial target
        self.reset_target()
        
        return obs

    def visualize_body_schema(self, save_path="body_schema.png"):
        """
        可视化身体图式
        Visualize body schema
        :param save_path: 图像保存路径
        :param save_path: Image save path
        """
        # 创建3D可视化
        # Create 3D visualization
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # 设置颜色映射
        # Set colormap
        cmap = plt.get_cmap('viridis')
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        
        # 绘制每个身体部位
        # Plot each body part
        for part, data in self.body_schema.items():
            x, y, z = data['positions']
            
            # 计算完成度 (0-1)
            # Calculate completion (0-1)
            if data['target_touches'] > 0:
                completion = min(data['touch_count'] / data['target_touches'], 1.0)
            else:
                completion = 0.0
                
            color = cmap(norm(completion))
            
            # 绘制身体部位
            # Plot body part
            ax.scatter(x, y, z, s=200, c=color, alpha=0.7)
            
            # 添加标签
            # Add label
            ax.text(x, y, z, f"Part {part}\n{data['touch_count']}/{data['target_touches']}", 
                    fontsize=9, ha='center', va='bottom')
        
        # 添加颜色条
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.7)
        cbar.set_label('探索完成度/Exploration Completion', fontsize=12)
        
        # 设置坐标轴标签
        # Set axis labels
        ax.set_xlabel('X 位置/X Position', fontsize=12)
        ax.set_ylabel('Y 位置/Y Position', fontsize=12)
        ax.set_zlabel('Z 位置/Z Position', fontsize=12)
        
        # 设置标题
        # Set title
        ax.set_title('身体图式可视化/Body Schema Visualization', fontsize=16)
        
        # 调整视角
        # Adjust view angle
        ax.view_init(elev=30, azim=45)
        
        # 保存图像
        # Save image
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"身体图式已保存至/Saved to: {save_path}")
        
        # 显示图像
        # Display image
        plt.show()

def main():
    # 加载配置文件
    # Load configuration file
    config_path = "examples/config_selftouch.yml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 配置修改
    # Configuration modifications
    config["training"] = True
    config["save_logs_every"] = 100  # 每100步保存日志/Save logs every 100 steps
    config["max_episode_steps"] = 500  # 每episode最大步数/Max steps per episode
    config["save_dir"] = "results/body_schema_training"  # 设置保存目录/Set save directory
    
    # 创建环境
    # Create environment
    env = make_env(config)
    
    # 包装环境
    # Wrap environment
    wrapped_env = BodySchemaWrapper(env)
    
    # 创建矢量化环境
    # Create vectorized environment
    vec_env = DummyVecEnv([lambda: wrapped_env])
    
    # 初始化PPO模型
    # Initialize PPO model
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
        device="cpu"  # 使用CPU/Use CPU
    )
    
    # 训练模型
    # Train model
    total_timesteps = 50000
    model.learn(total_timesteps=total_timesteps)
    
    # 保存模型
    # Save model
    model.save("body_schema_model")
    print("模型已保存至/Saved to: body_schema_model.zip")
    
    # 可视化身体图式
    # Visualize body schema
    wrapped_env.visualize_body_schema()

if __name__ == "__main__":
    main()