import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import babybench.utils as bb_utils
import gymnasium as gym

# 1. 定义预测模型 (基于VAE的变分预测编码器)
class PredictiveEncoder(nn.Module):
    """
    变分预测编码器
    基于预测加工理论：学习身体状态的动态表示
    """
    def __init__(self, input_dim, latent_dim, action_dim):
        super().__init__()
        # 存储维度信息
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        # 编码器：将感官输入映射到潜在空间
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)  # 输出均值和方差
        )
        
        # 解码器：从潜在空间重建感官输入
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )
        
        # 预测器：预测下一时刻状态
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 128),  # 使用实际动作维度
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x):
        """编码输入数据"""
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def decode(self, z):
        """解码潜在表示"""
        return self.decoder(z)
    
    def predict(self, z, action):
        """预测下一时刻状态"""
        return self.predictor(torch.cat([z, action], dim=-1))
    
    def forward(self, x, action, next_x):
        """完整前向传播"""
        # 编码当前状态
        z, mu, logvar = self.encode(x)
        
        # 重建当前状态
        recon_x = self.decode(z)
        
        # 预测下一状态
        pred_z = self.predict(z, action)
        pred_x = self.decode(pred_z)
        
        # 编码下一状态（用于计算预测误差）
        next_z, _, _ = self.encode(next_x)
        
        return recon_x, pred_x, z, next_z, mu, logvar

# 2. 定义环境包装器（计算内在奖励）
class PredictiveProcessingWrapper(gym.Wrapper):
    """预测加工环境包装器
    核心功能：根据预测误差计算内在奖励
    """
    def __init__(self, env, latent_dim=128, learning_rate=1e-4):
        super().__init__(env)
        
        # 获取观察空间维度
        proprio_dim = env.observation_space['observation'].shape[0]
        touch_dim = env.observation_space['touch'].shape[0]
        self.state_dim = proprio_dim + touch_dim
        self.action_dim = env.action_space.shape[0]  # 获取动作空间维度
        
        print(f"状态维度: {self.state_dim}, 动作维度: {self.action_dim}")
        
        # 初始化预测模型
        self.predictive_model = PredictiveEncoder(
            self.state_dim, 
            latent_dim, 
            self.action_dim  # 传入实际动作维度
        )
        self.optimizer = optim.Adam(self.predictive_model.parameters(), lr=learning_rate)
        
        # 存储前一个状态和动作
        self.last_state = None
        self.last_action = None
        
        # 损失函数
        self.recon_loss_fn = nn.MSELoss()
        self.pred_loss_fn = nn.MSELoss()
    
    def compute_intrinsic_reward(self, obs):
        """计算基于预测误差的内在奖励"""
        if self.last_state is None or self.last_action is None:
            return 0.0  # 初始状态无预测
        
        # 准备数据
        current_state = np.concatenate([
            self.last_state['observation'], 
            self.last_state['touch']
        ])
        next_state = np.concatenate([
            obs['observation'], 
            obs['touch']
        ])
        
        # 转换为张量
        current_tensor = torch.FloatTensor(current_state).unsqueeze(0)
        next_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        action_tensor = torch.FloatTensor(self.last_action).unsqueeze(0)
        
        # 前向传播
        recon_x, pred_x, z, next_z, mu, logvar = self.predictive_model(
            current_tensor, action_tensor, next_tensor
        )
        
        # 计算预测误差（KL散度 + 重建误差）
        recon_loss = self.recon_loss_fn(recon_x, current_tensor)
        pred_loss = self.pred_loss_fn(pred_x, next_tensor)
        
        # KL散度 (变分正则项)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # 总预测误差
        total_error = recon_loss + pred_loss + 0.1 * kl_div
        
        # 内在奖励 = 预测误差的负值（减少误差即获得奖励）
        intrinsic_reward = -total_error.item()
        
        return intrinsic_reward
    
    def update_model(self, obs):
        """更新预测模型"""
        if self.last_state is None or self.last_action is None:
            return
        
        # 准备数据
        current_state = np.concatenate([
            self.last_state['observation'], 
            self.last_state['touch']
        ])
        next_state = np.concatenate([
            obs['observation'], 
            obs['touch']
        ])
        
        # 转换为张量
        current_tensor = torch.FloatTensor(current_state).unsqueeze(0)
        next_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        action_tensor = torch.FloatTensor(self.last_action).unsqueeze(0)
        
        # 前向传播
        recon_x, pred_x, z, next_z, mu, logvar = self.predictive_model(
            current_tensor, action_tensor, next_tensor
        )
        
        # 计算损失
        recon_loss = self.recon_loss_fn(recon_x, current_tensor)
        pred_loss = self.pred_loss_fn(pred_x, next_tensor)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + pred_loss + 0.1 * kl_div
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
    
    def step(self, action):
        # 执行动作
        obs, extrinsic_reward, terminated, truncated, info = self.env.step(action)
        
        # 计算内在奖励
        intrinsic_reward = self.compute_intrinsic_reward(obs)
        
        # 更新预测模型
        self.update_model(obs)
        
        # 更新前一个状态和动作
        self.last_state = obs
        self.last_action = action
        
        # 总奖励 = 内在奖励 + 外部奖励（外部奖励始终为0）
        total_reward = intrinsic_reward + extrinsic_reward
        
        return obs, total_reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        # 重置环境
        obs = self.env.reset(**kwargs)
        
        # 重置状态跟踪
        self.last_state = obs
        self.last_action = None
        
        return obs

# 3. 训练回调（记录训练进度）
class TrainingCallback(BaseCallback):
    """自定义训练回调"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
    
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        # 计算平均奖励
        rewards = np.array([ep_info["r"] for ep_info in self.model.ep_info_buffer])
        if len(rewards) > 0:
            mean_reward = np.mean(rewards)
            self.episode_rewards.append(mean_reward)
            print(f"Step: {self.num_timesteps}, Mean Reward: {mean_reward:.2f}")
        return super()._on_rollout_end()

# 4. 主训练函数
def train_predictive_processing():
    # 加载配置文件
    config_path = 'examples/config_selftouch.yml'  # 自我触摸配置文件
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 创建环境
    env = bb_utils.make_env(config)
    print(f"环境创建成功: 观察空间={env.observation_space}, 动作空间={env.action_space}")
    
    # 应用预测加工包装器
    wrapped_env = PredictiveProcessingWrapper(env)
    print("预测加工包装器应用成功")
    
    # 向量化环境（Stable Baselines3要求）
    vec_env = DummyVecEnv([lambda: wrapped_env])
    
    # 初始化PPO模型
    model = PPO(
        policy="MultiInputPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1
    )
    
    # 创建回调
    callback = TrainingCallback()
    
    # 训练模型
    total_timesteps = 4000  # 总训练步数
    print(f"开始训练，总步数={total_timesteps}")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        log_interval=1
    )
    
    # 保存模型
    model.save("predictive_processing_model")
    print("训练完成，模型已保存")

# 5. 运行主函数
if __name__ == "__main__":
    # 设置随机种子（确保可重复性）
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 开始训练
    train_predictive_processing()