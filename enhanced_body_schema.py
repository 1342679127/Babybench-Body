import yaml
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from sklearn.cluster import MiniBatchKMeans
from babybench.utils import make_env
import matplotlib.pyplot as plt

# 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class EnhancedBodySchemaWrapper(gym.Wrapper):
    def __init__(self, env, debug=False):
        super().__init__(env)
        self.debug = debug

        # 传感器到身体部位映射
        self.body_parts = list(env.touch.sensor_positions.keys())
        self.sensor_to_body = {}
        for part, positions in env.touch.sensor_positions.items():
            for _ in positions:
                self.sensor_to_body[len(self.sensor_to_body)] = part

        # 空间记忆预分配缓冲
        max_points = 10000
        self.buf_body = np.zeros((max_points, 3), dtype=np.float32)
        self.buf_obj = np.zeros((max_points, 3), dtype=np.float32)
        self.buf_unknown = np.zeros((max_points, 3), dtype=np.float32)
        self.count = {'body':0, 'object':0, 'unknown':0}

        # 身体图式状态
        self.body_schema = {
            part: {'touch_count':0,
                   'target_touches':0,
                   'completed':False,
                   'positions': env.touch.sensor_positions[part][0],
                   'confidence':0.0}
            for part in self.body_parts
        }

        self._step_counter = 0
        self.reset_target()
        if self.debug:
            print("[DEBUG] Enhanced body schema initialized")

    def reset_target(self):
        incomplete = [p for p, d in self.body_schema.items() if not d['completed']]
        if not incomplete:
            self.current_target = None
            if self.debug:
                print("[DEBUG] All parts explored.")
            return
        confidences = [self.body_schema[p]['confidence'] for p in incomplete]
        min_conf = min(confidences)
        candidates = [p for p in incomplete if self.body_schema[p]['confidence'] <= min_conf + 0.1]
        self.current_target = np.random.choice(candidates)
        touches = np.random.randint(2, 6)
        self.body_schema[self.current_target]['target_touches'] = touches
        self.body_schema[self.current_target]['touch_count'] = 0
        if self.debug:
            print(f"[DEBUG] New target: {self.current_target}, touches: {touches}")

    def update_spatial(self, pos, is_body):
        if is_body:
            key = 'body'; buf = self.buf_body
        else:
            key = 'object'; buf = self.buf_obj
        idx = self.count[key]
        if idx < buf.shape[0]:
            buf[idx] = pos
            self.count[key] += 1
        if not is_body:
            # 如果非身体部位，检测是否靠近已知身体，若是则作为未知区域
            body_pts = self.buf_body[:self.count['body']]
            if body_pts.size and np.min(np.linalg.norm(body_pts - pos, axis=1)) < 0.1:
                idx2 = self.count['unknown']
                if idx2 < self.buf_unknown.shape[0]:
                    self.buf_unknown[idx2] = pos
                    self.count['unknown'] += 1

    def predict_next_target(self):
        cnt = self.count['unknown']
        if cnt == 0:
            return None
        pts = self.buf_unknown[:cnt]
        if cnt < 5:
            return pts.mean(axis=0)
        mbk = MiniBatchKMeans(n_clusters=min(3, cnt), batch_size=50, n_init=1)
        labels = mbk.fit_predict(pts)
        largest = np.argmax(np.bincount(labels))
        return mbk.cluster_centers_[largest]

    def compute_intrinsic_reward(self, obs):
        reward = 0.0
        touch_obs = obs.get('touch', None)
        if touch_obs is None:
            if self.debug:
                print("[DEBUG] No 'touch' observation in obs.")
            return reward
        active = np.nonzero(touch_obs > 1e-6)[0]
        for idx in active:
            part = self.sensor_to_body.get(idx, None)
            if part is None or part not in self.body_schema:
                if self.debug:
                    print(f"[DEBUG] Invalid sensor index {idx}, part={part}")
                continue
            pos = self.body_schema[part]['positions']
            self.update_spatial(pos, True)
            if part == self.current_target:
                s = self.body_schema[part]
                s['touch_count'] += 1
                reward += 0.2
                if s['touch_count'] >= s['target_touches']:
                    s['completed'] = True
                    s['confidence'] = 1.0
                    reward += 2.0
                    self.reset_target()
        if active.size > 0:
            reward += 0.1
        unk_cnt = self.count['unknown']
        if unk_cnt > 10:
            reward += min(0.05 * unk_cnt, 0.5)
        return reward

    def step(self, action):
        obs, ext_r, terminated, truncated, info = self.env.step(action)
        # 调试: 检查obs中非法键
        if any(k is None for k in obs.keys()):
            print(f"[WARNING] obs keys contain None: {obs.keys()}")
        int_r = self.compute_intrinsic_reward(obs)
        self._step_counter += 1
        if self._step_counter % 10 == 0:
            for p, d in self.body_schema.items():
                if not d['completed']:
                    d['confidence'] *= 0.95
        return obs, ext_r + int_r, terminated, truncated, info

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        for d in self.body_schema.values():
            d.update({'touch_count':0, 'target_touches':0, 'completed':False, 'confidence':0.0})
        for k in self.count:
            self.count[k] = 0
        self._step_counter = 0
        self.reset_target()
        return obs

    def visualize_body_schema(self, save_path="enhanced_body_schema.png"):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # 绘制身体部位
        for part, data in self.body_schema.items():
            x, y, z = data['positions']
            completion = (data['touch_count'] / data['target_touches']) if data['target_touches'] > 0 else 0.0
            size = 200 + 100 * min(completion, 1.0)
            ax.scatter(x, y, z, s=size, alpha=0.8, edgecolors='k')
            ax.text(x, y, z, f"{part}\n{data['touch_count']}/{data['target_touches']}\nC:{data['confidence']:.2f}",
                    fontsize=8, ha='center')
        # 绘制空间记忆点
        for key, buf in [('body', self.buf_body), ('object', self.buf_obj), ('unknown', self.buf_unknown)]:
            cnt = self.count[key]
            if cnt:
                pts = buf[:cnt]
                ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=30, alpha=0.5, label=key)
        next_t = self.predict_next_target()
        if next_t is not None:
            ax.scatter(*next_t, s=200, marker='*', alpha=0.9, label='predicted')
        ax.set_xlabel('X位置'); ax.set_ylabel('Y位置'); ax.set_zlabel('Z位置')
        ax.legend(); plt.tight_layout(); plt.savefig(save_path)
        if self.debug:
            print(f"[DEBUG] Saved visualization to {save_path}")
        plt.show()
        return next_t


def main():
    # 加载并修改配置
    config = yaml.safe_load(open("examples/config_selftouch.yml"))
    config.update({
        "training": True,
        "save_logs_every": 100,
        "max_episode_steps": 500,
        "save_dir": "results/enhanced_body_schema"
    })

    # 并行环境
    def make_wrapped():
        return EnhancedBodySchemaWrapper(make_env(config), debug=False)
    vec_env = make_vec_env(make_wrapped, n_envs=4, vec_env_cls=DummyVecEnv)

    model = PPO(
        "MultiInputPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        device="cpu"
    )
    model.learn(total_timesteps=50000)
    model.save("enhanced_body_schema_model")
    print("Model saved to enhanced_body_schema_model.zip")

    # 可视化示例
    wrapped = make_wrapped()
    wrapped.reset()
    wrapped.compute_intrinsic_reward({'touch': np.zeros(len(wrapped.sensor_to_body))})
    wrapped.visualize_body_schema()

if __name__ == "__main__":
    main()
