import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from config import Config
from memory import PrioritizedReplayBuffer
from model import DQN
from help import safe_action_nb

class DQNAgent:
    def __init__(self, input_dim):
        # 模型相关
        self.policy_net = DQN(input_dim, Config.HIDDEN_LAYERS, Config.OUTPUT_DIM)
        self.target_net = DQN(input_dim, Config.HIDDEN_LAYERS, Config.OUTPUT_DIM)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # 优化器
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=Config.LEARNING_RATE)
        
        # 经验回放（使用优先级）
        self.memory = PrioritizedReplayBuffer(
            Config.MEMORY_CAPACITY,
            alpha=Config.PRIO_ALPHA,
            beta_start=Config.PRIO_BETA_START,
            beta_frames=Config.PRIO_BETA_FRAMES
        )
        
        # 训练参数
        self.steps_done = 0
        self.epsilon_threshold = Config.EPS_START
        self.episode = 0
        self.scores = []
        self.best_score = 0
        
        # 尝试加载模型
        self.policy_net.load()
        
        # 开启训练模式
        self.policy_net.train()
    
    def select_action(self, state):
        """选择动作（带numba加速的防自杀机制）"""
        eps_threshold = Config.EPS_END + (Config.EPS_START - Config.EPS_END) * \
                        np.exp(-1. * self.steps_done / Config.EPS_DECAY)
        self.steps_done += 1
        self.epsilon_threshold = eps_threshold

        # 获取当前状态的危险信息（前3个元素代表前方/右方/左方的危险）
        danger_signals = state[:3]
        
        # ε-greedy策略
        if np.random.random() > eps_threshold:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(Config.device)
                q_values = self.policy_net(state_tensor).cpu().numpy().flatten()
                
                # 使用numba加速的防自杀机制选择动作
                action, danger_actions = safe_action_nb(
                    q_values, 
                    danger_signals,
                    Config.COLLISION_PENALTY,
                    state
                )
                
                # 记录危险选择的惩罚（用于训练）
                for action_idx in danger_actions:
                    self.memory.add(
                        state, 
                        action_idx, 
                        Config.COLLISION_PENALTY * 0.5,  # 给予部分惩罚
                        state,  # 保持状态不变
                        True    # 标记为终止状态
                    )
                
                return action
        else:
            # 随机探索时也避免危险动作
            safe_actions = [i for i in range(Config.OUTPUT_DIM) 
                          if danger_signals[i] <= 0.5]
            return (np.random.choice(safe_actions) if safe_actions 
                   else np.random.randint(Config.OUTPUT_DIM))

    def optimize_model(self):
        """优化模型（采用优先级采样）"""
        if len(self.memory) < Config.BATCH_SIZE:
            return 0.0, []
        
        # 从内存中采样批次（带权重）
        (states, actions, rewards, next_states, dones, 
        indices, weights) = self.memory.sample(Config.BATCH_SIZE)
        
        # 修复警告：使用 detach().clone() 替代 torch.tensor()
        weights = weights.detach().clone().to(Config.device)
        
        # 计算当前状态的Q值
        state_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # 计算下一个状态的最大Q值
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
        
        # 计算期望Q值 (Bellman方程)
        expected_q_values = rewards + (1 - dones) * Config.GAMMA * next_q_values
        
        # 计算损失（加入重要性采样权重）
        loss = F.smooth_l1_loss(state_q_values.squeeze(), expected_q_values, reduction='none')
        weighted_loss = (weights * loss).mean()
        
        # 计算TD误差（用于优先级）
        td_errors = loss.detach().cpu().numpy()
        
        # 优化模型
        self.optimizer.zero_grad()
        weighted_loss.backward()
        
        # 梯度裁剪
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        
        self.optimizer.step()
        
        # 更新优先级
        self.memory.update_priorities(indices, td_errors)
        
        return weighted_loss.item(), td_errors

    def update_target_net(self):
        """更新目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self, is_best=False):
        """保存模型"""
        suffix = f"_best_{self.best_score}.pth" if is_best else ".pth"
        score = self.scores[-1] if self.scores else 0
        filename = f"snake_dqn_ep{self.episode}_sc{score}{suffix}"
        self.policy_net.save(filename)
        
        # 保存最佳模型
        if is_best:
            self.policy_net.save("snake_dqn_best.pth")
    
    def record_score(self, score):
        """记录分数"""
        self.scores.append(score)
        
        # 更新最高分
        if score > self.best_score:
            self.best_score = score
            self.save_model(is_best=True)
            print(f"新记录! 分数: {score}")
        
        # 定期保存模型
        if self.episode % 200 == 0:
            self.save_model()
        
        # 定期更新目标网络
        if self.episode % Config.TARGET_UPDATE == 0:
            self.update_target_net()
            print("目标网络已更新")
        
        self.episode += 1