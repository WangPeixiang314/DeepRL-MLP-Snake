import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.optim as optim

from config import Config
from memory import PrioritizedReplayBuffer
from model import DuelingDQN
from help import safe_action_nb


class DuelingDDQNAgent:
    """
    Dueling Double DQN Agent
    结合Dueling架构和Double DQN的优势
    """
    
    def __init__(self, input_dim):
        print(f"初始化Dueling Double DQN Agent")
        print(f"输入维度: {input_dim}")

        # 模型相关 - 使用Dueling DQN
        self.policy_net = DuelingDQN(input_dim, Config.HIDDEN_LAYERS, Config.OUTPUT_DIM)
        self.target_net = DuelingDQN(input_dim, Config.HIDDEN_LAYERS, Config.OUTPUT_DIM)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # 优化器 - 根据配置选择
        if Config.OPTIMIZER == 'AdamW':
            self.optimizer = optim.AdamW(
                self.policy_net.parameters(), 
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY
            )
        elif Config.OPTIMIZER == 'Adam':
            self.optimizer = optim.Adam(
                self.policy_net.parameters(), 
                lr=Config.LEARNING_RATE
            )
        else:
            self.optimizer = optim.AdamW(
                self.policy_net.parameters(), 
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY
            )
        
        # 学习率调度器
        if Config.LR_SCHEDULER == 'CosineAnnealing':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=Config.COSINE_T_MAX, eta_min=1e-6
            )
        else:
            self.scheduler = None
        
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
        
        # 训练指标
        self.losses = []
        self.td_errors = []
        
        # 尝试加载模型
        self.load_model()
        
        # 开启训练模式
        self.policy_net.train()
    
    def select_action(self, state):
        """选择动作（Dueling DDQN版本）"""
        eps_threshold = Config.EPS_END + (Config.EPS_START - Config.EPS_END) * \
                        np.exp(-1. * self.steps_done / Config.EPS_DECAY)
        self.steps_done += 1
        self.epsilon_threshold = eps_threshold

        # 获取当前状态的危险信息
        danger_signals = state[:3]
        
        # ε-greedy策略
        if np.random.random() > eps_threshold:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(Config.device)
                
                # 使用Dueling网络获取Q值
                q_values = self.policy_net(state_tensor).cpu().numpy().flatten()
                
                # 使用防自杀机制选择动作
                action, danger_actions = safe_action_nb(
                    q_values, 
                    danger_signals,
                    Config.COLLISION_PENALTY,
                    state
                )
                
                # 记录危险选择的惩罚
                for action_idx in danger_actions:
                    self.memory.add(
                        state, 
                        action_idx, 
                        Config.COLLISION_PENALTY * 0.5,
                        state,
                        True
                    )
                
                return action
        else:
            # 随机探索时也避免危险动作
            safe_actions = [i for i in range(Config.OUTPUT_DIM) 
                          if danger_signals[i] <= 0.5]
            return (np.random.choice(safe_actions) if safe_actions 
                   else np.random.randint(Config.OUTPUT_DIM))
    
    def optimize_model(self):
        """优化模型（Dueling DDQN版本）"""
        if len(self.memory) < Config.BATCH_SIZE:
            return 0.0, []
        
        # 从内存中采样批次（带权重）
        (states, actions, rewards, next_states, dones, 
        indices, weights) = self.memory.sample(Config.BATCH_SIZE)
        
        weights = weights.detach().clone().to(Config.device)
        
        # 使用Dueling网络计算当前Q值
        state_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Dueling DDQN的目标计算
        with torch.no_grad():
            # 使用当前网络选择下一个状态的最佳动作
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            
            # 使用目标网络评估这些动作的价值（Dueling结构）
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
        
        # 计算期望Q值
        expected_q_values = rewards + (1 - dones) * Config.GAMMA * next_q_values
        
        # 计算Huber损失（比L1损失更稳定）
        loss = F.smooth_l1_loss(state_q_values.squeeze(), expected_q_values, reduction='none')
        weighted_loss = (weights * loss).mean()
        
        # 计算TD误差
        td_errors = loss.detach().cpu().numpy()
        
        # 优化模型
        self.optimizer.zero_grad()
        weighted_loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=Config.GRADIENT_CLIP)
        
        self.optimizer.step()
        
        # 更新学习率（如果启用了调度器）
        if self.scheduler is not None:
            self.scheduler.step()
        
        # 更新优先级
        self.memory.update_priorities(indices, td_errors)
        
        # 记录训练指标
        self.losses.append(weighted_loss.item())
        self.td_errors.extend(td_errors)
        
        return weighted_loss.item(), td_errors
    
    def update_target_net(self):
        """软更新目标网络"""
        if Config.TARGET_SOFT_UPDATE:
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(Config.TARGET_TAU * policy_param.data + (1.0 - Config.TARGET_TAU) * target_param.data)
    
    def hard_update_target_net(self):
        """硬更新目标网络（完全复制）"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self, is_best=False):
        """保存模型（包含Dueling信息）"""
        suffix = f"_best_{self.best_score}.pth" if is_best else ".pth"
        score = self.scores[-1] if self.scores else 0
        filename = f"snake_dueling_dqn_ep{self.episode}_sc{score}{suffix}"
        
        # 保存完整检查点
        checkpoint = {
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'episode': self.episode,
            'best_score': self.best_score,
            'scores': self.scores,
            'model_type': 'dueling_dqn',
            'config': {
                'hidden_layers': Config.HIDDEN_LAYERS,
                'learning_rate': Config.LEARNING_RATE,
                'gamma': Config.GAMMA
            }
        }
        
        path = os.path.join(Config.MODEL_DIR, filename)
        torch.save(checkpoint, path)
        
        # 保存最佳模型（简化版）
        if is_best:
            best_path = os.path.join(Config.MODEL_DIR, 'snake_dueling_dqn_best.pth')
            torch.save(checkpoint, best_path)
            print(f"新的最佳Dueling DQN模型已保存: {best_path}")
            print(f"Dueling DQN模型已保存: {path}")
    
    def load_model(self, filename=None):
        """加载模型"""
        if filename is None:
            filename = 'snake_dueling_dqn_best.pth'
        
        path = os.path.join(Config.MODEL_DIR, filename)
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location=Config.device)
                
                if 'model_type' in checkpoint and checkpoint['model_type'] == 'dueling_dqn':
                    self.policy_net.load_state_dict(checkpoint['model_state_dict'])
                    
                    if 'optimizer_state_dict' in checkpoint:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    
                    if 'scheduler_state_dict' in checkpoint:
                        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    
                    if 'episode' in checkpoint:
                        self.episode = checkpoint['episode']
                    
                    if 'best_score' in checkpoint:
                        self.best_score = checkpoint['best_score']
                    
                    if 'scores' in checkpoint:
                        self.scores = checkpoint['scores']
                    
                    self.policy_net.eval()
                    print(f"Dueling DQN模型已加载: {path}")
                    print(f"当前最佳分数: {self.best_score}")
                    return True
                else:
                    print("尝试加载的模型不是Dueling DQN格式")
                    return False
                    
            except Exception as e:
                print(f"❌ 加载模型时出错: {e}")
                return False
        return False
    
    def record_score(self, score):
        """记录分数"""
        self.scores.append(score)
        
        # 更新最高分
        if score > self.best_score:
            self.best_score = score
            self.save_model(is_best=True)
            print(f"🏆 新记录! 分数: {score}")
        
        # 定期保存模型
        if self.episode % 100 == 0:
            self.save_model()
        
        # 定期更新目标网络
        if self.episode % Config.TARGET_UPDATE == 0:
            self.hard_update_target_net()
            print("🔄 目标网络已更新")
        
        self.episode += 1
    
    def get_stats(self):
        """获取训练统计信息"""
        return {
            'episode': self.episode,
            'best_score': self.best_score,
            'current_score': self.scores[-1] if self.scores else 0,
            'epsilon': self.epsilon_threshold,
            'learning_rate': self.scheduler.get_last_lr()[0],
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0,
            'avg_td_error': np.mean(self.td_errors[-1000:]) if self.td_errors else 0
        }
    
    def reset_noise(self):
        """重置噪声（如果使用Noisy Networks）"""
        if hasattr(self.policy_net, 'reset_noise'):
            self.policy_net.reset_noise()
        if hasattr(self.target_net, 'reset_noise'):
            self.target_net.reset_noise()


# 兼容性别名
DuelingDoubleDQNAgent = DuelingDDQNAgent