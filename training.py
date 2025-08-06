import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

n_mean = 500

# 配置中文显示
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False

# 设置图表样式
style.use('ggplot')

class TrainingStats:
    def __init__(self):
        # 训练数据存储
        self.scores = []          # 每局分数
        self.rewards = []         # 每局总奖励
        self.steps = []           # 每局步数
        self.losses = []          # 每局平均损失
        self.epsilons = []        # 每局探索率
        
        # 移动平均数据
        self.avg_scores = []      # 分数移动平均
        self.avg_rewards = []     # 奖励移动平均
        self.avg_steps = []       # 步数移动平均
        self.avg_losses = []      # 损失移动平均
        self.td_errors = []       # TD误差统计
    
    def update(self, score, reward, steps, loss, epsilon, td_errors=None):
        """更新统计数据"""
        # 记录当前局数据
        self.scores.append(score)
        self.rewards.append(reward)
        self.steps.append(steps)
        self.losses.append(loss)
        self.epsilons.append(epsilon)
        
        # 记录TD误差统计
        if td_errors is not None and len(td_errors) > 0:
            self.td_errors.append(np.mean(td_errors))
        
        # 计算移动平均 (n_mean局)
        window = min(n_mean, len(self.scores))
        if len(self.scores) >= window:
            self.avg_scores.append(np.mean(self.scores[-window:]))
            self.avg_rewards.append(np.mean(self.rewards[-window:]))
            self.avg_steps.append(np.mean(self.steps[-window:]))
            self.avg_losses.append(np.mean(self.losses[-window:]))
    
    def get_stats(self):
        """获取格式化统计数据"""
        stats = {
            'scores': self.scores,
            'rewards': self.rewards,
            'steps': self.steps,
            'losses': self.losses,
            'epsilons': self.epsilons,
            'avg_scores': self.avg_scores,
            'avg_rewards': self.avg_rewards,
            'avg_steps': self.avg_steps,
            'avg_losses': self.avg_losses
        }
        
        if self.td_errors:
            stats['td_errors'] = self.td_errors
            
        return stats


class TrainingPlotter:
    def __init__(self):
        # 创建4个子图
        plt.ion()  # 开启交互模式
        self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('蛇游戏AI训练指标')
        
        # 初始化空图表
        self.score_line, = self.axs[0, 0].plot([], [], 'b-', label='分数')
        self.avg_score_line, = self.axs[0, 0].plot([], [], 'r-', label=f'{n_mean}局平均')
        self.axs[0, 0].set_title('每局最终长度')
        self.axs[0, 0].set_xlabel('局数')
        self.axs[0, 0].set_ylabel('长度')
        self.axs[0, 0].legend()
        self.axs[0, 0].grid(True, alpha=0.3)
        
        self.reward_line, = self.axs[0, 1].plot([], [], 'b-')
        self.axs[0, 1].set_title('每局总奖励')
        self.axs[0, 1].set_xlabel('局数')
        self.axs[0, 1].set_ylabel('奖励')
        self.axs[0, 1].grid(True, alpha=0.3)
        
        self.steps_line, = self.axs[1, 0].plot([], [], 'b-')
        self.axs[1, 0].set_title('每局步数')
        self.axs[1, 0].set_xlabel('局数')
        self.axs[1, 0].set_ylabel('步数')
        self.axs[1, 0].grid(True, alpha=0.3)
        
        self.loss_line, = self.axs[1, 1].plot([], [], 'g-', label='平均损失')
        self.epsilon_line, = self.axs[1, 1].plot([], [], 'b-', label='探索率')
        self.axs[1, 1].set_title('训练指标')
        self.axs[1, 1].set_xlabel('局数')
        self.axs[1, 1].set_ylabel('值')
        self.axs[1, 1].legend()
        self.axs[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    def update(self, stats):
        """更新图表数据"""
        episodes = list(range(len(stats['scores'])))
        
        # 更新分数图表
        self.score_line.set_data(episodes, stats['scores'])
        self.avg_score_line.set_data(episodes, stats['avg_scores'])
        self.axs[0, 0].relim()
        self.axs[0, 0].autoscale_view()
        
        # 更新奖励图表
        self.reward_line.set_data(episodes, stats['rewards'])
        self.axs[0, 1].relim()
        self.axs[0, 1].autoscale_view()
        
        # 更新步数图表
        self.steps_line.set_data(episodes, stats['steps'])
        self.axs[1, 0].relim()
        self.axs[1, 0].autoscale_view()
        
        # 更新损失和探索率图表
        self.loss_line.set_data(episodes, stats['losses'])
        self.epsilon_line.set_data(episodes, stats['epsilons'])
        self.axs[1, 1].relim()
        self.axs[1, 1].autoscale_view()
        
        # 刷新图表
        self.fig.canvas.flush_events()
        plt.pause(0.01)
        
    def _moving_average(self, data, window_size=n_mean):
        """计算移动平均值"""
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    def close(self):
        """关闭图表窗口"""
        plt.close(self.fig)