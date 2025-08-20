import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
import sys

# 动态matplotlib显示支持
plt.ion()  # 启用交互模式

n_mean = 500

# 配置中文显示
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False

# 设置图表样式
style.use('ggplot')

# ANSI颜色代码
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def disable():
        Colors.HEADER = ''
        Colors.OKBLUE = ''
        Colors.OKCYAN = ''
        Colors.OKGREEN = ''
        Colors.WARNING = ''
        Colors.FAIL = ''
        Colors.ENDC = ''
        Colors.BOLD = ''
        Colors.UNDERLINE = ''

# Windows系统可能需要特殊处理
if sys.platform.startswith('win'):
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except:
        Colors.disable()

class TrainingStats:
    def __init__(self):
        # 训练数据存储
        self.scores = []          # 每局分数
        self.rewards = []         # 每局总奖励
        self.steps = []           # 每局步数
        self.losses = []          # 每局平均损失
        self.epsilons = []        # 每局探索率
        self.final_lengths = []   # 每局最终蛇长度
        
        # 移动平均数据
        self.avg_scores = []      # 分数移动平均
        self.avg_rewards = []     # 奖励移动平均
        self.avg_steps = []       # 步数移动平均
        self.avg_losses = []      # 损失移动平均
        self.avg_final_lengths = []  # 最终长度移动平均
        self.td_errors = []       # TD误差统计
    
    def update(self, score, reward, steps, loss, epsilon, final_length, td_errors=None):
        """更新统计数据"""
        # 记录当前局数据
        self.scores.append(score)
        self.rewards.append(reward)
        self.steps.append(steps)
        self.losses.append(loss)
        self.epsilons.append(epsilon)
        self.final_lengths.append(final_length)
        
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
            self.avg_final_lengths.append(np.mean(self.final_lengths[-window:]))
    
    def get_stats(self):
        """获取格式化统计数据"""
        stats = {
            'scores': self.scores,
            'rewards': self.rewards,
            'steps': self.steps,
            'losses': self.losses,
            'epsilons': self.epsilons,
            'final_lengths': self.final_lengths,
            'avg_scores': self.avg_scores,
            'avg_rewards': self.avg_rewards,
            'avg_steps': self.avg_steps,
            'avg_losses': self.avg_losses,
            'avg_final_lengths': self.avg_final_lengths
        }
        
        if self.td_errors:
            stats['td_errors'] = self.td_errors
            
        return stats


class TrainingPlotter:
    def __init__(self):
        # 创建4个子图
        try:
            matplotlib.use('TkAgg')  # 使用TkAgg后端，更稳定
            plt.ion()  # 开启交互模式
            self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 8))
            self.fig.suptitle('蛇游戏AI训练指标')
            self.initialized = True
        except Exception as e:
            print(f"警告: 无法初始化实时绘图: {e}")
            self.initialized = False
            self.fig = None
            self.axs = None
        
        # 初始化空图表 - 第一个子图显示最终长度而非分数
        self.length_line, = self.axs[0, 0].plot([], [], 'b-', label='最终长度')
        self.avg_length_line, = self.axs[0, 0].plot([], [], 'r-', label=f'最近{n_mean}局平均')
        self.axs[0, 0].set_title('每局游戏最终长度')
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
        self.axs[1, 1].set_yscale('log')  # 设置y轴为对数坐标
        self.axs[1, 1].legend()
        self.axs[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    def update(self, stats):
        """更新图表数据"""
        episodes = list(range(len(stats['scores'])))
        
        # 更新最终长度图表（第一个子图）
        self.length_line.set_data(episodes, stats['final_lengths'])
        self.avg_length_line.set_data(episodes, stats['avg_final_lengths'])
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

class ConsoleLogger:
    def __init__(self):
        self.episode_count = 0
        
    def print_training_header(self):
        """打印训练标题"""
        print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}Dueling DDQN 训练系统{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
        
    def print_episode_status(self, episode, score, avg_score, loss, epsilon, steps, lr):
        """每5局打印一次训练状态"""
        if episode % 5 == 0:
            print(f"\n{Colors.OKCYAN}[回合 {episode:4d}] {Colors.ENDC}")
            print(f"{Colors.OKGREEN}  当前得分: {score:3d} {Colors.ENDC}")
            print(f"{Colors.OKBLUE}  平均得分: {avg_score:6.2f} {Colors.ENDC}")
            print(f"{Colors.WARNING}  学习率:   {lr:.6f} {Colors.ENDC}")
            print(f"{Colors.FAIL}  探索率:   {epsilon:.4f} {Colors.ENDC}")
            print(f"{Colors.OKCYAN}  步数:     {steps:4d} {Colors.ENDC}")
            print(f"{Colors.HEADER}{'-'*40}{Colors.ENDC}")
    
    def print_best_score(self, episode, score):
        """打印最佳分数"""
        print(f"{Colors.OKGREEN}{Colors.BOLD}新的最佳Dueling DQN模型已保存: 回合 {episode} 分数 {score}{Colors.ENDC}")
    
    def print_model_saved(self, filepath):
        """打印模型保存信息"""
        print(f"{Colors.OKBLUE}模型已保存: {filepath}{Colors.ENDC}")
    
    def print_training_complete(self, total_episodes, max_score, avg_score, total_time):
        """打印训练完成信息"""
        print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}训练完成!{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
        print(f"{Colors.OKGREEN}总训练时间: {total_time:.1f}秒{Colors.ENDC}")
        print(f"{Colors.OKGREEN}总训练局数: {total_episodes}{Colors.ENDC}")
        print(f"{Colors.OKGREEN}最高分数: {max_score}{Colors.ENDC}")
        print(f"{Colors.OKGREEN}最终平均分数: {avg_score:.2f}{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")