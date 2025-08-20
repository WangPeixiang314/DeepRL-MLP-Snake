import os
import torch

class Config:
    # ==================== 系统配置 ====================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==================== 游戏配置 ====================
    GRID_WIDTH = 12   # 地图网格宽度（单位：格）
    GRID_HEIGHT = 12  # 地图网格高度（单位：格）
    BLOCK_SIZE = 40   # 每个网格块的像素大小
    WIDTH = GRID_WIDTH * BLOCK_SIZE
    HEIGHT = GRID_HEIGHT * BLOCK_SIZE
    SPEED = 600  # 游戏速度（毫秒）
    
    # ==================== Dueling DDQN 模型配置 ====================
    # 网络架构
    HIDDEN_LAYERS = [512, 256, 128, 64, 32]  # Dueling网络隐藏层配置
    OUTPUT_DIM = 3   # [直行, 右转, 左转]
    
    # Dueling网络特有参数
    FEATURE_LAYER_RATIO = 0.67  # 特征层占总隐藏层的比例 (0.6-0.7推荐)
    VALUE_ADVANTAGE_RATIO = 0.5  # Value和Advantage流的中间层比例
    DROPOUT_RATE = 0.1  # Dropout比例
    
    # 权重初始化方法
    WEIGHT_INIT = 'xavier_uniform'  # 'xavier_uniform', 'xavier_normal', 'kaiming_uniform'
    
    # ==================== 训练优化器配置 ====================
    BATCH_SIZE = 128
    MEMORY_CAPACITY = 200_000
    LEARNING_RATE = 3.481792268387811e-05
    GAMMA = 0.995
    
    # 优化器配置
    OPTIMIZER = 'AdamW'  # 'Adam', 'AdamW', 'RAdam'
    WEIGHT_DECAY = 1e-5  # L2正则化
    GRADIENT_CLIP = 1.0  # 梯度裁剪阈值
    
    # 学习率调度
    LR_SCHEDULER = 'CosineAnnealing'  # 'CosineAnnealing', 'StepLR', 'ExponentialLR'
    COSINE_T_MAX = 1000  # Cosine退火周期
    
    # 目标网络更新策略
    TARGET_UPDATE = 100  # 硬更新间隔（局数）
    TARGET_SOFT_UPDATE = True  # 是否使用软更新
    TARGET_TAU = 0.005  # 软更新系数
    
    # ==================== 经验回放配置 ====================
    # 优先级采样参数
    PRIO_ALPHA = 0.6  # 控制采样的随机性程度 (0~1)
    PRIO_BETA_START = 0.4  # 重要性采样权重的初始值
    PRIO_BETA_FRAMES = 60_000  # beta增加到1所需的帧数
    PRIO_EPS = 1e-6  # 防止优先级为0的小常数
    
    # ==================== 探索策略配置 ====================
    EPS_START = 1.0
    EPS_END = 0.02
    EPS_DECAY = 20000  # ε衰减步数 (增加以延长探索)
    
    # 探索策略类型
    EXPLORATION = 'epsilon_greedy'  # 'epsilon_greedy', 'noisy_networks', 'ucb'
    
    # ==================== 奖励塑形配置 ====================
    REWARD_CONFIG = {
        'food_reward': 16.0,
        'collision_penalty': -10.0,
        'step_penalty': -0.01,  # 改为负值鼓励快速吃食物
        'progress_reward': 0.1,  # 向食物靠近的奖励
        'survival_bonus': 0.001,  # 存活奖励
        'danger_penalty': -0.1,  # 危险行为惩罚
    }
    
    # 快捷访问
    FOOD_REWARD = REWARD_CONFIG['food_reward']
    COLLISION_PENALTY = REWARD_CONFIG['collision_penalty']
    STEP_PENALTY = REWARD_CONFIG['step_penalty']
    PROGRESS_REWARD = REWARD_CONFIG['progress_reward']
    
    # ==================== 训练过程配置 ====================
    MAX_EPISODES = 2000  # 最大训练局数
    MAX_STEPS_WITHOUT_FOOD = 500  # 最大无食物步数
    SAVE_INTERVAL = 100  # 模型保存间隔
    LOG_INTERVAL = 50  # 日志打印间隔
    PLOT_INTERVAL = 100  # 绘图间隔
    
    # ==================== 文件路径配置 ====================
    MODEL_DIR = './models'
    MODEL_FILE = 'snake_dueling_dqn.pth'
    BEST_MODEL_FILE = 'snake_dueling_dqn_best.pth'
    LOG_FILE = 'dueling_training_log.csv'
    PLOT_FILE = 'dueling_training_progress.png'
    
    # ==================== 状态特征配置 ====================
    STATE_CONFIG = {
        'eight_direction_steps': 5,  # 八方向危险检查的格子数量
        'local_grid_radius': 5,      # 局部网络半径
        'action_history_length': 7,  # 动作历史长度
    }
    
    # 快捷访问
    EIGHT_DIRECTION_STEPS = STATE_CONFIG['eight_direction_steps']
    LOCAL_GRID_RADIUS = STATE_CONFIG['local_grid_radius']
    ACTION_HISTORY_LENGTH = STATE_CONFIG['action_history_length']
    
    # ==================== 高级功能开关 ====================
    ENABLE_PER = True  # 优先级经验回放
    ENABLE_DUELING = True  # Dueling网络架构
    ENABLE_DOUBLE = True  # Double DQN
    ENABLE_NOISY = False  # Noisy Networks (实验功能)
    ENABLE_DUELING_ANALYSIS = False  # Dueling网络分析模式
    
    @classmethod
    def get_model_filename(cls, episode, score, is_best=False):
        """生成模型文件名"""
        if is_best:
            return cls.BEST_MODEL_FILE
        return f"snake_dueling_dqn_ep{episode}_sc{score}.pth"
    
    @classmethod
    def print_config(cls):
        """打印当前配置"""
        print("Dueling DDQN 配置信息")
        print("=" * 50)
        print(f"设备: {cls.device}")
        print(f"网络架构: {cls.HIDDEN_LAYERS}")
        print(f"学习率: {cls.LEARNING_RATE}")
        print(f"批次大小: {cls.BATCH_SIZE}")
        print(f"经验回放: {'启用' if cls.ENABLE_PER else '禁用'}")
        print(f"Dueling网络: {'启用' if cls.ENABLE_DUELING else '禁用'}")
        print(f"Double DQN: {'启用' if cls.ENABLE_DOUBLE else '禁用'}")
        print("=" * 50)
    
    @staticmethod
    def init():
        """创建必要的目录"""
        if not os.path.exists(Config.MODEL_DIR):
            os.makedirs(Config.MODEL_DIR)
            print(f"📁 创建模型目录: {Config.MODEL_DIR}")
        
        # 打印配置信息
        Config.print_config()

# 初始化配置
Config.init()