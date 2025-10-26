import os
import torch
import os

import torch

class Config:
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 网格参数
    GRID_WIDTH = 12   # 地图网格宽度（单位：格）
    GRID_HEIGHT = 12  # 地图网格高度（单位：格）
    BLOCK_SIZE = 40   # 每个网格块的像素大小
    WIDTH = GRID_WIDTH * BLOCK_SIZE
    HEIGHT = GRID_HEIGHT * BLOCK_SIZE
    
    # 游戏参数
    SPEED = 600
    
    # 模型参数
    HIDDEN_LAYERS = [128, 64]  # 隐藏层配置
    OUTPUT_DIM = 3   # [直行, 右转, 左转]
    
    # 训练参数
    BATCH_SIZE = 128
    MEMORY_CAPACITY = 200_000
    LEARNING_RATE = 3.481792268387811e-05
    GAMMA = 0.99
    TARGET_UPDATE = 100  # 更新目标网络的间隔
    MAX_STEPS_WITHOUT_FOOD = 500  # 最大无食物步数
    
    # 优先级采样参数
    PRIO_ALPHA = 0.6000000000000001  # 控制采样的随机性程度 (0~1)
    PRIO_BETA_START = 0.4  # 重要性采样权重的初始值
    PRIO_BETA_FRAMES = 60_000  # beta增加到1所需的帧数
    PRIO_EPS = 1e-6  # 防止优先级为0的小常数
    
    # 探索参数
    EPS_START = 1.0
    EPS_END = 0.02
    EPS_DECAY = 7000  # ε衰减步数 (增加以延长探索)
    
    # 奖励参数
    FOOD_REWARD = 16.0
    COLLISION_PENALTY = -10.0
    STEP_PENALTY = 0.01
    PROGRESS_REWARD = 0.1  # 向食物靠近的奖励
    
    # 文件路径
    MODEL_DIR = './models'
    MODEL_FILE = 'snake_dqn.pth'
    LOG_FILE = 'training_log.csv'
    
    # 绘图参数
    PLOT_INTERVAL = 2  # 每N局游戏更新绘图 
    
    @staticmethod
    def init():
        """创建必要的目录"""
        if not os.path.exists(Config.MODEL_DIR):
            os.makedirs(Config.MODEL_DIR)


# 初始化配置
Config.init()