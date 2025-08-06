from enum import Enum

import numpy as np
import pygame

from config import Config
from help import distance_nb, is_collision_nb, random_position_nb, place_food_nb, step_logic_nb
from direction import Direction
from help import get_head_pos_nb, get_food_pos_nb, get_relative_distance_nb, get_manhattan_distance_nb,\
    get_direction_onehot_nb, get_eight_direction_dangers_nb, get_boundary_distances_nb, get_snake_length_nb, \
        get_free_space_ratio_nb, get_local_grid_view_nb, get_local_grid_view_nb, get_action_history_onehot_nb

class Game:
    RIGHT = (1, 0)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    UP = (0, -1)

class SnakeGame:
    def __init__(self, width=Config.WIDTH, height=Config.HEIGHT, visualize=True):
        self.width = width
        self.height = height
        self.visualize = visualize
        self.action_history = [0] * 5  # 初始化动作历史为5个默认动作(0)
        
        if visualize:
            pygame.init()
            self.display = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption('Snake AI Training')
            self.clock = pygame.time.Clock()
            # 使用系统中的SimHei字体支持中文显示
            self.font = pygame.font.SysFont(['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC'], 25)
        else:
            self.display = None
            self.clock = None
            self.font = None
            
        self.reset()
    
    def reset(self):
        """重置游戏状态"""
        self.action_history = [0] * 5  # 重置为5个默认动作(0)
        self.direction = Direction.RIGHT
        self.head = self._random_position()
        
        # 初始化蛇身
        self.snake = [
            self.head,
            (self.head[0] - Config.BLOCK_SIZE, self.head[1]),
            (self.head[0] - Config.BLOCK_SIZE * 2, self.head[1])
        ]
        
        self.score = 0
        self.food = self._place_food()
        self.steps_since_food = 0
        self.game_over = False
        self.prev_distance = self._calc_distance(self.head, self.food)
        
        return self.get_state()
    
    def _random_position(self):
        """生成随机位置"""
        return random_position_nb(self.width, self.height, Config.BLOCK_SIZE)

    def _place_food(self):
        """放置食物，避开蛇身"""
        return place_food_nb(self.snake, self.width, self.height, Config.BLOCK_SIZE)

    def _calc_distance(self, pos1, pos2):
        """计算两点之间的曼哈顿距离"""
        return distance_nb(pos1, pos2)
    
    def _is_collision(self, pos=None):
        """检测碰撞"""
        if pos is None:
            pos = self.head
        return is_collision_nb(self.snake, self.width, self.height, pos)
    
    def get_state(self):
        grid_area = Config.GRID_WIDTH * Config.GRID_HEIGHT
        grid_size = Config.BLOCK_SIZE
        
        # 使用numba加速函数提取状态特征
        danger = np.array([
            self._is_collision((self.head[0] + grid_size * self.direction.value[0], 
                            self.head[1] + grid_size * self.direction.value[1])),
            self._is_collision((self.head[0] + grid_size * -self.direction.value[1], 
                            self.head[1] + grid_size * self.direction.value[0])),
            self._is_collision((self.head[0] + grid_size * self.direction.value[1], 
                            self.head[1] + grid_size * -self.direction.value[0]))
        ], dtype=np.float32)
        
        # 合并所有状态特征
        return np.concatenate([
            danger,
            get_head_pos_nb(self.head, self.width, self.height),
            get_food_pos_nb(self.food, self.width, self.height),
            get_relative_distance_nb(self.head, self.food, self.width, self.height),
            get_manhattan_distance_nb(self.head, self.food, self.width, self.height),
            get_direction_onehot_nb(self.direction.value),
            get_eight_direction_dangers_nb(self.head, self.snake, self.width, self.height, grid_size),
            get_boundary_distances_nb(self.head, self.width, self.height),
            get_snake_length_nb(self.snake, grid_area),
            get_free_space_ratio_nb(self.snake, grid_area),
            get_local_grid_view_nb(self.head, self.snake, self.food, self.width, self.height, grid_size),
            get_action_history_onehot_nb(self.action_history)
        ])
    
    def step(self, action):
        """
        执行游戏步骤
        
        参数:
            action: 整数表示的动作 [0: 直行, 1: 右转, 2: 左转]
        
        返回:
            state: 新状态
            reward: 获得的奖励
            done: 游戏是否结束
            score: 当前分数
        """
        
        # 0. 记录当前动作到历史
        self.action_history.append(action)
        # 只保留最近的5个动作
        if len(self.action_history) > 5:
            self.action_history.pop(0)

        # 1. 处理事件
        if self.visualize:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None, 0, True, self.score
        
        # 2. 调用numba优化后的核心逻辑
        (self.snake, self.head, new_dir_vec, self.food, 
        self.steps_since_food, self.score, reward, done, 
        self.prev_distance) = step_logic_nb(
            self.snake,
            self.head,
            self.direction.value,
            self.food,
            self.steps_since_food,
            self.score,
            self.prev_distance,
            self.width,
            self.height,
            Config.BLOCK_SIZE,
            action,
            Config.MAX_STEPS_WITHOUT_FOOD,
            Config.FOOD_REWARD,
            Config.COLLISION_PENALTY,
            Config.PROGRESS_REWARD,
            Config.STEP_PENALTY
        )
        
        # 3. 更新方向枚举
        for dir_enum in Direction:
            if dir_enum.value == new_dir_vec:
                self.direction = dir_enum
                break
        
        # 5. 获取新状态
        next_state = self.get_state()
        
        # 6. 渲染游戏
        self.render()
        
        return next_state, reward, done, self.score

    def render(self):
        """渲染游戏画面"""
        if not self.visualize:
            return
            
        self.display.fill((0, 0, 0))  # 黑色背景
        
        # 绘制网格线
        for x in range(0, self.width, Config.BLOCK_SIZE):
            pygame.draw.line(self.display, (40, 40, 40), (x, 0), (x, self.height))
        for y in range(0, self.height, Config.BLOCK_SIZE):
            pygame.draw.line(self.display, (40, 40, 40), (0, y), (self.width, y))
        
        # 绘制蛇
        for i, pos in enumerate(self.snake):
            color = (0, 200, 0) if i == 0 else (0, 100, 255)  # 头部绿色，身体蓝色
            pygame.draw.rect(self.display, color, 
                           pygame.Rect(pos[0], pos[1], Config.BLOCK_SIZE, Config.BLOCK_SIZE))
            pygame.draw.rect(self.display, (0, 0, 0), 
                           pygame.Rect(pos[0] + 4, pos[1] + 4, Config.BLOCK_SIZE - 8, Config.BLOCK_SIZE - 8))
        
        # 绘制食物
        pygame.draw.rect(self.display, (255, 0, 0), 
                       pygame.Rect(self.food[0], self.food[1], 
                                  Config.BLOCK_SIZE, Config.BLOCK_SIZE))
        
        # 绘制分数
        score_text = self.font.render(f"分数: {self.score}", True, (255, 255, 255))
        steps_text = self.font.render(f"步数: {self.steps_since_food}", True, (255, 255, 255))
        self.display.blit(score_text, [5, 5])
        self.display.blit(steps_text, [5, 35])
        
        pygame.display.flip()
        self.clock.tick(Config.SPEED)