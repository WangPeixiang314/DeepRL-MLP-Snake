from enum import Enum

import numpy as np
import pygame

from config import Config
from help import distance_nb, random_position_nb, place_food_nb, step_logic_nb
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
        
        # 初始化循环数组 - 用于存储蛇身位置
        self.max_snake_length = Config.GRID_WIDTH * Config.GRID_HEIGHT  # 蛇最大可能长度
        self.snake_body = [(0, 0)] * self.max_snake_length  # 固定大小的循环数组
        self.snake_start = 0  # 蛇头在循环数组中的索引
        self.snake_length = 3  # 当前蛇长度
        
        # 初始化蛇身位置
        self.snake_body[0] = self.head
        self.snake_body[1] = (self.head[0] - Config.BLOCK_SIZE, self.head[1])
        self.snake_body[2] = (self.head[0] - Config.BLOCK_SIZE * 2, self.head[1])
        
        # 初始化位图 - 用于O(1)碰撞检测
        self.grid_width = self.width // Config.BLOCK_SIZE
        self.grid_height = self.height // Config.BLOCK_SIZE
        self.occupied = np.zeros((self.grid_width, self.grid_height), dtype=np.bool_)
        
        # 标记蛇身占据的位置
        for i in range(self.snake_length):
            x, y = self.snake_body[i]
            grid_x, grid_y = x // Config.BLOCK_SIZE, y // Config.BLOCK_SIZE
            self.occupied[grid_x, grid_y] = True
        
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
        """放置食物，避开蛇身 - 使用位图实现O(1)时间复杂度"""
        grid_width = self.width // Config.BLOCK_SIZE
        grid_height = self.height // Config.BLOCK_SIZE
        max_attempts = 1000
        
        for _ in range(max_attempts):
            food_x = np.random.randint(0, grid_width) * Config.BLOCK_SIZE
            food_y = np.random.randint(0, grid_height) * Config.BLOCK_SIZE
            food_grid_x = food_x // Config.BLOCK_SIZE
            food_grid_y = food_y // Config.BLOCK_SIZE
            
            # 使用位图检查食物是否与蛇身重叠 - O(1)时间复杂度
            if not self.occupied[food_grid_x, food_grid_y]:
                return (food_x, food_y)
        
        # 如果找不到合适位置，返回默认位置
        return (Config.BLOCK_SIZE, Config.BLOCK_SIZE)

    def _calc_distance(self, pos1, pos2):
        """计算两点之间的曼哈顿距离"""
        return distance_nb(pos1, pos2)
    
    def _is_collision(self, pos=None):
        """检测碰撞 - O(1)时间复杂度"""
        if pos is None:
            pos = self.head
        
        # 边界检查
        if (pos[0] >= self.width or pos[0] < 0 or 
            pos[1] >= self.height or pos[1] < 0):
            return True
        
        # 使用位图进行O(1)碰撞检测
        grid_x, grid_y = pos[0] // Config.BLOCK_SIZE, pos[1] // Config.BLOCK_SIZE
        return self.occupied[grid_x, grid_y]
    
    def _get_snake_list(self):
        """获取蛇身列表 - 用于兼容原有代码"""
        snake = []
        for i in range(self.snake_length):
            idx = (self.snake_start + i) % self.max_snake_length
            snake.append(self.snake_body[idx])
        return snake
    
    def get_state(self):
        grid_area = Config.GRID_WIDTH * Config.GRID_HEIGHT
        grid_size = Config.BLOCK_SIZE
        snake = self._get_snake_list()  # 获取蛇身列表用于兼容性
        
        # 使用numba加速函数提取状态特征
        danger = np.array([
            self._is_collision((self.head[0] + grid_size * self.direction.value[0], 
                            self.head[1] + grid_size * self.direction.value[1])),
            self._is_collision((self.head[0] + grid_size * -self.direction.value[1], 
                            self.head[1] + grid_size * self.direction.value[0])),
            self._is_collision((self.head[0] + grid_size * self.direction.value[1], 
                            self.head[1] + grid_size * -self.direction.value[0]))
        ], dtype=np.float32)
        
        # 使用位图优化八方向危险检测
        eight_direction_dangers = get_eight_direction_dangers_nb(
            self.head, self.occupied, self.width, self.height, grid_size
        )
        
        # 合并所有状态特征
        return np.concatenate([
            danger,
            get_head_pos_nb(self.head, self.width, self.height),
            get_food_pos_nb(self.food, self.width, self.height),
            get_relative_distance_nb(self.head, self.food, self.width, self.height),
            get_manhattan_distance_nb(self.head, self.food, self.width, self.height),
            get_direction_onehot_nb(self.direction.value),
            eight_direction_dangers,
            get_boundary_distances_nb(self.head, self.width, self.height),
            get_snake_length_nb(snake, grid_area),
            get_free_space_ratio_nb(snake, grid_area),
            get_local_grid_view_nb(self.head, snake, self.food, self.width, self.height, grid_size),
            get_action_history_onehot_nb(self.action_history)
        ])
    
    def step(self, action):
        """
        执行游戏步骤 - 使用O(1)时间复杂度的实现
        
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
        
        # 2. 更新方向
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # RIGHT, DOWN, LEFT, UP
        current_vec = self.direction.value
        current_idx = directions.index(current_vec) if current_vec in directions else 0
        
        if action == 0:  # 直行，方向不变
            new_dir_vec = directions[current_idx]
        elif action == 1:  # 右转
            new_dir_vec = directions[(current_idx + 1) % 4]
        elif action == 2:  # 左转
            new_dir_vec = directions[(current_idx - 1) % 4]
        else:
            new_dir_vec = directions[current_idx]
        
        # 更新方向枚举
        for dir_enum in Direction:
            if dir_enum.value == new_dir_vec:
                self.direction = dir_enum
                break
        
        # 3. 计算新蛇头位置
        dx, dy = new_dir_vec
        new_head = (self.head[0] + dx * Config.BLOCK_SIZE, self.head[1] + dy * Config.BLOCK_SIZE)
        
        # 4. 检查碰撞 - O(1)时间复杂度
        done = False
        reward = Config.STEP_PENALTY
        
        # 边界碰撞检查
        if (new_head[0] >= self.width or new_head[0] < 0 or 
            new_head[1] >= self.height or new_head[1] < 0):
            done = True
            reward = Config.COLLISION_PENALTY
        
        # 自身碰撞检查 - 使用位图
        if not done:
            grid_x, grid_y = new_head[0] // Config.BLOCK_SIZE, new_head[1] // Config.BLOCK_SIZE
            if self.occupied[grid_x, grid_y]:
                done = True
                reward = Config.COLLISION_PENALTY
        
        # 检查是否超过最大无食物步数
        if not done and self.steps_since_food >= Config.MAX_STEPS_WITHOUT_FOOD:
            done = True
            reward = Config.COLLISION_PENALTY
        
        if done:
            return self.get_state(), reward, done, self.score
        
        # 5. 更新蛇身 - 使用循环数组
        # 计算新的蛇头索引（向前移动一位）
        new_snake_start = (self.snake_start - 1) % self.max_snake_length
        self.snake_body[new_snake_start] = new_head
        
        # 标记新蛇头位置
        self.occupied[grid_x, grid_y] = True
        
        # 检查是否吃到食物
        if new_head == self.food:
            # 吃到食物，蛇身增长
            self.snake_length += 1
            self.score += 1
            reward = Config.FOOD_REWARD
            self.steps_since_food = 0
            self.food = self._place_food()
            self.prev_distance = self._calc_distance(new_head, self.food)
        else:
            # 没吃到食物，移除蛇尾
            tail_idx = (self.snake_start + self.snake_length - 1) % self.max_snake_length
            tail_x, tail_y = self.snake_body[tail_idx]
            tail_grid_x, tail_grid_y = tail_x // Config.BLOCK_SIZE, tail_y // Config.BLOCK_SIZE
            self.occupied[tail_grid_x, tail_grid_y] = False
            
            # 计算距离奖励
            distance = self._calc_distance(new_head, self.food)
            reward += Config.PROGRESS_REWARD * (self.prev_distance - distance)
            self.prev_distance = distance
            
            self.steps_since_food += 1
        
        # 更新蛇头位置和起始索引
        self.head = new_head
        self.snake_start = new_snake_start
        
        # 6. 获取新状态
        next_state = self.get_state()
        
        # 7. 渲染游戏
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
        
        # 绘制蛇 - 使用循环数组
        snake = self._get_snake_list()
        for i, pos in enumerate(snake):
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