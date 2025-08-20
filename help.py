import numba
import numpy as np
from pygame import fastevent

njit_decorator = numba.njit(fastmath=True, cache=True, inline='always', looplift=True, error_model='numpy')


# 辅助函数
@njit_decorator
def distance_nb(pos1, pos2):
    """计算两点之间的曼哈顿距离"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

@njit_decorator
def is_collision_nb(snake, width, height, pos=None):
    """检测碰撞"""
    if pos is None:
        pos = snake[0]
    
    # 边界检查
    if (pos[0] >= width or pos[0] < 0 or 
        pos[1] >= height or pos[1] < 0):
        return True
    
    # 自身碰撞检查
    for segment in snake[1:]:
        if segment[0] == pos[0] and segment[1] == pos[1]:
            return True
    
    return False

@njit_decorator
def random_position_nb(width, height, BLOCK_SIZE):
    """生成随机位置"""
    return (
        np.random.randint(0, ((width - BLOCK_SIZE) // BLOCK_SIZE) + 1) * BLOCK_SIZE,
        np.random.randint(0, ((height - BLOCK_SIZE) // BLOCK_SIZE) + 1) * BLOCK_SIZE
    )

@njit_decorator
def place_food_nb(snake, width, height, BLOCK_SIZE):
    """放置食物，避开蛇身"""
    while True:
        food = random_position_nb(width, height, BLOCK_SIZE)
        if food not in snake:
            return food

@njit_decorator
def update_direction_nb(current_dir, action):
    """根据动作更新方向"""
    # 获取当前方向的索引
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # RIGHT, DOWN, LEFT, UP
    current_vec = (current_dir[0], current_dir[1])
    
    # 找到当前方向在列表中的位置
    current_idx = -1
    for i, vec in enumerate(directions):
        if vec == current_vec:
            current_idx = i
            break
    
    if current_idx == -1:
        # 如果没找到，保持原方向
        return current_dir
    
    # 更新方向
    if action == 0:  # 直行，方向不变
        return directions[current_idx]
    elif action == 1:  # 右转
        new_idx = (current_idx + 1) % 4
    elif action == 2:  # 左转
        new_idx = (current_idx - 1) % 4
    else:
        return directions[current_idx]
    
    return directions[new_idx]

@njit_decorator
def step_logic_nb(
    snake, 
    head, 
    direction_vec, 
    food, 
    steps_since_food, 
    score, 
    prev_distance,
    width, 
    height, 
    BLOCK_SIZE,
    action,
    MAX_STEPS_WITHOUT_FOOD,
    FOOD_REWARD,
    COLLISION_PENALTY,
    PROGRESS_REWARD,
    STEP_PENALTY
):
    """
    游戏逻辑核心部分 (numba加速)
    
    返回:
        new_snake, new_head, new_direction_vec, new_food, 
        new_steps_since_food, new_score, reward, done, new_prev_distance
    """
    # 1. 更新方向
    new_direction_vec = update_direction_nb(direction_vec, action)
    
    # 2. 移动蛇头
    dx, dy = new_direction_vec
    new_head = (head[0] + dx * BLOCK_SIZE, head[1] + dy * BLOCK_SIZE)
    new_snake = snake.copy()
    new_snake.insert(0, new_head)
    
    # 3. 检查游戏结束
    done = False
    if is_collision_nb(new_snake, width, height, new_head) or steps_since_food >= MAX_STEPS_WITHOUT_FOOD:
        reward = COLLISION_PENALTY
        done = True
        return new_snake, new_head, new_direction_vec, food, steps_since_food, score, reward, done, prev_distance
    
    # 4. 检查吃到食物
    elif new_head == food:
        new_score = score + 1
        new_food = place_food_nb(new_snake, width, height, BLOCK_SIZE)
        new_steps_since_food = 0
        reward = FOOD_REWARD
        new_prev_distance = distance_nb(new_head, new_food)
        done = False
    else:
        new_snake.pop()
        new_steps_since_food = steps_since_food + 1
        new_score = score
        new_food = food
        
        # 计算距离奖励
        distance = distance_nb(new_head, food)
        reward = PROGRESS_REWARD * (prev_distance - distance)
        new_prev_distance = distance
        
        # 添加步数惩罚
        reward += STEP_PENALTY
        done = False
    
    return new_snake, new_head, new_direction_vec, new_food, new_steps_since_food, new_score, reward, done, new_prev_distance


@njit_decorator
def propagate_nb(tree, idx, change):
    parent = (idx - 1) // 2
    while parent >= 0:
        tree[parent] += change
        if parent == 0:
            break
        parent = (parent - 1) // 2

@njit_decorator
def retrieve_nb(tree, capacity, s):
    idx = 0
    while idx < capacity - 1:
        left = 2 * idx + 1
        if s <= tree[left]:
            idx = left
        else:
            s -= tree[left]
            idx = left + 1
    return idx

# 顺序版本（安全）
@njit_decorator
def batch_retrieve_nb(tree, capacity, s_values):
    n = s_values.shape[0]
    indices = np.empty(n, dtype=np.int64)
    for i in range(n):
        s = s_values[i]
        idx = 0
        while idx < capacity - 1:
            left = 2 * idx + 1
            if s <= tree[left]:
                idx = left
            else:
                s -= tree[left]
                idx = left + 1
        indices[i] = idx
    return indices

# 并行版本（高性能）
@numba.njit(fastmath=True, cache=True, inline='always', looplift=True, error_model='numpy', parallel=True)
def batch_retrieve_par_nb(tree, capacity, s_values):
    n = s_values.shape[0]
    indices = np.empty(n, dtype=np.int64)
    for i in numba.prange(n):  # 并行循环
        s = s_values[i]
        idx = 0
        while idx < capacity - 1:
            left = 2 * idx + 1
            if s <= tree[left]:
                idx = left
            else:
                s -= tree[left]
                idx = left + 1
        indices[i] = idx
    return indices

@njit_decorator
def safe_action_nb(q_values, danger_signals, collision_penalty, state):
    """
    防自杀机制的核心逻辑 (numba加速)
    
    参数:
        q_values: 原始Q值数组 (3个动作)
        danger_signals: 危险信号数组 (3个方向)
        collision_penalty: 碰撞惩罚值
        state: 当前状态向量
        
    返回:
        action: 选择的动作
        danger_actions: 危险动作列表
    """
    # 创建安全动作掩码
    safe_mask = np.ones_like(q_values, dtype=np.bool_)
    danger_actions = []
    
    # 标记危险动作
    for i in range(len(danger_signals)):
        if danger_signals[i] > 0.5:
            safe_mask[i] = False
            danger_actions.append(i)
    
    # 创建安全Q值数组
    safe_q_values = q_values.copy()
    safe_q_values[~safe_mask] = -np.inf
    
    # 选择安全动作中Q值最高的
    if np.all(~safe_mask):
        # 所有动作都危险时选择原始Q值最高的
        action = np.argmax(q_values)
    else:
        action = np.argmax(safe_q_values)
    
    return action, danger_actions

@njit_decorator
def get_head_pos_nb(head, width, height):
    """获取蛇头坐标 (归一化)"""
    return np.array([
        head[0] / width,
        head[1] / height
    ], dtype=np.float32)

@njit_decorator
def get_food_pos_nb(food, width, height):
    """获取食物坐标 (归一化)"""
    return np.array([
        food[0] / width,
        food[1] / height
    ], dtype=np.float32)

@njit_decorator
def get_relative_distance_nb(head, food, width, height):
    """获取与食物的相对距离 (归一化)"""
    return np.array([
        (food[0] - head[0]) / width,
        (food[1] - head[1]) / height
    ], dtype=np.float32)

@njit_decorator
def get_manhattan_distance_nb(head, food, width, height):
    """获取与食物的曼哈顿距离 (归一化)"""
    dist = abs(food[0] - head[0]) + abs(food[1] - head[1])
    return np.array([dist / (width + height)], dtype=np.float32)

from direction import Direction

@njit_decorator
def get_direction_onehot_nb(direction):
    """获取当前移动方向 (4维one-hot)"""
    direction_vec = np.zeros(4, dtype=np.float32)
    if direction == Direction.RIGHT.value:
        direction_vec[0] = 1.0
    elif direction == Direction.LEFT.value:
        direction_vec[1] = 1.0
    elif direction == Direction.UP.value:
        direction_vec[2] = 1.0
    elif direction == Direction.DOWN.value:
        direction_vec[3] = 1.0
    return direction_vec

@njit_decorator
def get_eight_direction_dangers_nb(head, snake, width, height, BLOCK_SIZE, steps=3):
    """八方向的危险检查，可配置检查格子数量"""
    eight_directions = [
        (1, 0),   # 右
        (1, 1),   # 右下
        (0, 1),   # 下
        (-1, 1),  # 左下
        (-1, 0),  # 左
        (-1, -1), # 左上
        (0, -1),  # 上
        (1, -1)   # 右上
    ]
    
    direction_dangers = np.zeros(8 * steps, dtype=np.float32)
    for dir_idx, (dx, dy) in enumerate(eight_directions):
        for step in range(1, steps + 1):  # 检测1-steps格距离
            check_x = head[0] + dx * BLOCK_SIZE * step
            check_y = head[1] + dy * BLOCK_SIZE * step
            danger_idx = dir_idx * steps + (step - 1)
            direction_dangers[danger_idx] = is_collision_nb(
                snake, width, height, (check_x, check_y)
            )
    return direction_dangers

@njit_decorator
def get_boundary_distances_nb(head, width, height):
    """蛇头到四边界的距离 (归一化)"""
    return np.array([
        head[0] / width,                    # 左边界
        (width - head[0]) / width,          # 右边界
        head[1] / height,                   # 上边界
        (height - head[1]) / height         # 下边界
    ], dtype=np.float32)

@njit_decorator
def get_snake_length_nb(snake, grid_area):
    """当前蛇的长度 (归一化)"""
    return np.array([len(snake) / grid_area], dtype=np.float32)

@njit_decorator
def get_free_space_ratio_nb(snake, grid_area):
    """剩余空格比例 (归一化)"""
    return np.array([(grid_area - len(snake)) / grid_area], dtype=np.float32)

@njit_decorator
def get_local_grid_view_nb(head, snake, food, width, height, BLOCK_SIZE, radius=3):
    """以蛇头为中心的局部网格状态，可配置半径"""
    grid_view = np.zeros((2 * radius) * (2 * radius), dtype=np.float32)
    center_x, center_y = head[0] // BLOCK_SIZE, head[1] // BLOCK_SIZE
    
    for dx in range(-radius, radius):
        for dy in range(-radius, radius):
            # 计算网格位置
            grid_x = center_x + dx
            grid_y = center_y + dy
            idx = (dx + radius) * (2 * radius) + (dy + radius)
            
            # 检查是否在边界内
            if (0 <= grid_x < (width // BLOCK_SIZE) and 
                0 <= grid_y < (height // BLOCK_SIZE)):
                # 转换为像素坐标
                pixel_x = grid_x * BLOCK_SIZE
                pixel_y = grid_y * BLOCK_SIZE
                
                # 检查是否有蛇身或食物
                if (pixel_x, pixel_y) in snake:
                    grid_view[idx] = 1.0  # 蛇身
                elif (pixel_x, pixel_y) == food:
                    grid_view[idx] = 0.5  # 食物
                else:
                    grid_view[idx] = 0.0  # 空地
            else:
                grid_view[idx] = 1.0  # 边界外视为障碍
    return grid_view

@njit_decorator
def get_all_grid_view_nb(snake, food, width, height, BLOCK_SIZE):
    """获取整个地图的直接网格信息并展平为向量"""
    grid_width = width // BLOCK_SIZE
    grid_height = height // BLOCK_SIZE
    grid_area = grid_width * grid_height
    
    # 创建单通道地图
    full_map = np.zeros(grid_area, dtype=np.float32)
    
    # 填充蛇身
    for i in range(len(snake)):
        segment_x = int(snake[i][0] // BLOCK_SIZE)
        segment_y = int(snake[i][1] // BLOCK_SIZE)
        if 0 <= segment_x < grid_width and 0 <= segment_y < grid_height:
            idx = segment_y * grid_width + segment_x
            full_map[idx] = 1.0
    
    # 填充食物
    food_x = int(food[0] // BLOCK_SIZE)
    food_y = int(food[1] // BLOCK_SIZE)
    if 0 <= food_x < grid_width and 0 <= food_y < grid_height:
        idx = food_y * grid_width + food_x
        full_map[idx] = 0.5
    
    return full_map

@njit_decorator
def get_action_history_onehot_nb(action_history, history_length=5):
    """当前的动作历史记录，可配置长度"""
    action_vec = np.zeros(history_length * 3, dtype=np.float32)
    start_idx = max(0, history_length - len(action_history))
    
    for i, action in enumerate(action_history[-history_length:]):
        action_idx = (start_idx + i) * 3
        if action == 0:  # 直行
            action_vec[action_idx] = 1.0
        elif action == 1:  # 右转
            action_vec[action_idx + 1] = 1.0
        elif action == 2:  # 左转
            action_vec[action_idx + 2] = 1.0
    return action_vec