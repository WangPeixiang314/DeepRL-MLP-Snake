import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.ERROR)

optuna_logger = logging.getLogger('optuna')
optuna_logger.setLevel(logging.ERROR)

import optuna
from config import Config
from train import train

def objective(trial):
    # 1. 网络结构参数
    layer_count = trial.suggest_int('layer_count', 3, 8)
    base_size = trial.suggest_categorical('base_size', [128, 256, 512, 768, 1024])
    size_decay = trial.suggest_float('size_decay', 0.3, 0.9)  # 层间衰减率
    Config.HIDDEN_LAYERS = [int(base_size * (size_decay ** i)) for i in range(layer_count)]
    
    # 2. 学习率和优化参数
    Config.LEARNING_RATE = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
    Config.BATCH_SIZE = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512, 1024, 2048])
    
    # 3. 探索策略参数
    Config.EPS_START = trial.suggest_float('eps_start', 0.8, 1.0)
    Config.EPS_END = trial.suggest_float('eps_end', 0.01, 0.2)
    Config.EPS_DECAY = trial.suggest_int('eps_decay', 1000, 30000, step=1000)
    
    # 4. 经验回放参数
    Config.MEMORY_CAPACITY = trial.suggest_categorical('memory_capacity', 
                                                      [50_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000])
    Config.PRIO_ALPHA = trial.suggest_float('prio_alpha', 0.4, 0.9, step=0.1)
    Config.PRIO_BETA_START = trial.suggest_float('prio_beta_start', 0.3, 0.7)
    Config.PRIO_BETA_FRAMES = trial.suggest_int('prio_beta_frames', 10_000, 200_000, step=10_000)
    
    # 5. 折扣因子和更新频率
    Config.GAMMA = trial.suggest_float('gamma', 0.9, 0.999, step=0.01)
    Config.TARGET_UPDATE = trial.suggest_int('target_update', 50, 500, step=50)
    
    # 6. 奖励函数参数（增加碰撞惩罚优化）
    Config.FOOD_REWARD = trial.suggest_float('food_reward', 5.0, 30.0, step=1.0)
    Config.COLLISION_PENALTY = trial.suggest_float('collision_penalty', -30.0, -5.0, step=1.0)
    Config.PROGRESS_REWARD = trial.suggest_float('progress_reward', 0, 2, step=0.01)
    Config.STEP_PENALTY = trial.suggest_float('step_penalty', -1, 1, step=0.01)
    
    # 7. 训练稳定性参数
    Config.MAX_STEPS_WITHOUT_FOOD = trial.suggest_int('max_steps_without_food', 200, 1000, step=100)
    
    # 训练并返回分数
    score = train(num_episodes=3500, visualize=False, verbose=False)
    
    # 记录超参数优化过程信息
    log_file = "hyperparameter_optimization_log.txt"
    try:
        current_best_score = trial.study.best_value
    except ValueError:
        current_best_score = score

    log_line = f"当前超参数组合为：{trial.params}\n该组合评分为：{score:.4f}，当前历史最高评分：{max(current_best_score, score):.4f}"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_line + '\n')
    print(log_line)
    
    return score


import json
from optuna.trial import FrozenTrial, TrialState

def load_initial_trials(file_path):
    """加载已知超参数组合及其分数作为初始试验"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            initial_data = json.load(f)
            
        trials = []
        for i, entry in enumerate(initial_data):
            trial = FrozenTrial(
                number=i,
                state=TrialState.COMPLETE,
                params=entry['params'],
                value=entry['score'],
                user_attrs={},
                system_attrs={},
                intermediate_values={},
            )
            trials.append(trial)
        return trials
    except FileNotFoundError:
        print(f"警告: 未找到初始超参数文件 {file_path}，将使用默认搜索策略")
        return []
    except Exception as e:
        print(f"加载初始超参数时出错: {str(e)}")
        return []

if __name__ == '__main__':
    # 创建Optuna研究
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(n_startup_trials=35),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=400),
    )
    
    # 加载并添加已知超参数组合
    initial_trials = load_initial_trials('initial_hyperparameters.json')
    for trial in initial_trials:
        study.add_trial(trial)
    
    # 运行优化
    study.optimize(objective, n_trials=1000, show_progress_bar=True)
    
    # 打印最佳结果
    print("\n" + "="*50)
    print(f"最佳分数: {study.best_value:.2f}")
    print("最佳超参数组合:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    
    # 保存最佳参数
    with open('best_hyperparameters.txt', 'w') as f:
        f.write(f"最佳分数: {study.best_value:.2f}\n")
        f.write("最佳超参数组合:\n")
        for key, value in study.best_params.items():
            f.write(f"{key}: {value}\n")
    
    # 可视化优化过程
    fig = optuna.visualization.plot_optimization_history(study)
    fig.show()
    
    fig = optuna.visualization.plot_param_importances(study)
    fig.show()