#!/usr/bin/env python3
"""
Dueling Double DQN 训练脚本
结合Dueling架构和Double DQN的优势
支持命令行参数: --episodes <数量> --model_path <路径> --visualize
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd
import os
import argparse

from game import SnakeGame as SnakeGameAI
from agent import DuelingDDQNAgent
from config import Config
from training import ConsoleLogger, TrainingStats, TrainingPlotter
import pygame


def get_state_size():
    """通过实际游戏状态获取状态空间大小"""
    # 创建一个临时游戏实例来获取实际状态维度
    game = SnakeGameAI(visualize=False)
    state = game.reset()
    return len(state)


def train_dueling_dqn(max_episodes=None, model_path=None, visualize=True, fresh_start=False):
    """训练Dueling Double DQN，支持自定义参数"""
    
    # 初始化
    state_size = get_state_size()
    agent = DuelingDDQNAgent(state_size)
    
    # 如果提供了模型路径且不是强制重新开始，尝试加载
    if not fresh_start and model_path and os.path.exists(model_path):
        print(f"加载已有模型: {model_path}")
        agent.load_model(model_path)
    elif fresh_start:
        print("🔄 强制重新开始训练，跳过模型加载")
    
    game = SnakeGameAI(visualize=visualize)
    
    # 训练统计
    scores = []
    mean_scores = []
    losses = []
    epsilons = []
    
    # 最近100局的平均分数
    recent_scores = deque(maxlen=100)
    
    # 记录训练开始时间
    start_time = time.time()
    
    # 初始化控制台日志和绘图器
    console_logger = ConsoleLogger()
    console_logger.print_training_header()
    
    # 初始化训练统计和实时绘图（仅在可视化模式下）
    if visualize:
        training_stats = TrainingStats()
        plotter = TrainingPlotter()
        print("实时绘图已启用")
    else:
        training_stats = None
        plotter = None
        print("实时绘图已禁用（非可视化模式）")
    print(f"状态维度: {state_size}")
    print(f"动作维度: {Config.OUTPUT_DIM}")
    print("=" * 80)
    
    # 训练循环
    total_episodes = max_episodes if max_episodes else Config.MAX_EPISODES
    
    try:
        for episode in range(total_episodes):
            
            # 重置游戏
            state = game.reset()
            total_loss = 0
            steps = 0
            
            while True:
                # 选择动作
                action = agent.select_action(state)
                
                # 执行动作
                next_state, reward, done, score = game.step(action)
                
                # 存储经验
                agent.memory.add(state, action, reward, next_state, done)
                
                # 优化模型
                loss, td_errors = agent.optimize_model()
                if loss > 0:
                    total_loss += loss
                    losses.append(loss)
                
                state = next_state
                steps += 1
                
                if done:
                    break
            
            # 记录分数
            scores.append(score)
            recent_scores.append(score)
            epsilons.append(agent.epsilon_threshold)
            
            # 计算最近100局平均分数
            mean_score = np.mean(recent_scores) if recent_scores else 0
            mean_scores.append(mean_score)
            
            # 更新agent记录
            agent.record_score(score)
            
            # 更新实时训练统计
            if training_stats is not None:
                # 计算每局总奖励（简化版本）
                total_reward = score * Config.FOOD_REWARD - (steps - score) * 0.1
                training_stats.update(
                    score=score,
                    reward=total_reward,
                    steps=steps,
                    loss=total_loss/steps if steps > 0 else 0,
                    epsilon=agent.epsilon_threshold,
                    final_length=score + 3  # 简化：蛇初始长度3 + 得分
                )
                
                # 实时更新图表
                plotter.update(training_stats.get_stats())
            
            # 每5局打印训练状态
            elapsed_time = time.time() - start_time
            console_logger.print_episode_status(
                episode=episode,
                score=score,
                avg_score=mean_score,
                loss=total_loss/steps if steps > 0 else 0,
                epsilon=agent.epsilon_threshold,
                steps=steps,
                lr=Config.LEARNING_RATE
            )
            
            # 每100局绘制训练曲线
            if episode > 0 and episode % 100 == 0:
                plot_training_progress(scores, mean_scores, losses, epsilons)
                save_training_log(scores, mean_scores, losses, epsilons)
                
                # 检查是否达到目标分数（使用50作为默认目标）
                target_score = 50  # 默认目标分数
                if mean_score >= target_score:
                    print(f"达到目标分数 {target_score}，训练完成！")
                    agent.save_model(is_best=True)
                    break
    
    except KeyboardInterrupt:
        print("\n训练被中断")
    
    finally:
        # 保存最终模型
        agent.save_model()
        
        # 关闭实时绘图窗口
        if plotter is not None:
            plotter.close()
            print("实时绘图窗口已关闭")
        
        # 绘制最终训练曲线
        plot_training_progress(scores, mean_scores, losses, epsilons)
        save_training_log(scores, mean_scores, losses, epsilons)
        
        # 打印训练总结
        elapsed_time = time.time() - start_time
        console_logger.print_training_complete(
            total_episodes=len(scores),
            max_score=max(scores),
            avg_score=np.mean(recent_scores),
            total_time=elapsed_time
        )


def plot_training_progress(scores, mean_scores, losses, epsilons):
    """绘制训练进度图"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Dueling Double DQN 训练进度', fontsize=16)
    
    # 分数曲线
    ax1.plot(scores, alpha=0.6, label='每局分数', color='lightblue')
    ax1.plot(mean_scores, label='100局平均', color='blue', linewidth=2)
    ax1.set_title('游戏分数')
    ax1.set_xlabel('局数')
    ax1.set_ylabel('分数')
    ax1.legend()
    ax1.grid(True)
    
    # 损失曲线
    if losses:
        # 平滑损失曲线
        window = min(100, len(losses))
        smoothed_losses = pd.Series(losses).rolling(window=window).mean()
        ax2.plot(smoothed_losses, color='red', linewidth=2)
        ax2.set_title('训练损失')
        ax2.set_xlabel('训练步数')
        ax2.set_ylabel('损失')
        ax2.grid(True)
    
    # ε衰减曲线
    ax3.plot(epsilons, color='green', linewidth=2)
    ax3.set_title('Epsilon衰减')
    ax3.set_xlabel('局数')
    ax3.set_ylabel('Epsilon')
    ax3.grid(True)
    
    # 分数分布直方图
    if scores:
        ax4.hist(scores[-100:], bins=20, color='purple', alpha=0.7)
        ax4.axvline(np.mean(scores[-100:]), color='red', linestyle='--', 
                   label=f'平均: {np.mean(scores[-100:]):.1f}')
        ax4.set_title('最近100局分数分布')
        ax4.set_xlabel('分数')
        ax4.set_ylabel('频次')
        ax4.legend()
        ax4.grid(True)
    
    plt.tight_layout()
    
    # 保存图片
    plot_path = os.path.join(Config.MODEL_DIR, 'dueling_training_progress.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"训练进度图已保存: {plot_path}")


def save_training_log(scores, mean_scores, losses, epsilons):
    """保存训练日志"""
    
    df = pd.DataFrame({
        'episode': range(len(scores)),
        'score': scores,
        'mean_score_100': mean_scores,
        'epsilon': epsilons[:len(scores)],
        'loss': [np.mean(losses[max(0, i-100):i+1]) if losses else 0 
                for i in range(len(scores))]
    })
    
    log_path = os.path.join(Config.MODEL_DIR, 'dueling_training_log.csv')
    df.to_csv(log_path, index=False)
    
    print(f"📝 训练日志已保存: {log_path}")


def compare_models():
    """对比Dueling DDQN和普通DDQN的性能"""
    
    print("🔍 模型性能对比功能")
    print("=" * 50)
    
    # 检查是否存在普通DDQN模型
    from agent import DDQNAgent
    
    # 创建测试环境
    state_size = get_state_size()
    
    # 测试Dueling DDQN
    dueling_agent = DuelingDDQNAgent(state_size)
    
    # 测试普通DDQN
    regular_agent = DDQNAgent(state_size)
    
    print("模型对比功能已就绪！")
    print("使用 train_dueling.py 训练Dueling版本")
    print("使用 train.py 训练普通版本")


if __name__ == "__main__":
    
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='Dueling Double DQN 训练系统')
    parser.add_argument('--episodes', type=int, default=None,
                        help='训练局数 (默认使用config.py中的设置)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='要加载的模型路径')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='是否显示游戏画面')
    parser.add_argument('--no-visualize', dest='visualize', action='store_false',
                        help='关闭游戏画面显示')
    parser.add_argument('--fresh', action='store_true',
                        help='强制重新开始训练，不加载旧模型')
    
    args = parser.parse_args()
    
    print("Dueling Double DQN 训练系统")
    print("=" * 50)
    print(f"训练局数: {args.episodes if args.episodes else Config.MAX_EPISODES}")
    print(f"模型路径: {args.model_path if args.model_path else '新模型'}")
    print(f"可视化: {'开启' if args.visualize else '关闭'}")
    print(f"强制重新开始: {'是' if args.fresh else '否'}")
    
    # 显示当前模型配置
    print("=" * 50)
    print("当前模型配置:")
    print(f"  增强版模型: {'启用' if Config.USE_ENHANCED_MODEL else '禁用'}")
    if Config.USE_ENHANCED_MODEL:
        print(f"  激活函数: {Config.ENHANCED_ACTIVATION}")
        print(f"  注意力机制: {'启用' if Config.USE_ATTENTION else '禁用'}")
        print(f"  残差连接: {'启用' if Config.USE_RESIDUAL else '禁用'}")
    print("=" * 50)
    
    # 检查CUDA可用性
    import torch
    print(f"设备: {Config.device}")
    print(f"PyTorch版本: {torch.__version__}")
    
    # 开始训练
    train_dueling_dqn(
        max_episodes=args.episodes,
        model_path=args.model_path,
        visualize=args.visualize,
        fresh_start=args.fresh
    )