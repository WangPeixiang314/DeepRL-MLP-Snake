#!/usr/bin/env python3
"""
架构兼容性测试脚本
验证增强版模型架构的正确性和兼容性
"""

import torch
import numpy as np
import os
from config import Config
from model import EnhancedDuelingDQN, DuelingDQN
from agent import DuelingDDQNAgent

def test_enhanced_architecture():
    """测试增强版架构"""
    print("🔬 测试增强版Dueling DQN架构...")
    
    # 设置测试参数
    state_size = 50
    action_size = 4
    batch_size = 32
    
    # 测试不同配置
    configs = [
        {"use_attention": True, "activation": "swish", "use_residual": True},
        {"use_attention": False, "activation": "gelu", "use_residual": True},
        {"use_attention": True, "activation": "relu", "use_residual": False},
        {"use_attention": False, "activation": "swish", "use_residual": False},
    ]
    
    for i, cfg in enumerate(configs):
        print(f"\n配置 {i+1}: attention={cfg['use_attention']}, "
              f"activation={cfg['activation']}, residual={cfg['use_residual']}")
        
        try:
            # 创建模型
            model = EnhancedDuelingDQN(
                input_dim=state_size,
                hidden_layers=[256, 128, 64],
                output_dim=action_size,
                activation=cfg['activation'],
                use_attention=cfg['use_attention'],
                use_residual=cfg['use_residual']
            )
            
            # 测试前向传播
            test_input = torch.randn(batch_size, state_size)
            output = model(test_input)
            
            print(f"  ✅ 输入形状: {test_input.shape}")
            print(f"  ✅ 输出形状: {output.shape}")
            print(f"  ✅ 参数数量: {sum(p.numel() for p in model.parameters()):,}")
            
            # 测试价值函数和优势函数
            value = model.get_value(test_input)
            advantage = model.get_advantage(test_input)
            
            print(f"  ✅ 价值输出形状: {value.shape}")
            print(f"  ✅ 优势输出形状: {advantage.shape}")
            
            # 验证Dueling公式
            expected_q = value + advantage - advantage.mean(dim=1, keepdim=True)
            assert torch.allclose(output, expected_q, atol=1e-6), "Dueling公式验证失败"
            print(f"  ✅ Dueling公式验证通过")
            
        except Exception as e:
            print(f"  ❌ 测试失败: {e}")
            return False
    
    return True

def test_agent_initialization():
    """测试Agent初始化"""
    print("\n🔬 测试Agent初始化...")
    
    try:
        # 测试增强版Agent
        Config.USE_ENHANCED_MODEL = True
        Config.USE_ATTENTION = True
        Config.USE_RESIDUAL = True
        Config.ENHANCED_ACTIVATION = "swish"
        
        state_size = 50
        agent = DuelingDDQNAgent(state_size)
        
        print(f"  ✅ 增强版Agent创建成功")
        print(f"  ✅ 策略网络类型: {type(agent.policy_net).__name__}")
        print(f"  ✅ 目标网络类型: {type(agent.target_net).__name__}")
        
        # 测试旧版Agent
        Config.USE_ENHANCED_MODEL = False
        agent_old = DuelingDDQNAgent(state_size)
        
        print(f"  ✅ 旧版Agent创建成功")
        print(f"  ✅ 策略网络类型: {type(agent_old.policy_net).__name__}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Agent初始化失败: {e}")
        return False

def test_model_save_load():
    """测试模型保存和加载"""
    print("\n🔬 测试模型保存和加载...")
    
    try:
        # 创建测试模型
        state_size = 50
        action_size = 4
        
        # 测试增强版模型
        Config.USE_ENHANCED_MODEL = True
        Config.USE_ATTENTION = True
        Config.USE_RESIDUAL = True
        
        agent = DuelingDDQNAgent(state_size)
        
        # 保存模型
        test_path = "models/test_enhanced_model.pth"
        os.makedirs("models", exist_ok=True)
        agent.save_model()
        
        print(f"  ✅ 增强版模型保存成功")
        
        # 创建新agent并加载
        new_agent = DuelingDDQNAgent(state_size)
        success = new_agent.load_model("snake_enhanced_dqn_ep0_sc0.pth")
        
        if success:
            print(f"  ✅ 增强版模型加载成功")
        else:
            print(f"  ❌ 增强版模型加载失败")
            return False
            
        # 清理测试文件
        try:
            os.remove("models/snake_enhanced_dqn_ep0_sc0.pth")
        except:
            pass
            
        return True
        
    except Exception as e:
        print(f"  ❌ 保存/加载测试失败: {e}")
        return False

def test_memory_efficiency():
    """测试内存效率"""
    print("\n🔬 测试内存效率...")
    
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # 测试不同配置的内存使用
        state_size = 50
        action_size = 4
        
        configs = [
            ("旧版Dueling", False, False, False, "relu"),
            ("增强版基础", True, False, False, "swish"),
            ("增强版+残差", True, False, True, "swish"),
            ("增强版+注意力", True, True, False, "swish"),
            ("增强版完整", True, True, True, "swish"),
        ]
        
        for name, enhanced, attention, residual, activation in configs:
            Config.USE_ENHANCED_MODEL = enhanced
            Config.USE_ATTENTION = attention
            Config.USE_RESIDUAL = residual
            Config.ENHANCED_ACTIVATION = activation
            
            # 创建模型并测量内存
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            if enhanced:
                model = EnhancedDuelingDQN(state_size, [256, 128, 64], action_size, 
                                         activation, attention, residual)
            else:
                model = DuelingDQN(state_size, [256, 128, 64], action_size)
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            
            param_count = sum(p.numel() for p in model.parameters())
            memory_used = memory_after - memory_before
            
            print(f"  {name}: {param_count:,} 参数, {memory_used:.1f} MB")
        
        return True
        
    except ImportError:
        print("  ⚠️  psutil未安装，跳过内存效率测试")
        return True
    except Exception as e:
        print(f"  ❌ 内存效率测试失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("🚀 开始架构兼容性测试...")
    print("=" * 50)
    
    tests = [
        ("增强版架构测试", test_enhanced_architecture),
        ("Agent初始化测试", test_agent_initialization),
        ("保存/加载测试", test_model_save_load),
        ("内存效率测试", test_memory_efficiency),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        success = test_func()
        results.append((test_name, success))
        print("-" * 30)
    
    # 总结结果
    print("\n📊 测试结果总结:")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 个测试通过")
    
    if passed == len(results):
        print("🎉 所有测试通过！架构兼容性良好")
    else:
        print("⚠️  部分测试失败，请检查问题")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)