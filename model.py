import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


class DuelingDQN(nn.Module):
    """
    Dueling DQN架构：
    - 共享特征提取层
    - 分流为Value流和Advantage流
    - 支持Double DQN的训练策略
    """
    
    def __init__(self, input_dim, hidden_layers, output_dim):
        super(DuelingDQN, self).__init__()
        
        # 共享特征提取层
        self.feature_layers = nn.ModuleList()
        prev_dim = input_dim
        
        # 特征提取层（通常占总隐藏层的60-70%）
        feature_layers_count = max(1, len(hidden_layers) * 2 // 3)
        for i in range(feature_layers_count):
            layer_dim = hidden_layers[i]
            self.feature_layers.append(nn.Linear(prev_dim, layer_dim))
            self.feature_layers.append(nn.ReLU())
            # 添加Dropout防止过拟合
            self.feature_layers.append(nn.Dropout(Config.DROPOUT_RATE))
            prev_dim = layer_dim
        
        # 记录特征维度
        self.feature_dim = prev_dim
        
        # Value流 - 估计状态价值V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(self.feature_dim // 2, 1)  # 输出单个标量V(s)
        )
        
        # Advantage流 - 估计动作优势A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(self.feature_dim // 2, output_dim)  # 输出每个动作的advantage
        )
        
        # 权重初始化
        self.apply(self._init_weights)
        
        self.to(Config.device)
    
    def _init_weights(self, module):
        """权重初始化策略"""
        if isinstance(module, nn.Linear):
            if Config.WEIGHT_INIT == 'xavier_uniform':
                nn.init.xavier_uniform_(module.weight)
            elif Config.WEIGHT_INIT == 'xavier_normal':
                nn.init.xavier_normal_(module.weight)
            elif Config.WEIGHT_INIT == 'kaiming_uniform':
                nn.init.kaiming_uniform_(module.weight)
            elif Config.WEIGHT_INIT == 'kaiming_normal':
                nn.init.kaiming_normal_(module.weight)
            else:
                nn.init.xavier_uniform_(module.weight)
            
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        前向传播：
        Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
        
        Args:
            x: 输入状态 [batch_size, input_dim]
            
        Returns:
            Q值 [batch_size, output_dim]
        """
        # 特征提取
        features = x
        for layer in self.feature_layers:
            features = layer(features)
        
        # Value流
        value = self.value_stream(features)  # [batch_size, 1]
        
        # Advantage流
        advantage = self.advantage_stream(features)  # [batch_size, output_dim]
        
        # Dueling DQN的核心公式
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
        # 使用mean来保持可识别性
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values
    
    def get_value(self, x):
        """获取状态价值V(s) - 用于分析"""
        features = x
        for layer in self.feature_layers:
            features = layer(features)
        return self.value_stream(features)
    
    def get_advantage(self, x):
        """获取动作优势A(s,a) - 用于分析"""
        features = x
        for layer in self.feature_layers:
            features = layer(features)
        return self.advantage_stream(features)
    
    def save(self, filename=Config.MODEL_FILE):
        """保存模型"""
        path = os.path.join(Config.MODEL_DIR, filename)
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_type': 'dueling_dqn',
            'input_dim': self.feature_dim,
            'output_dim': self.advantage_stream[-1].out_features
        }, path)
        print(f"Dueling DQN模型已保存: {path}")
    
    def load(self, filename=Config.MODEL_FILE):
        """加载模型"""
        path = os.path.join(Config.MODEL_DIR, filename)
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=Config.device)
            if 'model_type' in checkpoint and checkpoint['model_type'] == 'dueling_dqn':
                self.load_state_dict(checkpoint['model_state_dict'])
                self.eval()
                print(f"Dueling DQN模型已加载: {path}")
                return True
            else:
                print("警告：尝试加载的模型不是Dueling DQN格式")
                return False
        return False


class DuelingDQNWithNoisy(DuelingDQN):
    """
    带噪声网络的Dueling DQN（Noisy Networks）
    用于替代ε-greedy的探索策略
    """
    
    def __init__(self, input_dim, hidden_layers, output_dim):
        super().__init__(input_dim, hidden_layers, output_dim)
        
        # 替换线性层为噪声层（可选的高级功能）
        # 这里可以添加NoisyLinear层的实现
        
    def reset_noise(self):
        """重置噪声（用于每个episode）"""
        pass  # 如果使用Noisy Networks，这里实现噪声重置


# 测试代码
if __name__ == "__main__":
    # 测试Dueling DQN
    test_input_dim = 50
    test_hidden = [256, 128, 64]
    test_output = 3
    
    model = DuelingDQN(test_input_dim, test_hidden, test_output)
    
    # 测试前向传播
    test_input = torch.randn(32, test_input_dim).to(Config.device)
    output = model(test_input)
    
    print(f"输入维度: {test_input_dim}")
    print(f"输出维度: {test_output}")
    print(f"输出形状: {output.shape}")
    print(f"Dueling DQN模型创建成功！")