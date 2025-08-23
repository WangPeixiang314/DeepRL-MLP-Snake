import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
import math


class ResidualBlock(nn.Module):
    """
    残差连接块
    支持维度匹配和不匹配的情况
    """
    def __init__(self, input_dim, output_dim, activation='swish'):
        super(ResidualBlock, self).__init__()
        
        # 主路径
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        
        # 激活函数
        if activation == 'swish':
            self.activation = nn.SiLU()  # Swish激活函数
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
        
        # 层归一化
        self.norm1 = nn.LayerNorm(output_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(Config.DROPOUT_RATE)
        
        # 跳跃连接适配器（当维度不匹配时）
        self.shortcut = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        
    def forward(self, x):
        # 跳跃连接
        identity = self.shortcut(x)
        
        # 主路径
        out = self.fc1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.norm2(out)
        
        # 残差连接
        out = out + identity
        out = self.activation(out)
        
        return out


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    增强对关键特征的捕捉能力
    """
    def __init__(self, input_dim, num_heads=4, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert input_dim % num_heads == 0
        
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        # 线性变换
        self.q_linear = nn.Linear(input_dim, input_dim)
        self.k_linear = nn.Linear(input_dim, input_dim)
        self.v_linear = nn.Linear(input_dim, input_dim)
        self.out_linear = nn.Linear(input_dim, input_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        
        # 线性变换
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 注意力计算
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 加权求和
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, input_dim)
        
        # 输出变换
        output = self.out_linear(attended)
        
        return output


class EnhancedDuelingDQN(nn.Module):
    """
    增强版Dueling DQN架构：
    - 残差连接：缓解梯度消失，支持深层网络训练
    - 注意力机制：增强关键特征捕捉
    - 优化激活函数：Swish/GELU替代ReLU
    - 层归一化：稳定训练过程
    """
    
    def __init__(self, input_dim, hidden_layers, output_dim, activation='swish', use_attention=True):
        super(EnhancedDuelingDQN, self).__init__()
        
        self.use_attention = use_attention
        
        # 激活函数选择
        if activation == 'swish':
            self.activation = nn.SiLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
        
        # 输入层
        self.input_projection = nn.Linear(input_dim, hidden_layers[0])
        self.input_norm = nn.LayerNorm(hidden_layers[0])
        
        # 残差块层
        self.residual_blocks = nn.ModuleList()
        for i in range(len(hidden_layers)):
            if i == 0:
                self.residual_blocks.append(
                    ResidualBlock(hidden_layers[0], hidden_layers[0], activation)
                )
            else:
                self.residual_blocks.append(
                    ResidualBlock(hidden_layers[i-1], hidden_layers[i], activation)
                )
        
        # 注意力机制
        if use_attention:
            self.attention = MultiHeadAttention(hidden_layers[-1])
            self.attention_norm = nn.LayerNorm(hidden_layers[-1])
        
        # 特征维度
        self.feature_dim = hidden_layers[-1]
        
        # Value流 - 使用Tanh作为最后一层激活
        self.value_stream = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            self.activation,
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(self.feature_dim // 2, self.feature_dim // 4),
            self.activation,
            nn.Linear(self.feature_dim // 4, 1)  # 输出单个标量V(s)
        )
        
        # Advantage流 - 使用Tanh作为最后一层激活
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            self.activation,
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(self.feature_dim // 2, self.feature_dim // 4),
            self.activation,
            nn.Linear(self.feature_dim // 4, output_dim)  # 输出每个动作的advantage
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
        # 输入投影
        features = self.input_projection(x)
        features = self.input_norm(features)
        features = self.activation(features)
        
        # 残差块特征提取
        for block in self.residual_blocks:
            features = block(features)
        
        # 注意力机制
        if self.use_attention:
            # 添加序列维度用于注意力计算
            features_seq = features.unsqueeze(1)  # [batch_size, 1, feature_dim]
            attended = self.attention(features_seq)
            attended = attended.squeeze(1)  # [batch_size, feature_dim]
            features = self.attention_norm(features + attended)
        
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
        features = self.input_projection(x)
        features = self.input_norm(features)
        features = self.activation(features)
        
        for block in self.residual_blocks:
            features = block(features)
            
        if self.use_attention:
            features_seq = features.unsqueeze(1)
            attended = self.attention(features_seq)
            attended = attended.squeeze(1)
            features = self.attention_norm(features + attended)
            
        return self.value_stream(features)
    
    def get_advantage(self, x):
        """获取动作优势A(s,a) - 用于分析"""
        features = self.input_projection(x)
        features = self.input_norm(features)
        features = self.activation(features)
        
        for block in self.residual_blocks:
            features = block(features)
            
        if self.use_attention:
            features_seq = features.unsqueeze(1)
            attended = self.attention(features_seq)
            attended = attended.squeeze(1)
            features = self.attention_norm(features + attended)
            
        return self.advantage_stream(features)
    
    def save(self, filename=Config.MODEL_FILE):
        """保存模型"""
        path = os.path.join(Config.MODEL_DIR, filename)
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_type': 'enhanced_dueling_dqn',
            'input_dim': self.feature_dim,
            'output_dim': self.advantage_stream[-1].out_features,
            'use_attention': self.use_attention
        }, path)
        print(f"增强版Dueling DQN模型已保存: {path}")
    
    def load(self, filename=Config.MODEL_FILE):
        """加载模型"""
        path = os.path.join(Config.MODEL_DIR, filename)
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=Config.device)
            if 'model_type' in checkpoint and checkpoint['model_type'] == 'enhanced_dueling_dqn':
                self.load_state_dict(checkpoint['model_state_dict'])
                self.eval()
                print(f"增强版Dueling DQN模型已加载: {path}")
                return True
            else:
                print("警告：尝试加载的模型不是增强版Dueling DQN格式")
                return False
        return False


class DuelingDQN(nn.Module):
    """
    原始Dueling DQN架构（保持兼容性）
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
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values
    
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
    # 测试参数
    test_input_dim = 50
    test_hidden = [256, 128, 64]
    test_output = 3
    
    print("=== 测试增强版Dueling DQN ===")
    enhanced_model = EnhancedDuelingDQN(test_input_dim, test_hidden, test_output, 
                                      activation='swish', use_attention=True)
    
    # 测试前向传播
    test_input = torch.randn(32, test_input_dim).to(Config.device)
    enhanced_output = enhanced_model(test_input)
    
    print(f"输入维度: {test_input_dim}")
    print(f"隐藏层: {test_hidden}")
    print(f"输出维度: {test_output}")
    print(f"增强版输出形状: {enhanced_output.shape}")
    print(f"增强版Dueling DQN模型创建成功！")
    
    # 测试注意力机制
    if enhanced_model.use_attention:
        print("✅ 注意力机制已启用")
    
    # 测试激活函数
    print(f"✅ 激活函数: Swish")
    
    print("\n=== 测试原始Dueling DQN ===")
    original_model = DuelingDQN(test_input_dim, test_hidden, test_output)
    original_output = original_model(test_input)
    print(f"原始输出形状: {original_output.shape}")
    print(f"原始Dueling DQN模型创建成功！")