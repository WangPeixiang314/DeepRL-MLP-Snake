import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class DQN(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim):
        super(DQN, self).__init__()
        # 创建隐藏层
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for layer_dim in hidden_layers:
            self.layers.append(nn.Linear(prev_dim, layer_dim))
            prev_dim = layer_dim
        # 创建输出层
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        self.to(Config.device)
    
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.output_layer(x)
    
    def save(self, filename=Config.MODEL_FILE):
        """保存模型"""
        path = os.path.join(Config.MODEL_DIR, filename)
        torch.save(self.state_dict(), path)
        print(f"模型已保存: {path}")
    
    def load(self, filename=Config.MODEL_FILE):
        """加载模型"""
        path = os.path.join(Config.MODEL_DIR, filename)
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, map_location=Config.device))
            self.eval()
            print(f"模型已加载: {path}")
            return True
        return False