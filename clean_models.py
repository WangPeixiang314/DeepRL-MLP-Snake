#!/usr/bin/env python3
"""
模型清理工具
用于清理旧的模型文件，确保与新的增强版架构兼容
"""

import os
import shutil
import argparse
from pathlib import Path

def clean_old_models(models_dir="models", backup=True):
    """清理旧的模型文件"""
    
    models_path = Path(models_dir)
    if not models_path.exists():
        print(f"模型目录不存在: {models_dir}")
        return
    
    # 创建备份目录
    if backup:
        backup_dir = models_path / "backup_old_models"
        backup_dir.mkdir(exist_ok=True)
        print(f"旧模型将备份到: {backup_dir}")
    
    # 识别旧版模型文件
    old_model_files = []
    enhanced_model_files = []
    
    for file in models_path.glob("*.pth"):
        if "enhanced" in file.name.lower():
            enhanced_model_files.append(file)
        else:
            old_model_files.append(file)
    
    print(f"找到 {len(old_model_files)} 个旧版模型文件")
    print(f"找到 {len(enhanced_model_files)} 个增强版模型文件")
    
    if not old_model_files:
        print("没有需要清理的旧版模型文件")
        return
    
    # 显示将要清理的文件
    print("\n将要清理的旧版模型文件:")
    for file in old_model_files:
        size_mb = file.stat().st_size / (1024*1024)
        print(f"  {file.name} ({size_mb:.1f} MB)")
    
    # 确认清理
    response = input("\n确认清理这些文件? (y/N): ").strip().lower()
    if response != 'y':
        print("取消清理操作")
        return
    
    # 执行清理
    cleaned_count = 0
    for file in old_model_files:
        try:
            if backup:
                shutil.move(str(file), str(backup_dir / file.name))
            else:
                file.unlink()
            cleaned_count += 1
        except Exception as e:
            print(f"清理失败 {file.name}: {e}")
    
    print(f"已清理 {cleaned_count} 个旧版模型文件")
    
    if backup:
        print(f"旧模型已备份到: {backup_dir}")
        print("如需恢复，请手动从backup_old_models目录复制文件")

def list_models(models_dir="models"):
    """列出所有模型文件"""
    
    models_path = Path(models_dir)
    if not models_path.exists():
        print(f"模型目录不存在: {models_dir}")
        return
    
    print("当前模型文件:")
    print("-" * 50)
    
    enhanced_files = []
    old_files = []
    
    for file in models_path.glob("*.pth"):
        size_mb = file.stat().st_size / (1024*1024)
        if "enhanced" in file.name.lower():
            enhanced_files.append((file.name, size_mb))
        else:
            old_files.append((file.name, size_mb))
    
    if enhanced_files:
        print("增强版模型:")
        for name, size in enhanced_files:
            print(f"  📁 {name} ({size:.1f} MB)")
    
    if old_files:
        print("旧版模型:")
        for name, size in old_files:
            print(f"  📄 {name} ({size:.1f} MB)")
    
    if not enhanced_files and not old_files:
        print("没有找到模型文件")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='模型清理工具')
    parser.add_argument('--models-dir', default='models', help='模型目录路径')
    parser.add_argument('--no-backup', action='store_true', help='不创建备份')
    parser.add_argument('--list-only', action='store_true', help='仅列出模型，不清理')
    
    args = parser.parse_args()
    
    if args.list_only:
        list_models(args.models_dir)
    else:
        clean_old_models(args.models_dir, backup=not args.no_backup)
    
    print("\n操作完成!")