#!/usr/bin/env python3
"""
统计PyTorch模型文件中的参数数量
"""

import torch
import os
import sys


def count_parameters(model_path):
    """
    统计模型参数数量
    
    Args:
        model_path: 模型文件路径
    
    Returns:
        dict: 包含各种统计信息的字典
    """
    if not os.path.exists(model_path):
        print(f"错误：文件不存在: {model_path}")
        return None
    
    try:
        # 加载模型参数
        print(f"正在加载模型文件: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 统计信息
        stats = {
            'total_params': 0,
            'trainable_params': 0,
            'param_details': {},
            'file_size_mb': os.path.getsize(model_path) / (1024 * 1024)
        }
        
        # 检查数据结构
        if isinstance(checkpoint, dict):
            print("检测到字典格式的checkpoint")
            
            # 尝试找到模型参数
            model_params = None
            if 'model' in checkpoint:
                model_params = checkpoint['model']
                print("找到 'model' 键")
            elif 'state_dict' in checkpoint:
                model_params = checkpoint['state_dict']
                print("找到 'state_dict' 键")
            else:
                # 假设整个字典就是参数
                model_params = checkpoint
                print("将整个字典视为模型参数")
            
            # 统计每个参数层
            for name, param in model_params.items():
                if isinstance(param, torch.Tensor):
                    param_count = param.numel()
                    stats['total_params'] += param_count
                    stats['param_details'][name] = {
                        'shape': list(param.shape),
                        'params': param_count,
                        'dtype': str(param.dtype)
                    }
            
            # 打印其他可能的键
            print(f"checkpoint中的键: {list(checkpoint.keys())}")
            
        elif isinstance(checkpoint, torch.nn.Module):
            print("检测到模型对象")
            for name, param in checkpoint.named_parameters():
                param_count = param.numel()
                stats['total_params'] += param_count
                if param.requires_grad:
                    stats['trainable_params'] += param_count
                stats['param_details'][name] = {
                    'shape': list(param.shape),
                    'params': param_count,
                    'requires_grad': param.requires_grad,
                    'dtype': str(param.dtype)
                }
        else:
            print(f"未知的数据类型: {type(checkpoint)}")
            return None
        
        # 如果没有找到trainable_params信息，假设所有参数都可训练
        if stats['trainable_params'] == 0:
            stats['trainable_params'] = stats['total_params']
        
        return stats
        
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        return None


def format_number(num):
    """格式化数字，添加千分位分隔符"""
    return f"{num:,}"


def print_stats(stats, model_path):
    """打印统计结果"""
    if stats is None:
        return
    
    print("\n" + "="*60)
    print(f"模型文件: {model_path}")
    print("="*60)
    
    print(f"文件大小: {stats['file_size_mb']:.2f} MB")
    print(f"总参数数量: {format_number(stats['total_params'])}")
    print(f"可训练参数: {format_number(stats['trainable_params'])}")
    
    # 转换为更友好的单位
    total_params = stats['total_params']
    if total_params >= 1e9:
        print(f"总参数数量: {total_params / 1e9:.2f}B (十亿)")
    elif total_params >= 1e6:
        print(f"总参数数量: {total_params / 1e6:.2f}M (百万)")
    elif total_params >= 1e3:
        print(f"总参数数量: {total_params / 1e3:.2f}K (千)")
    
    print("\n参数层详细信息:")
    print("-" * 80)
    print(f"{'层名':<40} {'形状':<20} {'参数数量':<15} {'数据类型'}")
    print("-" * 80)
    
    # 按参数数量排序显示
    sorted_params = sorted(stats['param_details'].items(), 
                          key=lambda x: x[1]['params'], reverse=True)
    
    for name, details in sorted_params:
        shape_str = str(details['shape'])
        if len(shape_str) > 18:
            shape_str = shape_str[:15] + "..."
        
        print(f"{name:<40} {shape_str:<20} {format_number(details['params']):<15} {details['dtype']}")
    
    print("-" * 80)
    print(f"总计: {len(stats['param_details'])} 个参数层")


def main():
    model_path = "/mnt/data-raid/yangguangzhao/t2/results_cls_math/0/aqua_rat_1_mm1_qwen306b_RL-lr0.002-mGN0.001-klC0.01-rrN0CNone-st/policy_params.pt"
    
    print("PyTorch模型参数统计工具")
    print("=" * 50)
    
    stats = count_parameters(model_path)
    print_stats(stats, model_path)


if __name__ == "__main__":
    main()
