#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将所有节点的reinforce_log.json转换为结构化的CSV文件
每个节点生成单独的CSV，按iter排序，缺失的iter填充NA
"""

import json
import pandas as pd
from pathlib import Path

# 基础目录
base_dir = Path("/mnt/data-raid/yangguangzhao/t2/results_history/results——lora——8")

print("="*60)
print("开始处理JSON文件...")
print("="*60)

# 存储所有节点的数据，用于合并
all_nodes_data = []

# 为每个节点单独处理
for node_id in range(10):
    node_dir = base_dir / str(node_id)
    
    print(f"\n节点 {node_id}:")
    
    # 查找该节点目录下的reinforce_log.json文件（任意子目录）
    json_files = list(node_dir.glob("*/reinforce_log.json"))
    
    if not json_files:
        print(f"  ✗ 未找到reinforce_log.json文件，跳过")
        continue
    
    json_file = json_files[0]  # 取第一个匹配的文件
    print(f"  JSON文件: {json_file}")
    
    # 读取所有JSON记录（文件中包含多个格式化的JSON对象，对象之间没有分隔符）
    records = []
    with open(json_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用JSONDecoder逐个解析JSON对象
    from json import JSONDecoder
    decoder = JSONDecoder()
    idx = 0
    while idx < len(content):
        # 跳过空白字符
        while idx < len(content) and content[idx].isspace():
            idx += 1
        
        if idx >= len(content):
            break
            
        try:
            obj, end_idx = decoder.raw_decode(content[idx:])
            records.append(obj)
            idx += end_idx
        except json.JSONDecodeError as e:
            print(f"  ⚠ 解析错误位置 {idx}: {e}")
            break
    
    print(f"  ✓ 读取到 {len(records)} 条记录")
    
    if len(records) == 0:
        print(f"  ✗ 无数据，跳过")
        continue
    
    # 转换为DataFrame
    df = pd.DataFrame(records)
    
    # 重新编号iter：每两条记录是一组，连续编号为 0,1,2,3,4,5...
    # 原始数据模式：[iter=0, iter=3, iter=0, iter=3, iter=0, iter=3, ...]
    # 重新编号为：   [iter=0, iter=1, iter=2, iter=3, iter=4, iter=5, ...]
    if 'iter' in df.columns:
        print(f"  原始iter模式: {df['iter'].head(10).tolist()}")
        
        # 为每条记录分配新的连续iter编号
        df['iter'] = range(len(df))
        
        print(f"  重新编号后: 共 {len(df)} 个iter (0 到 {len(df)-1})")
    else:
        print(f"  ⚠ 没有找到iter列")
        # 如果没有iter列，创建一个
        df.insert(0, 'iter', range(len(df)))
    
    # 添加node_id列
    df.insert(0, 'node_id', node_id)
    
    # 保存单独的节点CSV
    output_file = base_dir / f"node_{node_id}_training_data.csv"
    df_without_node = df.drop('node_id', axis=1)  # 单独的CSV不需要node_id列
    # df_without_node.to_csv(output_file, index=False, encoding='utf-8', na_rep='NA')
    
    print(f"  ✓ 保存CSV: {output_file}")
    print(f"  ✓ CSV包含 {len(df)} 行, {len(df.columns)-1} 列")
    
    # 将数据添加到总列表中
    all_nodes_data.append(df)

print("\n" + "="*60)
print("生成合并的长格式CSV...")
print("="*60)

# 合并所有节点的数据
if all_nodes_data:
    combined_df = pd.concat(all_nodes_data, ignore_index=True)
    
    # 按node_id和iter排序
    combined_df = combined_df.sort_values(['node_id', 'iter']).reset_index(drop=True)
    
    # 保存长格式CSV
    long_format_file = base_dir / "all_nodes_long_format.csv"
    combined_df.to_csv(long_format_file, index=False, encoding='utf-8', na_rep='NA')
    
    print(f"\n✓ 长格式CSV已保存: {long_format_file}")
    print(f"  总行数: {len(combined_df)}")
    print(f"  总列数: {len(combined_df.columns)}")
    print(f"  节点数: {combined_df['node_id'].nunique()}")
    print(f"  数据形状: {len(combined_df)} 行 × {len(combined_df.columns)} 列")
    print(f"\n  列名: node_id, iter, 以及 {len(combined_df.columns)-2} 个指标列")
else:
    print("⚠ 没有数据可以合并")

print("\n" + "="*60)
print("全部完成！")
print("="*60)

# 列出生成的文件
csv_files = list(base_dir.glob("node_*_training_data.csv"))
print(f"\n生成的文件:")
print(f"  单独节点CSV: {len(csv_files)} 个")
for f in sorted(csv_files):
    print(f"    - {f.name}")
if all_nodes_data:
    print(f"  合并长格式CSV: all_nodes_long_format.csv")

