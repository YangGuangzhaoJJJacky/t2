#!/usr/bin/env python3
"""
将本地分类结果推送到HuggingFace Hub
"""

import os
import json
import dotenv
dotenv.load_dotenv()
from datasets import Dataset

def main():
    # 配置
    TARGET_DATASET = "yangguangzhaojjj/aqua_rat_cls"
    LOCAL_FILE = "/mnt/data-raid/yangguangzhao/t2/my_utils/aqua_rat_test_classified.json"
    HF_TOKEN = os.getenv("HF_TOKEN", "")
    
    if not HF_TOKEN:
        print("错误: 未设置HF_TOKEN环境变量")
        print("请设置: export HF_TOKEN='your_huggingface_token_here'")
        return
    
    # 检查本地文件是否存在
    if not os.path.exists(LOCAL_FILE):
        print(f"错误: 找不到本地文件 {LOCAL_FILE}")
        return
    
    print(f"加载本地分类结果: {LOCAL_FILE}")
    
    # 加载本地数据
    with open(LOCAL_FILE, "r", encoding="utf-8") as f:
        classified_data = json.load(f)
    
    print(f"共加载 {len(classified_data)} 个样本")
    for sample in classified_data:
        try:
            sample['cls'] = int(sample['cls'])
        except (ValueError, TypeError):
            sample['cls'] = 10  # 默认值
    # 创建数据集
    dataset = Dataset.from_list(classified_data)
    
    # 推送到HuggingFace Hub
    try:
        print(f"\n开始推送到 {TARGET_DATASET} (test split)...")
        dataset.push_to_hub(
            TARGET_DATASET, 
            split="test",
            token=HF_TOKEN
        )
        print("✅ 成功推送到HuggingFace Hub!")
        print(f"数据集链接: https://huggingface.co/datasets/{TARGET_DATASET}")
        
    except Exception as e:
        print(f"❌ 推送失败: {e}")
        return

if __name__ == "__main__":
    main()
