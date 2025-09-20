#!/usr/bin/env python3
"""
AQuA-RAT数据集分类器
使用AI对AQuA-RAT数据集进行分类，并推送到HuggingFace Hub
"""

import os
import re
from typing import List, Dict, Any
import requests
from datasets import load_dataset, Dataset
from tqdm import tqdm
import json
import dotenv
dotenv.load_dotenv()

# 配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # 从环境变量获取
DATASET_NAME = "deepmind/aqua_rat"
TARGET_DATASET = "yangguangzhaojjj/aqua_rat_cls"
HF_TOKEN = os.getenv("HF_TOKEN", "")  # HuggingFace token

# 10个分类类别
CATEGORIES = {
    "1": "Probability & Statistics",
    "2": "Number Theory & Integer Properties", 
    "3": "Algebra & Equations",
    "4": "Ratios, Percentages & Proportion Applications",
    "5": "Averages & Data Analysis",
    "6": "Geometry & Measurement",
    "7": "Time, Speed & Distance",
    "8": "Work & Rate Problems",
    "9": "Sequences & Series",
    "10": "Word Problems & Logical Reasoning"
}

def load_classification_prompt():
    """加载分类提示词"""
    cls_file_path = "/mnt/data-raid/yangguangzhao/t2/my_utils/cls.txt"
    try:
        with open(cls_file_path, "r", encoding="utf-8") as f:
            system_msg = f.read()
    except FileNotFoundError:
        print(f"警告: 找不到 {cls_file_path}，使用默认分类")
        system_msg = """请将数学题分类为以下10个类别之一:
1. Probability & Statistics
2. Number Theory & Integer Properties
3. Algebra & Equations
4. Ratios, Percentages & Proportion Applications
5. Averages & Data Analysis
6. Geometry & Measurement
7. Time, Speed & Distance
8. Work & Rate Problems
9. Sequences & Series
10. Word Problems & Logical Reasoning"""
    
    system_msg += """

# 分析给定的数学题，将其分类到上述10个类别中的一个。

指令:
- 如果问题涉及多个类别，选择最主要的一个
- 在\\boxed{}中提供你的最终分类。例如: \\boxed{1}
- 只返回数字(1-10)

格式化你的回答如下:
分类: \\boxed{数字}
"""
    return system_msg

def extract_classification(text: str) -> str:
    """从AI回答中提取分类结果"""
    # 寻找 \\boxed{} 中的内容
    match = re.search(r"\\boxed\{([^}]*)\}", text)
    if match:
        result = match.group(1).strip()
        # 验证是否为有效的分类数字
        if result in [str(i) for i in range(1, 11)]:
            return result
    
    # 如果没有找到boxed，尝试寻找数字
    numbers = re.findall(r'\b([1-9]|10)\b', text)
    if numbers:
        return numbers[0]
    
    return "10"  # 默认分类为Word Problems & Logical Reasoning

def classify_with_ai(question: str, options: List[str], system_msg: str) -> str:
    """使用AI对单个问题进行分类"""
    if not OPENAI_API_KEY:
        print("警告: 未设置OPENAI_API_KEY，使用随机分类")
        import random
        return str(random.randint(1, 10))
    
    # 构建完整的问题文本
    full_question = question + "\nOptions:\n" + "\n".join(options)
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gpt-4o",  # 使用更便宜的模型
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": full_question}
        ],
        "temperature": 0,
        "max_tokens": 100
    }
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            ai_response = result['choices'][0]['message']['content']
            classification = extract_classification(ai_response)
            return classification
        else:
            print(f"API请求失败: {response.status_code}")
            return "10"  # 默认分类
            
    except Exception as e:
        print(f"分类时出错: {e}")
        return "10"  # 默认分类

def main():
    """主函数"""
    print("开始加载AQuA-RAT数据集...")
    
    # 加载数据集
    try:
        dataset = load_dataset(DATASET_NAME, "raw", split="test")
        print(f"成功加载数据集，共 {len(dataset)} 个样本")
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return
    
    # 加载分类提示词
    system_msg = load_classification_prompt()
    print("已加载分类提示词")
    
    # 对每个样本进行分类
    classified_data = []
    print("开始对样本进行分类...")
    
    for i, sample in enumerate(tqdm(dataset, desc="分类进度")):
        question = sample['question']
        options = sample['options']
        
        # 获取AI分类
        cls_result = classify_with_ai(question, options, system_msg)
        
        # 创建新的样本，添加cls字段
        new_sample = {
            **sample,  # 保留原有字段
            'cls': cls_result,
        }
        classified_data.append(new_sample)
        
        # 每100个样本打印一次进度
        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1}/{len(dataset)} 个样本")
    
    # 创建新的数据集
    new_dataset = Dataset.from_list(classified_data)
    print(f"分类完成，共 {len(new_dataset)} 个样本")
    
    # 打印分类统计
    cls_counts = {}
    for sample in classified_data:
        cls = sample['cls']
        cls_counts[cls] = cls_counts.get(cls, 0) + 1
    
    print("\n分类统计:")
    for cls_num, count in sorted(cls_counts.items()):
        cls_name = CATEGORIES.get(cls_num, "未知")
        print(f"  {cls_num}. {cls_name}: {count} 个样本")
    
    # 推送到HuggingFace Hub
    if HF_TOKEN:
        try:
            print(f"\n开始推送到 {TARGET_DATASET}...")
            new_dataset.push_to_hub(
                TARGET_DATASET, 
                split="test",
                token=HF_TOKEN
            )
            print("成功推送到HuggingFace Hub!")
        except Exception as e:
            print(f"推送失败: {e}")
            # 保存到本地作为备份
            local_path = "/mnt/data-raid/yangguangzhao/t2/aqua_rat_classified.json"
            with open(local_path, "w", encoding="utf-8") as f:
                json.dump(classified_data, f, ensure_ascii=False, indent=2)
            print(f"已保存到本地文件: {local_path}")
    else:
        print("警告: 未设置HF_TOKEN，无法推送到HuggingFace Hub")
        # 保存到本地
        local_path = "/mnt/data-raid/yangguangzhao/t2/aqua_rat_classified.json"
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(classified_data, f, ensure_ascii=False, indent=2)
        print(f"已保存到本地文件: {local_path}")

if __name__ == "__main__":
    main()
