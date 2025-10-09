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
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading
dotenv.load_dotenv()
# split_name = "subset_0"
# 配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # 从环境变量获取
DATASET_NAME = "yangguangzhaojjj/aqua_rat_raw"
TARGET_DATASET = "yangguangzhaojjj/aqua_rat_test"
HF_TOKEN = os.getenv("HF_TOKEN", "")

# 并发配置
MAX_CONCURRENT_REQUESTS = 1   # 串行处理，避免速率限制
RATE_LIMIT_DELAY = 1        # 请求间隔增加到2秒
REQUEST_TIMEOUT = 30          # 请求超时时间（秒）
MAX_RETRIES = 5               # 最大重试次数

# 全局锁用于速率限制
rate_limit_lock = threading.Lock()

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

def classify_with_ai_rate_limited(question: str, options: List[str], system_msg: str) -> str:
    """使用AI对单个问题进行分类，带速率限制"""
    # 更保守的速率限制
    with rate_limit_lock:
        time.sleep(RATE_LIMIT_DELAY)
    
    return classify_with_ai(question, options, system_msg)

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
        "model": "gpt-4o",  # 使用正确的模型名
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": full_question}
        ],
        "temperature": 0,
        "max_tokens": 100
    }
    
    # 重试机制
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=REQUEST_TIMEOUT
            )
            break  # 成功则跳出重试循环
        except requests.exceptions.RequestException as e:
            if attempt == MAX_RETRIES:
                print(f"API请求失败，已重试{MAX_RETRIES}次: {e}")
                return "10"
            print(f"API请求失败，正在重试 ({attempt + 1}/{MAX_RETRIES}): {e}")
            time.sleep(1)  # 重试前等待1秒
    
    try:
        
        if response.status_code == 200:
            result = response.json()
            ai_response = result['choices'][0]['message']['content']
            classification = extract_classification(ai_response)
            return classification
        else:
            print(f"API请求失败: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"错误详情: {error_detail}")
                
                # 如果是速率限制错误，等待更长时间
                if response.status_code == 429:
                    if 'retry-after' in response.headers:
                        retry_after = int(response.headers['retry-after'])
                        print(f"速率限制，等待 {retry_after} 秒...")
                        time.sleep(retry_after)
                    else:
                        # 从错误消息中提取等待时间
                        error_msg = error_detail.get('error', {}).get('message', '')
                        if 'Please try again in' in error_msg:
                            import re
                            match = re.search(r'Please try again in ([\d.]+)s', error_msg)
                            if match:
                                wait_time = float(match.group(1))
                                print(f"速率限制，等待 {wait_time} 秒...")
                                time.sleep(wait_time + 0.5)  # 额外等待0.5秒
                            else:
                                print("速率限制，等待 5 秒...")
                                time.sleep(5)
                        else:
                            print("速率限制，等待 5 秒...")
                            time.sleep(5)
            except:
                print(f"响应内容: {response.text}")
            return "10"  # 默认分类
            
    except Exception as e:
        print(f"分类时出错: {e}")
        return "10"  # 默认分类

def classify_batch_concurrent(dataset, system_msg: str, max_workers: int = MAX_CONCURRENT_REQUESTS) -> List[Dict[str, Any]]:
    """使用并发方式批量分类"""
    results = []
    
    def classify_single_item(item_with_index):
        """分类单个项目的包装函数"""
        index, item = item_with_index
        try:
            classification = classify_with_ai_rate_limited(
                item['question'], 
                item['options'], 
                system_msg
            )
            return {
                'index': index,
                'question': item['question'],
                'options': item['options'],
                'correct': item['correct'],
                'rationale': item['rationale'],
                'classification': classification
            }
        except Exception as e:
            print(f"分类第{index}个样本时出错: {e}")
            return {
                'index': index,
                'question': item['question'],
                'options': item['options'],
                'correct': item['correct'],
                'rationale': item['rationale'],
                'classification': "10"  # 默认分类
            }
    
    # 创建带索引的数据
    indexed_data = list(enumerate(dataset))
    
    # 使用线程池并发处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_index = {
            executor.submit(classify_single_item, item): item[0] 
            for item in indexed_data
        }
        
        # 使用tqdm显示进度
        with tqdm(total=len(dataset), desc="分类进度") as pbar:
            for future in as_completed(future_to_index):
                try:
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
                except Exception as e:
                    index = future_to_index[future]
                    print(f"处理第{index}个样本时出错: {e}")
                    pbar.update(1)
    
    # 按索引排序结果
    results.sort(key=lambda x: x['index'])
    
    # 移除索引字段
    for result in results:
        del result['index']
    
    return results

def main(split_name):
    """主函数"""
    print("开始加载AQuA-RAT数据集...")
    
    # 加载数据集
    try:
        dataset = load_dataset(DATASET_NAME, split=split_name)
        print(f"成功加载数据集，共 {len(dataset)} 个样本")
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return
    
    # 加载分类提示词
    system_msg = load_classification_prompt()
    print("已加载分类提示词")
    
    # 使用并发方式对样本进行分类
    print(f"开始并发分类 (最大并发数: {MAX_CONCURRENT_REQUESTS})...")
    start_time = time.time()
    
    classified_data = classify_batch_concurrent(dataset, system_msg)
    
    end_time = time.time()
    print(f"分类完成，耗时: {end_time - start_time:.2f}秒")
    
    # 将并发结果转换为数据集格式
    dataset_data = []
    for item in classified_data:
        dataset_data.append({
            'question': item['question'],
            'options': item['options'],
            'correct': item['correct'],
            'rationale': item['rationale'],
            'cls': item['classification']  # 添加cls字段
        })
    
    # 创建新的数据集
    new_dataset = Dataset.from_list(dataset_data)
    print(f"数据集创建完成，共 {len(new_dataset)} 个样本")
    
    # 打印分类统计
    cls_counts = {}
    for sample in dataset_data:
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
                split=split_name,
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
    # for split_name in range(10):
    main("test")
