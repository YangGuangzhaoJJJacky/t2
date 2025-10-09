from datasets import load_dataset, DatasetDict, concatenate_datasets
from huggingface_hub import login
import logging
import random
from collections import defaultdict

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """将按类别分的数据集重新均匀分配到10个subset中"""
    
    # 确保已登录Hugging Face
    try:
        login()
        logger.info("已登录Hugging Face Hub")
    except Exception as e:
        logger.warning(f"登录失败: {e}")
    
    # 加载所有现有的cls数据集
    logger.info("开始加载所有cls数据集...")
    all_datasets = []
    
    for i in range(1, 11):  # cls_1 到 cls_10
        try:
            logger.info(f"加载 cls_{i}...")
            dataset = load_dataset("yangguangzhaojjj/aqua_rat_cls_new", split=f"cls_{i}")
            all_datasets.append(dataset)
            logger.info(f"cls_{i} 加载完成，行数: {len(dataset)}")
        except Exception as e:
            logger.error(f"加载 cls_{i} 失败: {e}")
            continue
    
    if not all_datasets:
        logger.error("没有成功加载任何数据集")
        return
    
    # 合并所有数据集
    logger.info("开始合并数据集...")
    merged_dataset = concatenate_datasets(all_datasets)
    logger.info(f"合并完成，总行数: {len(merged_dataset)}")
    
    # 检查数据集结构
    logger.info(f"数据集列名: {merged_dataset.column_names}")
    
    # 按cls类别分组
    if "cls" not in merged_dataset.column_names:
        logger.error("数据集中没有找到 'cls' 列")
        return
    
    cls_groups = defaultdict(list)
    for idx, item in enumerate(merged_dataset):
        cls_value = item["cls"]
        cls_groups[cls_value].append(idx)
    
    logger.info(f"发现的cls类别: {sorted(cls_groups.keys())}")
    for cls_val, indices in cls_groups.items():
        logger.info(f"cls_{cls_val}: {len(indices)} 个样本")
    
    # 计算每个subset应该包含的样本数
    total_samples = len(merged_dataset)
    samples_per_subset = total_samples // 10
    logger.info(f"每个subset目标样本数: {samples_per_subset}")
    
    # 为每个subset分配样本
    subsets = []
    subset_samples = [[] for _ in range(10)]
    
    # 按类别循环分配，确保每个subset都包含所有类别的样本
    for cls_val, indices in cls_groups.items():
        # 随机打乱该类别的样本
        random.shuffle(indices)
        
        # 将该类别的样本均匀分配到10个subset中
        samples_per_subset_per_cls = len(indices) // 10
        remainder = len(indices) % 10
        
        start_idx = 0
        for subset_idx in range(10):
            # 计算这个subset应该分配多少个该类别的样本
            count = samples_per_subset_per_cls
            if subset_idx < remainder:  # 余数样本分配给前几个subset
                count += 1
            
            # 分配样本
            end_idx = start_idx + count
            subset_samples[subset_idx].extend(indices[start_idx:end_idx])
            start_idx = end_idx
    
    # 创建新的数据集
    logger.info("开始创建新的subset...")
    new_subsets = {}
    
    for i in range(10):
        # 随机打乱该subset的样本顺序
        random.shuffle(subset_samples[i])
        
        # 创建subset数据集
        subset_indices = subset_samples[i]
        subset_data = merged_dataset.select(subset_indices)
        
        new_subsets[f"subset_{i}"] = subset_data
        logger.info(f"subset_{i}: {len(subset_data)} 个样本")
        
        # 统计该subset中各类别的分布
        cls_counts = defaultdict(int)
        for item in subset_data:
            cls_counts[item["cls"]] += 1
        
        logger.info(f"  subset_{i} 类别分布: {dict(cls_counts)}")
    
    # 创建 DatasetDict
    dataset_dict = DatasetDict(new_subsets)
    
    # 验证数据完整性
    total_redistributed = sum(len(subset) for subset in new_subsets.values())
    logger.info(f"重新分配后总样本数: {total_redistributed}")
    
    if total_redistributed != total_samples:
        logger.warning(f"样本数不匹配！原始: {total_samples}, 重新分配后: {total_redistributed}")
    
    # 推送到Hub
    logger.info("开始推送到Hugging Face Hub...")
    try:
        dataset_dict.push_to_hub("yangguangzhaojjj/aqua_rat_random_new", private=True)
        logger.info("推送成功！")
    except Exception as e:
        logger.error(f"推送失败: {e}")
        return
    
    # 输出最终统计信息
    logger.info("=" * 60)
    logger.info("重新分配完成！最终统计:")
    logger.info("=" * 60)
    
    for subset_name, subset_data in dataset_dict.items():
        logger.info(f"{subset_name}: {len(subset_data)} 个样本")
        
        # 统计该subset中各类别的分布
        cls_counts = defaultdict(int)
        for item in subset_data:
            cls_counts[item["cls"]] += 1
        
        logger.info(f"  类别分布: {dict(sorted(cls_counts.items()))}")
    
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
