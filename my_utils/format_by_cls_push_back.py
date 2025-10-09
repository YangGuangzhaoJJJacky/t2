from datasets import load_dataset, DatasetDict
import os

# 加载并合并所有原始数据
all_datasets = []
for i in range(10):
    dataset = load_dataset("yangguangzhaojjj/aqua_rat_cls_new", split=f"subset_{i}")
    all_datasets.append(dataset)

# 合并所有数据集
from datasets import concatenate_datasets
merged_dataset = concatenate_datasets(all_datasets)
print(f"合并后总行数: {len(merged_dataset)}")

# 按 cls 拆分
cls_values = sorted(set(merged_dataset["cls"]))
print("发现的 cls 标签:", cls_values)

# 创建 DatasetDict
subsets = {}
for cls_value in cls_values:
    subset = merged_dataset.filter(lambda x: x["cls"] == cls_value)
    subsets[f"cls_{cls_value}"] = subset
    print(f"subset: cls_{cls_value}, 行数 = {len(subset)}")

# 合并为 DatasetDict
dataset_dict = DatasetDict(subsets)

# push 回原 repo
dataset_dict.push_to_hub("yangguangzhaojjj/aqua_rat_cls_new")
