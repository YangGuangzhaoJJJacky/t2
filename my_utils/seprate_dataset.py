from datasets import load_dataset, DatasetDict
from huggingface_hub import HfApi, login
import os

# Step 1: 登录 Hugging Face（推荐用环境变量）
token = ""
login(token)

# Step 2: 加载并打乱数据
train_data = load_dataset("deepmind/aqua_rat", "raw", split="train")
train_data = train_data.shuffle(seed=42)

# Step 3: 平均分为10个子集
total = len(train_data)
subset_size = total // 11
splits = []

for i in range(10):
    splits.append(train_data.select(range(i * subset_size, (i + 1) * subset_size)))

# Step 4: 创建 DatasetDict 并命名子集
subset_dict = DatasetDict({f"subset_{i}": ds for i, ds in enumerate(splits)})

# Step 5: 上传数据集到你的 Hub 仓库
subset_dict.push_to_hub("yangguangzhaojjj/aqua_rat")