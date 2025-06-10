import os
import glob
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def extract_json_objects(text):
    return re.findall(r'\{[^{}]+\}', text)

records = []
for node in range(10):
    for path in glob.glob(f"results/{node}/**/*.json", recursive=True):
        try:
            with open(path, "r") as f:
                content = f.read()
                json_blocks = extract_json_objects(content)
            for jb in json_blocks:
                try:
                    jb_fixed = re.sub(r'([,{])\s*(\w+)\s*:', r'\1 "\2":', jb)  # 给属性名加双引号
                    data = json.loads(jb_fixed)
                    data["node"] = node
                    data["path"] = str(Path(path).resolve())
                    records.append(data)
                except Exception as e2:
                    print(f"⚠️ Skip block in {path}: {e2}")
        except Exception as e:
            print(f"❌ Error reading {path}: {e}")

if not records:
    raise ValueError("❌ No valid JSON logs found.")

# 不进行排序，保持写入时的原始顺序
df = pd.DataFrame(records)
print(df)

# exit()  # 取消注释可用于调试

metrics = [
    "train_acc", "test_acc", "valid_acc", "best_val_acc", "test_at_best_val",
    "rewards/mean", "rewards/std", "pg", "loss", "kl_div"
]

plt.figure(figsize=(18, 12))

for i, metric in enumerate(metrics):
    plt.subplot(3, 4, i + 1)
    
    for node_id in sorted(df["node"].unique()):
        node_data = df[df["node"] == node_id].copy()
        
        # 获取metric列的值，处理NaN
        metric_values = node_data[metric]
        
        # 创建有效数据点的掩码（非NaN）
        valid_mask = ~pd.isna(metric_values)
        
        if valid_mask.any():  # 如果有有效数据点
            # 获取有效数据点的索引和值
            valid_indices = np.where(valid_mask)[0]
            valid_values = metric_values[valid_mask]
            
            # 绘制线图（跳过NaN点）
            plt.plot(valid_indices, valid_values, label=f"Node {node_id}", marker='o', markersize=3)
            
            # 检查iter=0的点并添加特殊标记
            if 'iter' in node_data.columns:
                iter_0_mask = (node_data['iter'] == 0) & valid_mask
                if iter_0_mask.any():
                    iter_0_indices = np.where(iter_0_mask)[0]
                    iter_0_values = metric_values[iter_0_mask]
                    # 使用星形标记标注iter=0的点
                    plt.scatter(iter_0_indices, iter_0_values, 
                              marker='o', s=10, c='red', 
                              edgecolors='black', linewidth=1,
                              zorder=5)  # 确保标记在线条上方
    
    plt.title(metric)
    plt.xlabel("Step")
    plt.ylabel(metric)
    plt.grid(True, alpha=0.3)
    
    if i == 0:
        plt.legend()

plt.tight_layout()
plt.suptitle("Training Metrics (Red stars mark iter=0 points)", fontsize=18, y=1.02)

os.makedirs("results", exist_ok=True)
plt.savefig("results/metrics.png", dpi=300, bbox_inches='tight')
plt.close()

print("✅ saved to results/metrics.png")