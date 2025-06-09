import os
import glob
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
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
        node_data = df[df["node"] == node_id]
        plt.plot(range(len(node_data)), node_data[metric], label=f"Node {node_id}")
    plt.title(metric)
    plt.xlabel("Step")
    plt.ylabel(metric)
    if i == 0:
        plt.legend()
plt.tight_layout()
plt.suptitle("Training Metrics", fontsize=18, y=1.02)
os.makedirs("results", exist_ok=True)
plt.savefig("results/metrics.png")
plt.close()

print("✅ saved to results/metrics.png")