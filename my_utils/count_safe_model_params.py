from safetensors import safe_open
from collections import defaultdict

model_path = "/mnt/data-raid/yangguangzhao/t2/models/Qwen3-0.6B/model.safetensors"

# 模块统计
module_params = defaultdict(int)
total_params = 0

with safe_open(model_path, framework="pt", device="cpu") as f:
    for key in f.keys():
        num = f.get_tensor(key).numel()
        total_params += num

        # 简单分类规则（可根据实际模型结构调整）
        if "embed" in key.lower():
            module_params["Embedding"] += num
        elif "attn" in key.lower() or "attention" in key.lower():
            module_params["Attention"] += num
        elif "mlp" in key.lower() or "ffn" in key.lower():
            module_params["MLP"] += num
        elif "norm" in key.lower() or "ln" in key.lower():
            module_params["LayerNorm"] += num
        elif "head" in key.lower():
            module_params["LM Head"] += num
        else:
            module_params["Other"] += num

# 输出结果
print(f"Total parameters: {total_params:,}  (~{total_params/1e9:.2f}B)\n")
for mod, num in module_params.items():
    print(f"{mod:<12}: {num:,}  (~{num/1e6:.2f}M)")
