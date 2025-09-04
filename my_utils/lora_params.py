import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# 1. 加载基础模型
model_name = "/mnt/data-raid/yangguangzhao/t2/models/Qwen3-0.6B"   # 这里用 Qwen 小模型举例
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

# 2. 配置 LoRA
lora_config = LoraConfig(
    r=8,                        # LoRA 秩
    lora_alpha=16,              # 缩放系数
    target_modules=[
                    "up_proj", "down_proj", "gate_proj"],  # 指定要插 LoRA 的层
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 3. 给模型注入 LoRA
model = get_peft_model(model, lora_config)

# 4. 打印 LoRA 参数量
model.print_trainable_parameters()