"""
LoRA Policy for Transformer²
与 SVD Policy 对比实验用的 LoRA 实现
"""
import torch
import torch.nn as nn


class LoRAPolicy(nn.Module):
    """
    LoRA (Low-Rank Adaptation) Policy
    
    权重更新方式: W' = W + A @ B
    其中 A: (d_out, r), B: (r, d_in), r 是 LoRA rank
    
    可训练参数: 每层的 A 和 B 矩阵
    """
    
    def __init__(
        self, 
        base_params, 
        gpu, 
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.0,
        train_layers="mlp",
        init_scale=0.01,
        **kwargs
    ):
        super().__init__()
        
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.train_layers = train_layers
        self.scaling = lora_alpha / lora_rank
        self.gpu = gpu
        
        # 存储 LoRA 参数
        self.lora_A = {}  # 下投影矩阵
        self.lora_B = {}  # 上投影矩阵
        self.num_params = 0
        
        print(f"🔧 初始化 LoRA Policy: rank={lora_rank}, alpha={lora_alpha}, train_layers={train_layers}")
        
        # 为每个需要训练的层创建 LoRA 参数
        for k, v in base_params.items():
            if self._should_train_layer(k) and len(v.shape) == 2:
                d_out, d_in = v.shape
                
                # A 矩阵: (d_out, r) - 使用高斯初始化
                self.lora_A[k] = nn.Parameter(
                    torch.randn(d_out, lora_rank, device=gpu, dtype=torch.bfloat16) * init_scale,
                    requires_grad=True
                )
                
                # B 矩阵: (r, d_in) - 初始化为零，确保训练开始时 delta_W = 0
                self.lora_B[k] = nn.Parameter(
                    torch.zeros(lora_rank, d_in, device=gpu, dtype=torch.bfloat16),
                    requires_grad=True
                )
                
                self.num_params += self.lora_A[k].numel() + self.lora_B[k].numel()
                
        print(f"✅ LoRA 总参数量: {self.num_params:,} ({self.num_params/1e6:.2f}M)")
        
        # 转换为 ParameterDict 供 PyTorch 管理
        self.lora_A_module = nn.ParameterDict({
            k.replace(".", "_"): v for k, v in self.lora_A.items()
        })
        self.lora_B_module = nn.ParameterDict({
            k.replace(".", "_"): v for k, v in self.lora_B.items()
        })
        
        # 可训练参数列表（供优化器使用）
        self.trainable_params = list(self.lora_A.values()) + list(self.lora_B.values())
        
        # Dropout（可选）
        if lora_dropout > 0:
            self.dropout = nn.Dropout(lora_dropout)
        else:
            self.dropout = None
    
    def _should_train_layer(self, layer_name):
        """判断是否应该训练指定层（与 SVD Policy 保持一致）"""
        if "norm" in layer_name:
            return False
            
        if self.train_layers == "mlp":
            return "mlp" in layer_name
        elif self.train_layers == "self_attn":
            return "self_attn" in layer_name
        elif self.train_layers == "both":
            return "mlp" in layer_name or "self_attn" in layer_name
        else:
            return False
    
    def get_learnable_params(self, detach=False):
        """
        返回 LoRA 参数字典
        与 SVD Policy 接口保持一致
        """
        if detach:
            return {
                k: {"A": self.lora_A[k].detach(), "B": self.lora_B[k].detach()}
                for k in self.lora_A.keys()
            }
        else:
            return {
                k: {"A": self.lora_A[k], "B": self.lora_B[k]}
                for k in self.lora_A.keys()
            }
    
    def set_trainable_params_values(self, new_values):
        """
        设置 LoRA 参数值（用于 CEM 等优化算法）
        new_values: 字典，每个键对应一个层，值是包含 A 和 B 的字典
        """
        with torch.no_grad():
            for k in self.lora_A.keys():
                if k in new_values:
                    self.lora_A[k].data.copy_(new_values[k]["A"])
                    self.lora_B[k].data.copy_(new_values[k]["B"])
    
    def compute_delta_weight(self, layer_name):
        """
        计算 LoRA 的权重增量: delta_W = (A @ B) * scaling
        
        Args:
            layer_name: 层名称
            
        Returns:
            delta_W: 权重增量矩阵
        """
        A = self.lora_A[layer_name]
        B = self.lora_B[layer_name]
        
        # 应用 dropout（训练时）
        if self.dropout is not None and self.training:
            B = self.dropout(B)
        
        # delta_W = A @ B * scaling
        delta_W = (A @ B) * self.scaling
        
        return delta_W
    
    def record_state(self, metrics_to_log):
        """记录 LoRA 参数统计信息"""
        # 计算 A 和 B 矩阵的统计信息
        a_norms = [torch.norm(v).item() for v in self.lora_A.values()]
        b_norms = [torch.norm(v).item() for v in self.lora_B.values()]
        
        metrics_to_log.update(**{
            "lora/mean_A_norm": sum(a_norms) / len(a_norms) if a_norms else 0,
            "lora/mean_B_norm": sum(b_norms) / len(b_norms) if b_norms else 0,
            "lora/max_A_norm": max(a_norms) if a_norms else 0,
            "lora/max_B_norm": max(b_norms) if b_norms else 0,
        })


def compose_lora_params(policy, param_name, base_params, learnable_params):
    """
    组合 LoRA 参数生成新的权重
    类似于 utils.py 中的 compose_new_params，但用于 LoRA
    
    Args:
        policy: LoRAPolicy 实例
        param_name: 参数名称
        base_params: 基础模型参数
        learnable_params: LoRA 参数字典
        
    Returns:
        new_weight: 更新后的权重 W' = W + A @ B * scaling
    """
    base_weight = base_params[param_name]
    
    if param_name in learnable_params:
        lora_params = learnable_params[param_name]
        A = lora_params["A"]
        B = lora_params["B"]
        
        # 计算增量
        delta_W = (A @ B) * policy.scaling
        
        # 返回更新后的权重
        return base_weight + delta_W
    else:
        # 不训练的层直接返回基础权重
        return base_weight


def forward_lora(policy, model, base_params, learnable_params):
    """
    LoRA 的前向传播
    与 utils.py 中的 forward 函数接口一致
    
    Args:
        policy: LoRAPolicy 实例
        model: 模型
        base_params: 基础参数
        learnable_params: LoRA 参数
        
    Returns:
        new_params: 更新后的参数字典
    """
    new_params = {}
    
    with torch.no_grad():
        for k in base_params:
            if policy._should_train_layer(k):
                new_params[k] = compose_lora_params(
                    policy, k, base_params, learnable_params
                )
                model.get_parameter(k).copy_(new_params[k])
            else:
                new_params[k] = base_params[k]
    
    return new_params


def backward_lora(policy, model, base_params, learnable_params):
    """
    LoRA 的反向传播
    与 utils.py 中的 backward 函数接口一致
    
    Args:
        policy: LoRAPolicy 实例
        model: 模型
        base_params: 基础参数
        learnable_params: LoRA 参数
    """
    keys_to_backprop = [k for k in base_params if policy._should_train_layer(k)]
    last_key = keys_to_backprop[-1]
    
    # 对除最后一层外的所有层进行反向传播（保留计算图）
    for k in keys_to_backprop[:-1]:
        compose_lora_params(policy, k, base_params, learnable_params).backward(
            model.get_parameter(k).grad, retain_graph=True
        )
    
    # 最后一层释放计算图
    compose_lora_params(policy, last_key, base_params, learnable_params).backward(
        model.get_parameter(last_key).grad, retain_graph=False
    )

