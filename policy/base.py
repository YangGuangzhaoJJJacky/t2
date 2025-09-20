import torch
import torch.nn as nn


def get_soft_mask(n, fraction):
    indices = torch.linspace(0, n - 1, n, dtype=torch.bfloat16) + 1
    scaled_indices = indices.to(fraction.device) - fraction * n
    result = torch.clamp(scaled_indices, 0, 1)
    return 1.0 - result


class Policy(nn.Module):
    def __init__(self, base_params, gpu, init_val, max_mult=1, train_layers="mlp", **kwargs):
        # Create learnable parameters.
        super().__init__()
        self.learnable_params = {}
        self.num_params = 0
        self.max_mult = max_mult
        # 🔥 控制训练哪些层: "mlp", "self_attn", "both"
        self.train_layers = train_layers
        for k, v in base_params.items():
            # each param initialized with small gaussian noise
            if self._should_train_layer(k):
                self.learnable_params[k] = torch.nn.Parameter(
                    data=(
                        torch.randn(
                            min(v.shape),
                            device=gpu,
                            dtype=torch.bfloat16,
                        )
                        * 0.01
                        + init_val
                    ),
                    requires_grad=True,
                )
                self.num_params += self.learnable_params[k].numel()
        print(f"#params={self.num_params}")
        self.learnable_params_list = list(self.learnable_params.values())
        self.trainable_params = self.learnable_params_list
        self.learnable_params_module_list = nn.ParameterList(self.learnable_params_list)

    def _should_train_layer(self, layer_name):
        """判断是否应该训练指定层"""
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
        return self.learnable_params

    def set_trainable_params_values(self, new_values):
        with torch.no_grad():
            for p, v in zip(self.trainable_params, new_values):
                p.data.copy_(v)

    def get_mask(self, p):
        return torch.sigmoid(p).to(torch.bfloat16) * self.max_mult

    def record_state(self, metrics_to_log):
        pass
