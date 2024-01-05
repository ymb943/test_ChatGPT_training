from typing import Optional, Union

import loralib as lora
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_approx_kl(log_probs: torch.Tensor,
                      log_probs_base: torch.Tensor,
                      action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        action_mask: Mask for actions.
    """
    
    # print('log_probs',log_probs)
    # print('log_probs_base',log_probs_base)
    # tensor([[-6.3748, -0.3850, -5.2702,  ..., -5.2411, -4.6299, -4.6946],
    #         [-2.4233, -0.1502, -0.0315,  ..., -5.3831, -4.7305, -4.7086],
    #         [-2.9520, -1.2308, -1.8419,  ..., -6.3464, -5.6861, -5.7310],
    #         ...,
    #         [-4.2715, -0.8264, -0.1740,  ..., -7.6705, -7.0685, -7.0264],
    #         [-2.4117, -0.5632, -1.3264,  ..., -0.1638, -0.3451, -0.3214],
    #         [-5.4037, -1.6125, -5.6595,  ..., -6.4581, -6.0071, -5.7032]], device='cuda:0')
    # tensor([[-6.3748, -0.3850, -5.2702,  ..., -5.2411, -4.6299, -4.6946],
    #         [-2.4233, -0.1502, -0.0315,  ..., -5.3831, -4.7305, -4.7086],
    #         [-2.9520, -1.2308, -1.8419,  ..., -6.3464, -5.6861, -5.7310],
    #         ...,
    #         [-4.2715, -0.8264, -0.1740,  ..., -7.6705, -7.0685, -7.0264],
    #         [-2.4117, -0.5632, -1.3264,  ..., -0.1638, -0.3451, -0.3214],
    #         [-5.4037, -1.6125, -5.6595,  ..., -6.4581, -6.0071, -5.7032]], device='cuda:0')
    
    log_ratio = log_probs - log_probs_base
    # print('log_ratio',log_ratio)
    # tensor([[0., 0., 0.,  ..., 0., 0., 0.],
    #         [0., 0., 0.,  ..., 0., 0., 0.],
    #         [0., 0., 0.,  ..., 0., 0., 0.],
    #         ...,
    #         [0., 0., 0.,  ..., 0., 0., 0.],
    #         [0., 0., 0.,  ..., 0., 0., 0.],
    #         [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0')

    approx_kl = (log_ratio.exp() - 1) - log_ratio
    # print('approx_kl',approx_kl)
    # print('approx_kl',approx_kl.shape)
    # tensor([[0., 0., 0.,  ..., 0., 0., 0.],
    #         [0., 0., 0.,  ..., 0., 0., 0.],
    #         [0., 0., 0.,  ..., 0., 0., 0.],
    #         ...,
    #         [0., 0., 0.,  ..., 0., 0., 0.],
    #         [0., 0., 0.,  ..., 0., 0., 0.],
    #         [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0')
    # approx_kl torch.Size([8, 107])

    # print('action_mask',action_mask)
    # tensor([[ True,  True, False,  ..., False, False, False],
    #         [ True,  True,  True,  ..., False, False, False],
    #         [ True,  True,  True,  ..., False, False, False],
    #         ...,
    #         [ True,  True,  True,  ..., False, False, False],
    #         [ True,  True,  True,  ...,  True,  True,  True],
    #         [ True,  True, False,  ..., False, False, False]], device='cuda:0')

    if action_mask is not None:
        approx_kl = masked_mean(approx_kl, action_mask, dim=1)
        # print('approx_kl',approx_kl)
        # print('approx_kl',approx_kl.shape)
        # approx_kl tensor([0., 0., 0.,  ..., 0., 0., 0.], device='cuda:0')
        # approx_kl torch.Size([8])
        return approx_kl
    
    approx_kl = approx_kl.mean(dim=1)
    # print('approx_kl',approx_kl)
    # print('approx_kl',approx_kl.shape)

    return approx_kl


def compute_reward(r: Union[torch.Tensor, float],
                   kl_coef: float,
                   log_probs: torch.Tensor,
                   log_probs_base: torch.Tensor,
                   action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    # print('kl_coef',kl_coef)
    # kl_coef 0.1

    if kl_coef <= 0.0:
        return r

    kl = compute_approx_kl(log_probs, log_probs_base, action_mask=action_mask)
    # print('kl',kl)
    # kl tensor([0., 0., 0.,  ..., 0., 0., 0.], device='cuda:0')

    reward = r - kl_coef * kl
    # print('reward',reward)
    # reward tensor([0.2177, 0.1005, 0.2401,  ..., 0.2930, 0.1571, 0.0807], device='cuda:0')
    
    return reward


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    # Select logit values by indices values (sequences)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    tensor = tensor * mask
    tensor = tensor.sum(dim=dim)
    mask_sum = mask.sum(dim=dim)
    mean = tensor / (mask_sum + 1e-8)
    return mean


def masked_normalize(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    tensor = tensor * mask
    mean = masked_mean(tensor, mask, dim=dim)
    mean_centered = tensor - mean
    var = masked_mean(mean_centered**2, mask, dim=dim)
    return mean_centered * var.clamp(min=eps).rsqrt()


def normalize(tensor: torch.Tensor, dim: int = 0, eps: float = 1e-8) -> torch.Tensor:
    mean = tensor.mean(dim)
    mean_centered = tensor - mean
    var = (mean_centered**2).mean(dim)
    norm = mean_centered * var.clamp(min=eps).rsqrt()
    return norm


def convert_to_lora(model: nn.Module,
                    input_size: int,
                    output_size: int,
                    lora_rank: int = 16,
                    lora_alpha: int = 1,
                    lora_dropout: float = 0.,
                    fan_in_fan_out: bool = False,
                    merge_weights: bool = True):
    if lora_rank > min(input_size, output_size):
        raise ValueError(f"LoRA rank {lora_rank} must be less or equal than {min(input_size, output_size)}")

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._modules[name] = lora.Linear(input_size,
                                                output_size,
                                                r=lora_rank,
                                                lora_alpha=lora_alpha,
                                                lora_dropout=lora_dropout,
                                                fan_in_fan_out=fan_in_fan_out,
                                                merge_weights=merge_weights)
