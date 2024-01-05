from typing import Optional

import torch
import torch.nn as nn

from ..lora import LoRAModule
from ..utils import masked_mean


class Critic(LoRAModule):
    """
    Critic model base class.

    Args:
        model (nn.Module): Critic model.
        value_head (nn.Module): Value head to get value.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(
        self,
        model: nn.Module,
        value_head: nn.Module,
        lora_rank: int = 0,
        lora_train_bias: str = 'none',
        use_action_mask: bool = False) -> None:

        super().__init__(lora_rank=lora_rank, lora_train_bias=lora_train_bias)
        self.model = model
        self.value_head = value_head
        self.use_action_mask = use_action_mask
        self.convert_to_lora()

    def forward(self,
                sequences: torch.LongTensor,
                action_mask: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # print('self.model',self.model)
        # self.model GPT2Model(
        #   (wte): Embedding(51201, 768)
        #   (wpe): Embedding(1024, 768)
        #   (drop): Dropout(p=0.1, inplace=False)
        #   (h): ModuleList(
        #     (0): GPT2Block(
        #       (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        #       (attn): GPT2Attention(
        #         (c_attn): Conv1D()
        #         (c_proj): Conv1D()
        #         (attn_dropout): Dropout(p=0.1, inplace=False)
        #         (resid_dropout): Dropout(p=0.1, inplace=False)
        #       )
        #       (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        #       (mlp): GPT2MLP(
        #         (c_fc): Conv1D()
        #         (c_proj): Conv1D()
        #         (act): NewGELUActivation()
        #         (dropout): Dropout(p=0.1, inplace=False)
        #       )
        #     )
        #     ...
        #       (mlp): GPT2MLP(
        #         (c_fc): Conv1D()
        #         (c_proj): Conv1D()
        #         (act): NewGELUActivation()
        #         (dropout): Dropout(p=0.1, inplace=False)
        #       )
        #     )
        #   )
        #   (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        # )

        outputs = self.model(sequences, attention_mask=attention_mask)

        # Each predicted token is represented by 768 dim vectors
        last_hidden_states = outputs['last_hidden_state']
        # print('last_hidden_states',last_hidden_states)
        # print('last_hidden_states',last_hidden_states.shape)
        # tensor([[[ 6.2314e-01,  4.6855e-01, -1.5072e+00,  ..., -7.5447e-01, 1.7254e+00, -2.8906e-01],
        #           ...,
        #           [ 2.8645e-01, -1.3650e+00, -1.1381e+00,  ..., -2.3932e+00, -1.2766e-02,  5.3934e-02]],
        #         [[ 7.5642e-01,  6.7043e-01, -1.7850e+00,  ..., -6.0732e-01, 1.5661e+00,  4.8169e-02],
        #         ...
        #         [[ 9.2055e-01,  6.7281e-01, -1.8027e+00,  ..., -5.2436e-01, 1.6301e+00, -2.2599e-01],
        #           ...,
        #           [ 7.0277e-01, -1.1047e+00, -1.0319e+00,  ..., -2.1420e+00, 9.0774e-03, -1.6356e-01]]], device='cuda:0')
        # last_hidden_states torch.Size([8, 128, 768])

        # print('self.value_head',self.value_head)
        # self.value_head Linear(in_features=768, out_features=1, bias=True)
        # Each predicted token is represented by scalar value
        values = self.value_head(last_hidden_states).squeeze(-1)
        # print('values',values)
        # print('values',values.shape)
        # values tensor([[ 0.6288,  0.5043,  0.7339,  ...,  0.1426,  0.1293,  0.1465],
        #                [ 0.5075,  0.4783,  0.9470,  ...,  0.0561,  0.0444,  0.0649],
        #                [ 0.4830,  0.2615,  0.5732,  ...,  0.1868,  0.1567,  0.1871],
        #                ...,
        #                [ 0.6212,  0.7335,  1.0321,  ...,  0.2336,  0.2247,  0.2428],
        #                [ 0.5768,  0.5844,  0.4518,  ...,  0.3814, -0.0205, -0.2165],
        #                [ 0.6578,  0.6196,  0.6709,  ...,  0.0302,  0.0046,  0.0327]], device='cuda:0')
        # values torch.Size([8, 128])

        # ================================================================================
        # print('action_mask',action_mask)
        # print('self.use_action_mask',self.use_action_mask)
        # action_mask tensor([[ True,  True, False,  ..., False, False, False],
        #         [ True,  True,  True,  ..., False, False, False],
        #         [ True,  True,  True,  ..., False, False, False],
        #         ...,
        #         [ True,  True,  True,  ..., False, False, False],
        #         [ True,  True,  True,  ...,  True,  True,  True],
        #         [ True,  True, False,  ..., False, False, False]], device='cuda:0')
        # self.use_action_mask False

        if action_mask is not None and self.use_action_mask:
            num_actions = action_mask.size(1)
            # print('num_actions',num_actions)

            prompt_mask = attention_mask[:, :-num_actions]
            # print('prompt_mask',prompt_mask)

            values = values[:, :-num_actions]
            # print('values',values)

            value = masked_mean(values, prompt_mask, dim=1)
            # print('value',value)

            return value

        # ================================================================================
        values = values[:, :-1]
        # Average value of 127 predicted tokens
        value = values.mean(dim=1)
        # print('values',values)
        # print('value',value)
        # print('values',values.shape)
        # print('value',value.shape)
        # tensor([[ 0.6288,  0.5043,  0.7339,  ...,  0.1241,  0.1426,  0.1293],
        #         [ 0.5075,  0.4783,  0.9470,  ...,  0.0357,  0.0561,  0.0444],
        #         [ 0.4830,  0.2615,  0.5732,  ...,  0.1762,  0.1868,  0.1567],
        #         ...,
        #         [ 0.6212,  0.7335,  1.0321,  ...,  0.2169,  0.2336,  0.2247],
        #         [ 0.5768,  0.5844,  0.4518,  ...,  0.3282,  0.3814, -0.0205],
        #         [ 0.6578,  0.6196,  0.6709,  ...,  0.0089,  0.0302,  0.0046]], device='cuda:0')
        # value tensor([0.2177, 0.1005, 0.2401,  ..., 0.2930, 0.1571, 0.0807], device='cuda:0')
        # values torch.Size([8, 127])
        # value torch.Size([8])

        return value
