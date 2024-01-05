from typing import Optional

import torch
import torch.nn as nn

from ..lora import LoRAModule


class RewardModel(LoRAModule):
    """
    Reward model base class.

    Args:
        model (nn.Module): Reward model.
        value_head (nn.Module): Value head to get reward score.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(self,
                 model: nn.Module,
                 value_head: Optional[nn.Module] = None,
                 lora_rank: int = 0,
                 lora_train_bias: str = 'none') -> None:

        super().__init__(lora_rank=lora_rank, lora_train_bias=lora_train_bias)
        self.model = model
        self.convert_to_lora()

        if value_head is not None:
            if value_head.out_features != 1:
                raise ValueError("The value head of reward model's output dim should be 1!")
            self.value_head = value_head
        else:
            self.value_head = nn.Linear(model.config.n_embd, 1)

    def forward(self, sequences: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # print('self.model',self.model)
        # self.model GPT2Model(
        #   (wte): Embedding(51201, 768)
        #   (wpe): Embedding(1024, 768)
        #   (drop): Dropout(p=0.1, inplace=False)
        #   (h): ModuleList(
        #     ...
        #       (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
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
        
        last_hidden_states = outputs['last_hidden_state']
        # print('last_hidden_states',last_hidden_states)
        # print('last_hidden_states',last_hidden_states.shape)
        # tensor([[[ 6.2314e-01,  4.6855e-01, -1.5072e+00,  ..., -7.5447e-01, 1.7254e+00, -2.8906e-01],
        #         ...,
        #         [ 2.8645e-01, -1.3650e+00, -1.1381e+00,  ..., -2.3932e+00, -1.2766e-02,  5.3934e-02]],
        #         ...
        #         [[ 9.2055e-01,  6.7281e-01, -1.8027e+00,  ..., -5.2436e-01, 1.6301e+00, -2.2599e-01],
        #         ...,
        #         [ 7.0277e-01, -1.1047e+00, -1.0319e+00,  ..., -2.1420e+00, 9.0774e-03, -1.6356e-01]]], device='cuda:0')
        # last_hidden_states torch.Size([8, 128, 768])
        
        values = self.value_head(last_hidden_states)[:, :-1]
        # print('values',values)
        # print('values',values.shape)
        # values tensor([[[ 0.6288],
        #         [ 0.5043],
        #         [ 0.7339],
        #         ...,
        #         [ 0.1241],
        #         [ 0.1426],
        #         [ 0.1293]],
        #         ...,
        #         [[ 0.6578],
        #         [ 0.6196],
        #         [ 0.6709],
        #         ...,
        #         [ 0.0089],
        #         [ 0.0302],
        #         [ 0.0046]]], device='cuda:0')
        # values torch.Size([8, 127, 1])
        
        value = values.mean(dim=1).squeeze(1)    # ensure shape is (B)
        # print('value',value)
        # print('value',value.shape)
        # value tensor([0.2177, 0.1005, 0.2401,  ..., 0.2930, 0.1571, 0.0807], device='cuda:0')
        # value torch.Size([8])
        
        return value
