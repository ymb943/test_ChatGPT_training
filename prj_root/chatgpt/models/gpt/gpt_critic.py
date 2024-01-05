from typing import Optional

import torch.nn as nn
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Model

from ..base import Critic

import torch
import numpy as np
torch.manual_seed(42)
np.random.seed(42)


class GPTCritic(Critic):
    """
    GPT Critic model.

    Args:
        pretrained (str): Pretrained model name or path.
        config (GPT2Config): Model config.
        checkpoint (bool): Enable gradient checkpointing.
        lora_rank (int): Rank of the LO-RA decomposition.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(self,
                 pretrained: Optional[str] = None,
                 config: Optional[GPT2Config] = None,
                 checkpoint: bool = False,
                 lora_rank: int = 0,
                 lora_train_bias: str = 'none') -> None:
        
        print('pretrained',pretrained)
        print('config',config)
        # pretrained ./output_2_RM
        # config None

        if pretrained is not None:
            model = GPT2Model.from_pretrained(pretrained)
        elif config is not None:
            model = GPT2Model(config)
        else:
            model = GPT2Model(GPT2Config())

        # print('model',model)
        # model GPT2Model(
        #   (wte): Embedding(51201, 768)
        #   (wpe): Embedding(1024, 768)
        #   (drop): Dropout(p=0.1, inplace=False)
        #   (h): ModuleList(
        #     (0): GPT2Block(
        #       (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        #    ...
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
        
        
        
        if checkpoint:
            model.gradient_checkpointing_enable()
        
        # print('model.config.n_embd',model.config.n_embd)
        # model.config.n_embd 768
        value_head = nn.Linear(model.config.n_embd, 1)
        # print('value_head',value_head)
        # value_head Linear(in_features=768, out_features=1, bias=True)
        
        super().__init__(model, value_head, lora_rank, lora_train_bias)
