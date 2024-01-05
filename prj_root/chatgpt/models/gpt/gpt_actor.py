from typing import Optional

from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

from ..base import Actor


class GPTActor(Actor):
    """
    GPT Actor model.

    Args:
        pretrained (str): Pretrained model name or path.
        config (GPT2Config): Model config.
        checkpoint (bool): Enable gradient checkpointing.
        lora_rank (int): Rank of the LoRa layer.
        lora_train_bias (str): Bias training strategy for the LoRa layer.
    """

    def __init__(self,
                 pretrained: Optional[str] = None,
                 config: Optional[GPT2Config] = None,
                 checkpoint: bool = False,
                 lora_rank: int = 0,
                 lora_train_bias: str = 'none') -> None:

        # print('pretrained',pretrained)
        # print('config',config)
        # pretrained ./output_1_SFT
        # config None
        
        if pretrained is not None:
            # print('GPT2LMHeadModel',GPT2LMHeadModel)
            # GPT2LMHeadModel <class 'transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel'>
            model = GPT2LMHeadModel.from_pretrained(pretrained)
            # print('model',model)
            # model GPT2LMHeadModel(
            #   (transformer): GPT2Model(
            #     (wte): Embedding(51200, 768)
            #     (wpe): Embedding(1024, 768)
            #     (drop): Dropout(p=0.1, inplace=False)
            #     (h): ModuleList(
            #       (0): GPT2Block(
            #      ...
            #     (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            #   )
            #   (lm_head): Linear(in_features=768, out_features=51200, bias=False)
            # )
        elif config is not None:
            model = GPT2LMHeadModel(config)
        else:
            model = GPT2LMHeadModel(GPT2Config())
        
        if checkpoint:
            model.gradient_checkpointing_enable()
        super().__init__(model, lora_rank, lora_train_bias)


