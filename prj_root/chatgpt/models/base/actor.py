from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..generation import generate
from ..lora import LoRAModule
from ..utils import log_probs_from_logits


class Actor(LoRAModule):
    """
    Actor model base class.

    Args:
        model (nn.Module): Actor Model.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(self, model: nn.Module, lora_rank: int = 0, lora_train_bias: str = 'none') -> None:
        super().__init__(lora_rank=lora_rank, lora_train_bias=lora_train_bias)
        self.model = model
        self.convert_to_lora()

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        return_action_mask: bool = True,
        **kwargs) -> Union[Tuple[torch.LongTensor, torch.LongTensor], Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor]]:
        
        # print('generate',generate)
        # print('generate',type(generate))
        # generate <function generate at 0x7fd81d6833a0>
        # generate <class 'function'>

        # print('input_ids',input_ids.shape)
        # print('input_ids',input_ids)
        # input_ids torch.Size([8, 21])
        # tensor([[ 9054,  9867, 11474,  8022,  8084,   406,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,   1],
        #         [40053, 12789,  8615,  9341,  8084,   406,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,   1],
        #         [19785, 12380,  8135, 13119, 13456,  9341,  8084,   406,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,   1],
        #         [15215,  9913, 26999, 11886, 15706, 16860, 13833, 16608,  9207, 10805, 10560,  9050,  8645,  8199,  7397, 12749,  9559, 26526,  9099, 11862, 406],
        #         [10165,  9059,  7048,  7162, 11064, 46651,  9625,  8017,  8006,  8084,   406,     1,     1,     1,     1,     1,     1,     1,     1,     1,   1],
        #         [ 9267, 12817,  7192,  8704, 10070,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,   1],
        #         [34790,  9448,   739,  7920,  9564,  9421,   406,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,   1],
        #         [ 9435,  7487,  9273, 15723, 10006, 15624, 11435,  9755,  8645,  7374, 32974,  9072,  8671,  9409, 10203,  9602, 22366,  8041,  8006,     1,   1]], device='cuda:0')
        
        # /data/hdd2/user/ympark/2023/12/01_chatgpt/prj_root/chatgpt/models/generation.py
        sequences = generate(self.model, input_ids, **kwargs)
        # print('sequences',sequences.shape)
        # print('sequences',sequences)
        # sequences torch.Size([8, 128])
        # tensor([[ 9054,  9867, 11474,  8022,  8084,   406,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,   1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1z],
        #         [40053, 12789,  8615,  9341,  8084,   406,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,   1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1],
        #         [19785, 12380,  8135, 13119, 13456,  9341,  8084,   406,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,   1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1],
        #         [15215,  9913, 26999, 11886, 15706, 16860, 13833, 16608,  9207, 10805, 10560,  9050,  8645,  8199,  7397, 12749,  9559, 26526,  9099, 11862, 406,   434,   452,   434,   452,   434,   452,  7965,  8645,  8199, 7397, 12749,  9908,  9270, 28401, 26055,  8765, 21154,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1],
        #         [10165,  9059,  7048,  7162, 11064, 46651,  9625,  8017,  8006,  8084,   406,     1,     1,     1,     1,     1,     1,     1,     1,     1,   1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1],
        #         [ 9267, 12817,  7192,  8704, 10070,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,   1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1],
        #         [34790,  9448,   739,  7920,  9564,  9421,   406,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,   1,   739,  7920, 35343, 10142,  9180,   739,  7920, 35343, 10142, 9180,   739,  7920, 35343, 10142, 21154,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1],
        #         [ 9435,  7487,  9273, 15723, 10006, 15624, 11435,  9755,  8645,  7374, 32974,  9072,  8671,  9409, 10203,  9602, 22366,  8041,  8006,     1,   1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,    1,     1,     1,     1,     1,     1,     1,     1]], device='cuda:0')
        
        # ================================================================================
        attention_mask = None
        
        pad_token_id = kwargs.get('pad_token_id', None)
        # print('pad_token_id',pad_token_id)
        # pad_token_id 1

        if pad_token_id is not None:
            # Store 1 at the position where sequeces has actual value
            attention_mask = sequences.not_equal(pad_token_id).to(dtype=torch.long, device=sequences.device)
            torch.set_printoptions(profile='default')
            # print('attention_mask',attention_mask)
            # print('attention_mask',attention_mask.shape)
            # @ tensor([[1, 1, 1,  ..., 0, 0, 0],
            #         [1, 1, 1,  ..., 0, 0, 0],
            #         [1, 1, 1,  ..., 0, 0, 0],
            #         ...,
            #         [1, 1, 1,  ..., 0, 0, 0],
            #         [1, 1, 1,  ..., 0, 0, 0],
            #         [1, 1, 1,  ..., 0, 0, 0]], device='cuda:0')
            # @ attention_mask torch.Size([8, 128])
        
        # ================================================================================
        # print('return_action_mask',return_action_mask)
        # return_action_mask True

        if not return_action_mask:
            return sequences, attention_mask, None
        
        # ================================================================================
        # Make action mask

        # Token length of question texts
        input_len = input_ids.size(1)
        # print('input_len',input_len)
        # input_len 21
       
        eos_token_id = kwargs.get('eos_token_id', None)
        # print('eos_token_id',eos_token_id)
        # eos_token_id 1

        # print('sequences',sequences.shape)
        # sequences torch.Size([8, 128])
        if eos_token_id is None:
            action_mask = torch.ones_like(sequences, dtype=torch.bool)
        else:
            # left padding may be applied, only mask action
            action_mask = (sequences[:, input_len:] == eos_token_id).cumsum(dim=-1) == 0
            action_mask = F.pad(action_mask, (1 + input_len, -1), value=True)    # include eos token and input

        # Insert "False" into the positions where input tokens locate
        # Insert "True" into the positions where actual answer tokens locate, otherise, "False"
        action_mask[:, :input_len] = False
        torch.set_printoptions(threshold=torch.inf)
        # print('action_mask',action_mask)
        # print('action_mask',action_mask.shape)
        # sequences torch.Size([8, 128])
        # tensor([[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
        #         [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
        #         [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
        #         [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
        #         [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
        #         [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
        #         [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True],
        #         [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]], device='cuda:0')

        # Remove first elements
        action_mask = action_mask[:, 1:]
        # print('action_mask',action_mask)
        # print('action_mask',action_mask.shape)
        # tensor([[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
        #         [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
        #         [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
        #         [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
        #         [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
        #         [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
        #         [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True],
        #         [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]], device='cuda:0')
        # action_mask torch.Size([8, 127])

        # Select only answer text part (Trues are actual answer token, Falses are eos token)
        # print('sequences.size(1) - input_len',sequences.size(1) - input_len)
        # "question text + answer text length (128)" - "question text (21)" = 107
        action_mask_result=action_mask[:, -(sequences.size(1) - input_len):]
        # print('action_mask_result',action_mask_result)
        # print('action_mask_result',action_mask_result.shape)
        # tensor([[ True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
        #         [ True,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
        #         [ True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
        #         [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
        #         [ True,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
        #         [ True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
        #         [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True],
        #         [ True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]], device='cuda:0')
        # action_mask_result torch.Size([8, 107])

        return sequences, attention_mask, action_mask_result

    def forward(self,
                sequences: torch.LongTensor,
                num_actions: int,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns action log probs
        """
        
        # Actor model
        # print('self.model',self.model)
        # GPT2LMHeadModel(
        #   (transformer): GPT2Model(
        #     (wte): Embedding(51200, 768)
        #     (wpe): Embedding(1024, 768)
        #     (drop): Dropout(p=0.1, inplace=False)
        #     (h): ModuleList(
        #     ...
        #     (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        #   )
        #   (lm_head): Linear(in_features=768, out_features=51200, bias=False)
        # )

        # print('sequences',sequences)
        # print('sequences',sequences.shape)
        # tensor([[ 9054,  9867, 11474,  ...,     1,     1,     1],
        #         [40053, 12789,  8615,  ...,     1,     1,     1],
        #         [19785, 12380,  8135,  ...,     1,     1,     1],
        #         ...,
        #         [ 9267, 12817,  7192,  ...,     1,     1,     1],
        #         [34790,  9448,   739,  ...,     1,     1,     1],
        #         [ 9435,  7487,  9273,  ...,     1,     1,     1]], device='cuda:0')
        # sequences torch.Size([8, 128])

        # Predict next token of input tokens
        # attention_mask has 1 at the position where tokens exist in question text
        output = self.model(sequences, attention_mask=attention_mask)

        logits = output['logits']
        # print('logits',logits)
        # print('logits',logits.shape)
        # @ tensor([[[-3.8986, -3.4193, -3.3926,  ..., -1.6675, -3.2028, -1.2846],
        #          ...,
        #          [-3.3784,  7.5431, -3.1762,  ..., -1.5754, -2.8480, -3.9282]],
        #         ...,
        #         [[-3.8756, -3.4642, -3.4005,  ..., -1.5833, -3.3147, -1.4206],
        #          ...,
        #          [-3.7845,  3.3379, -2.8064,  ..., -0.5975, -1.2739, -5.0797]]], device='cuda:0')
        # logits torch.Size([8, 128, 51200])

        # Exclude next token prediction for last answer token
        # print('logits[:, :-1, :]',logits[:, :-1, :])
        # print('logits[:, :-1, :]',logits[:, :-1, :].shape)
        # tensor([[[-3.8986, -3.4193, -3.3926,  ..., -1.6675, -3.2028, -1.2846],
        #         ...,
        #         [-3.4365,  8.2040, -3.2188,  ..., -1.8284, -3.1315, -3.8542]],
        #         ...,
        #         [[-3.8756, -3.4642, -3.4005,  ..., -1.5833, -3.3147, -1.4206],
        #         ...,
        #         [-3.9770,  4.4364, -2.8617,  ..., -0.9216, -1.6009, -5.3784]]], device='cuda:0')
        # logits[:, :-1, :] torch.Size([8, 127, 51200])

        # Exclude first token from question text
        # print('sequences[:, 1:]',sequences[:, 1:])
        # print('sequences[:, 1:]',sequences[:, 1:].shape)
        # tensor([[ 9867, 11474,  8022,  ...,     1,     1,     1],
        #         [12789,  8615,  9341,  ...,     1,     1,     1],
        #         [12380,  8135, 13119,  ...,     1,     1,     1],
        #         ...,
        #         [12817,  7192,  8704,  ...,     1,     1,     1],
        #         [ 9448,   739,  7920,  ...,     1,     1,     1],
        #         [ 7487,  9273, 15723,  ...,     1,     1,     1]], device='cuda:0')
        # sequences[:, 1:] torch.Size([8, 127])

        # Log prob for question and answer tokens
        log_probs = log_probs_from_logits(logits[:, :-1, :], sequences[:, 1:])
        # print('log_probs',log_probs)
        # print('log_probs',log_probs.shape)
        # tensor([[ -8.9209,  -8.9993,  -2.6457,  ...,  -4.3871,  -3.7550,  -3.7953],
        #         [ -9.2708,  -4.5657, -11.3568,  ...,  -6.4524,  -5.8840,  -5.8954],
        #         [ -8.9268,  -3.3010,  -7.7046,  ...,  -5.7153,  -5.0063,  -5.0792],
        #         ...,
        #         [ -8.8307,  -5.0031,  -1.3171,  ...,  -9.1095,  -8.8041,  -8.7211],
        #         [ -9.1180,  -5.9614,  -4.4073,  ...,  -7.3220,  -6.8033,  -6.7970],
        #         [ -8.6193,  -9.4590,  -6.2830,  ...,  -7.8957,  -7.5182,  -7.2093]], device='cuda:0')
        # log_probs torch.Size([8, 127])

        # Log prob for answer tokens
        final_log_probs=log_probs[:, -num_actions:]
        # print('final_log_probs',final_log_probs)
        # print('final_log_probs',final_log_probs.shape)
        # tensor([[-3.4579, -4.0531, -3.6882,  ..., -4.3871, -3.7550, -3.7953],
        #         [-5.1999, -5.9737, -5.6426,  ..., -6.4524, -5.8840, -5.8954],
        #         [-5.5437, -5.8969, -5.5854,  ..., -5.7153, -5.0063, -5.0792],
        #         ...,
        #         [-8.5410, -8.8707, -8.6848,  ..., -9.1095, -8.8041, -8.7211],
        #         [-2.4117, -0.5632, -3.8502,  ..., -7.3220, -6.8033, -6.7970],
        #         [-6.2464, -6.9314, -6.6832,  ..., -7.8957, -7.5182, -7.2093]], device='cuda:0')
        # final_log_probs torch.Size([8, 107])

        return final_log_probs
