from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

try:
    from transformers.generation_logits_process import (
        LogitsProcessorList,
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )
except ImportError:
    from transformers.generation import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper


def prepare_logits_processor(top_k: Optional[int] = None,
                             top_p: Optional[float] = None,
                             temperature: Optional[float] = None) -> LogitsProcessorList:
    
    # print('top_k',top_k)
    # print('top_p',top_p)
    # print('temperature',temperature)
    # top_k 50
    # top_p None
    # temperature 1.0

    top_k=0
    top_p=0
    temperature=None
    
    processor_list = LogitsProcessorList()
    if temperature is not None and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if top_k is not None and top_k != 0:
        processor_list.append(TopKLogitsWarper(top_k))
    if top_p is not None and top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    return processor_list


def _is_sequence_finished(unfinished_sequences: torch.Tensor) -> bool:
    if dist.is_initialized() and dist.get_world_size() > 1:
        # consider DP
        unfinished_sequences = unfinished_sequences.clone()
        dist.all_reduce(unfinished_sequences)
    return unfinished_sequences.max() == 0


def sample(model: nn.Module,
           input_ids: torch.Tensor,
           max_length: int,
           early_stopping: bool = False,
           eos_token_id: Optional[int] = None,
           pad_token_id: Optional[int] = None,
           top_k: Optional[int] = None,
           top_p: Optional[float] = None,
           temperature: Optional[float] = None,
           prepare_inputs_fn: Optional[Callable[[torch.Tensor, Any], dict]] = None,
           update_model_kwargs_fn: Optional[Callable[[dict, Any], dict]] = None,
           **model_kwargs) -> torch.Tensor:
    
    # print('input_ids.size',input_ids.size())
    # print('input_ids',input_ids)
    # input_ids.size torch.Size([8, 21])
    # tensor([[ 9054,  9867, 11474,  8022,  8084,   406,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,   1],
    #         [40053, 12789,  8615,  9341,  8084,   406,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,   1],
    #         [19785, 12380,  8135, 13119, 13456,  9341,  8084,   406,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,   1],
    #         [15215,  9913, 26999, 11886, 15706, 16860, 13833, 16608,  9207, 10805, 10560,  9050,  8645,  8199,  7397, 12749,  9559, 26526,  9099, 11862, 406],
    #         [10165,  9059,  7048,  7162, 11064, 46651,  9625,  8017,  8006,  8084,   406,     1,     1,     1,     1,     1,     1,     1,     1,     1,   1],
    #         [ 9267, 12817,  7192,  8704, 10070,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,   1],
    #         [34790,  9448,   739,  7920,  9564,  9421,   406,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,   1],
    #         [ 9435,  7487,  9273, 15723, 10006, 15624, 11435,  9755,  8645,  7374, 32974,  9072,  8671,  9409, 10203,  9602, 22366,  8041,  8006,     1,   1]], device='cuda:0')

    # ================================================================================
    if input_ids.size(1) >= max_length:
        return input_ids

    # ================================================================================
    # Give variability to answer text by using the following options (top k, top p, temperature)

    logits_processor = prepare_logits_processor(top_k, top_p, temperature)
    
    # ================================================================================
    # If the first-batch answer text from first-batch question is finished, first value will be 0

    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    # print('unfinished_sequences',unfinished_sequences)
    # unfinished_sequences tensor([1, 1, 1, 1, 1, 1, 1, 1], device='cuda:0')

    # ================================================================================
    # Generate answer text maximally 128 characters length
    # print('max_length',max_length)
    # max_length 128

    for _ in range(input_ids.size(1), max_length):

        # print('prepare_inputs_fn',prepare_inputs_fn)
        # prepare_inputs_fn None

        model_inputs = prepare_inputs_fn(input_ids, **model_kwargs) if prepare_inputs_fn is not None else {
            'input_ids': input_ids
        }

        # print('model_inputs',model_inputs)
        # {'input_ids': tensor([[ 9054,  9867, 11474,  8022,  8084,   406,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1],
        #                       [40053, 12789,  8615,  9341,  8084,   406,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1],
        #                       [19785, 12380,  8135, 13119, 13456,  9341,  8084,   406,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1],
        #                       [15215,  9913, 26999, 11886, 15706, 16860, 13833, 16608,  9207, 10805, 10560,  9050,  8645,  8199,  7397, 12749,  9559, 26526,  9099, 11862,   406],
        #                       [10165,  9059,  7048,  7162, 11064, 46651,  9625,  8017,  8006,  8084,   406,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1],
        #                       [ 9267, 12817,  7192,  8704, 10070,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1],
        #                       [34790,  9448,   739,  7920,  9564,  9421,   406,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1],
        #                       [ 9435,  7487,  9273, 15723, 10006, 15624, 11435,  9755,  8645,  7374, 32974,  9072,  8671,  9409, 10203,  9602, 22366,  8041,  8006,     1,     1]],device='cuda:0'), 
        #  'past_key_values': None, 
        #  'use_cache': None, 
        #  'position_ids': tensor([[ 0,  1,  2,  3,  4,  5,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
        #                          [ 0,  1,  2,  3,  4,  5,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
        #                          [ 0,  1,  2,  3,  4,  5,  6,  7,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
        #                          [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        #                          [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
        #                          [ 0,  1,  2,  3,  4,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
        #                          [ 0,  1,  2,  3,  4,  5,  6,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
        #                          [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,  1,  1]], device='cuda:0'), 
        #  'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]], device='cuda:0'), 
        #  'token_type_ids': None}

        # ================================================================================
        # Actor model which comes from SFT-performed GPT and generates answer text

        # print('model',model)
        # model GPT2LMHeadModel(
        #   (transformer): GPT2Model(
        #     (wte): Embedding(51200, 768)
        #     (wpe): Embedding(1024, 768)
        #     (drop): Dropout(p=0.1, inplace=False)
        #     (h): ModuleList(
        #       (0): GPT2Block(
        #     ...
        #     (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        #   )
        #   (lm_head): Linear(in_features=768, out_features=51200, bias=False)
        # )
        # print('model',type(model))
        # model <class 'transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel'>
        
        outputs = model(**model_inputs)
        # print('outputs',outputs)

        # ================================================================================
        # Logits for next predicted token

        next_token_logits = outputs['logits'][:, -1, :]
        torch.set_printoptions(profile='default')
        # print('next_token_logits',next_token_logits)
        # print('next_token_logits',next_token_logits.shape)
        # tensor([[-3.7187,  5.6681, -4.1936,  ..., -2.4930, -3.0394, -4.9214],
        #         [-3.4295,  3.3808, -4.4816,  ..., -0.5014, -2.6560, -4.2022],
        #         [-4.3052,  4.7548, -3.6852,  ..., -1.5579, -4.2912, -5.4364],
        #         ...,
        #         [-5.3198,  1.2971, -4.5880,  ..., -0.2356, -2.1810, -5.4626],
        #         [-4.7428,  3.8894, -4.6884,  ..., -1.1980, -3.6620, -4.7259],
        #         [-3.8332,  4.1226, -3.9694,  ..., -1.1172, -3.8643, -5.4416]], device='cuda:0')
        # next_token_logits torch.Size([8, 51200])

        # ================================================================================
        # Apply temperature, top_k, top_p options in selecting next token
        # pre-process distribution

        next_token_logits = logits_processor(input_ids, next_token_logits)
        # print('next_token_logits',next_token_logits)
        # print('next_token_logits',next_token_logits.shape)
        # tensor([[-inf, -inf, -inf,  ..., -inf, -inf, -inf],
        #         [-inf, -inf, -inf,  ..., -inf, -inf, -inf],
        #         [-inf, -inf, -inf,  ..., -inf, -inf, -inf],
        #         ...,
        #         [-inf, -inf, -inf,  ..., -inf, -inf, -inf],
        #         [-inf, -inf, -inf,  ..., -inf, -inf, -inf],
        #         [-inf, -inf, -inf,  ..., -inf, -inf, -inf]], device='cuda:0')
        # next_token_logits torch.Size([8, 51200])

        # ================================================================================
        # sample

        probs = torch.softmax(next_token_logits, dim=-1, dtype=torch.float)
        # print('probs',probs)
        # print('probs',probs.shape)
        # tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        #         [0., 0., 0.,  ..., 0., 0., 0.],
        #         [0., 0., 0.,  ..., 0., 0., 0.],
        #         ...,
        #         [0., 0., 0.,  ..., 0., 0., 0.],
        #         [0., 0., 0.,  ..., 0., 0., 0.],
        #         [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0')
        # probs torch.Size([8, 51200])
        
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        # print('next_tokens',next_tokens)
        # print('next_tokens',next_tokens.shape)
        # next_tokens tensor([  406,   739, 13119,   434,   739, 10586,   739,   406], device='cuda:0')
        # next_tokens torch.Size([8])

        # ================================================================================
        # finished sentences should have their next token be a padding token
        
        # print('eos_token_id',eos_token_id)
        # print('pad_token_id',pad_token_id)
        # eos_token_id 1
        # pad_token_id 1

        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            
            # print('next_tokens',next_tokens)
            # print('next_tokens',next_tokens.shape)
            # next_tokens tensor([  406,   739, 13119,   434,   739, 10586,   739,   406], device='cuda:0')
            # next_tokens torch.Size([8])
            
            # print('unfinished_sequences',unfinished_sequences)
            # print('unfinished_sequences',unfinished_sequences.shape)
            # unfinished_sequences tensor([1, 1, 1, 1, 1, 1, 1, 1], device='cuda:0')
            # unfinished_sequences torch.Size([8])
            
            # print('pad_token_id',pad_token_id)
            # pad_token_id 1
            
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            # print('next_tokens',next_tokens)
            # next_tokens tensor([  406,   739, 13119,   434,   739, 10586,   739,   406], device='cuda:0')

        # ================================================================================
        # Concatenate "input id" and "next token" which will be passed to Actor model to generate "next token"

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        # print('input_ids',input_ids)
        # tensor([[ 9054,  9867, 11474,  8022,  8084,   406,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,   406],
        #         [40053, 12789,  8615,  9341,  8084,   406,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,   739],
        #         [19785, 12380,  8135, 13119, 13456,  9341,  8084,   406,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1, 13119],
        #         [15215,  9913, 26999, 11886, 15706, 16860, 13833, 16608,  9207, 10805, 10560,  9050,  8645,  8199,  7397, 12749,  9559, 26526,  9099, 11862,   406,   434],
        #         [10165,  9059,  7048,  7162, 11064, 46651,  9625,  8017,  8006,  8084,   406,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,   739],
        #         [ 9267, 12817,  7192,  8704, 10070,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1, 10586],
        #         [34790,  9448,   739,  7920,  9564,  9421,   406,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,   739],
        #         [ 9435,  7487,  9273, 15723, 10006, 15624, 11435,  9755,  8645,  7374, 32974,  9072,  8671,  9409, 10203,  9602, 22366,  8041,  8006,     1,     1,   406]], device='cuda:0')

        # ================================================================================
        # print('update_model_kwargs_fn',update_model_kwargs_fn)
        # update_model_kwargs_fn <function update_model_kwargs_fn at 0x7fd81cf83c10>
        if update_model_kwargs_fn is not None:
            model_kwargs = update_model_kwargs_fn(outputs, **model_kwargs)

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id is not None:
            unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

        # stop when each sentence is finished if early_stopping=True
        if early_stopping and _is_sequence_finished(unfinished_sequences):
            break

    # ================================================================================
    # Question text with pads + answer text with pads
    
    torch.set_printoptions(threshold=torch.inf)
    # print('input_ids',input_ids)
    # print('input_ids',input_ids.shape)
    # tensor([[ 9054,  9867, 11474,  8022,  8084,   406,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,   406,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1],
    #     [40053, 12789,  8615,  9341,  8084,   406,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,   739,   110,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1],
    #     [19785, 12380,  8135, 13119, 13456,  9341,  8084,   406,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1, 13119,  9090, 29114, 17756,  9025, 32987,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1],
    #     [15215,  9913, 26999, 11886, 15706, 16860, 13833, 16608,  9207, 10805, 10560,  9050,  8645,  8199,  7397, 12749,  9559, 26526,  9099, 11862,   406,   434,   452,   434,   452,   434,   452,  7965,  8645,  8199,  7397, 12749,  9908,  9270, 28401, 26055,  8765, 21154,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1],
    #     [10165,  9059,  7048,  7162, 11064, 46651,  9625,  8017,  8006,  8084,   406,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,   739,   110,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1],
    #     [ 9267, 12817,  7192,  8704, 10070,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1, 10586, 12557,  9069,  7801,  8084,   376,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1],
    #     [34790,  9448,   739,  7920,  9564,  9421,   406,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,   739,  7920,  8146,  9366,  9031,  9144, 16691, 10351,  9022,  6866, 12371,  9366,  9031,  9144, 16691,   739,  7920,  8146,  9366,  9031,  9144, 16650,  9022,  6866, 12371,  9366,  9031,  9144, 10846,  9022,  6866, 12371,  9366,  9031,  9144, 10846,  9022,  6866, 12371,  9366,  9031,  9144, 10846,  9022,  6866, 12371,  9366,  9031,  9144, 10846,  9022,  6866, 12371,  9366,  9031,  9144, 10846,  9022,  6866, 12371,  9366,  9031,  9144, 10846,  9022,  6866, 12371,  9366,  9031,  9144, 10846,  9022,  6866, 12371,  9366,  9031,  9144, 10846,  9022,  6866, 12371,  9366,  9031,  9144, 10846,  9022,  6866, 12371,  9366,  9031,  9144, 10846,  9022,  6866, 12371,  9366,  9031,  9144, 10846,  9022,  6866, 12371,  9366,  9031,  9144, 10846,  9022],
    #     [ 9435,  7487,  9273, 15723, 10006, 15624, 11435,  9755,  8645,  7374, 32974,  9072,  8671,  9409, 10203,  9602, 22366,  8041,  8006,     1,     1,   406,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1]], device='cuda:0')input_ids torch.Size([8, 128])
    # input_ids torch.Size([8, 128])
    
    return input_ids

def generate(model: nn.Module,
             input_ids: torch.Tensor,
             max_length: int,
             num_beams: int = 1,
             do_sample: bool = True,
             early_stopping: bool = False,
             eos_token_id: Optional[int] = None,
             pad_token_id: Optional[int] = None,
             top_k: Optional[int] = None,
             top_p: Optional[float] = None,
             temperature: Optional[float] = None,
             prepare_inputs_fn: Optional[Callable[[torch.Tensor, Any], dict]] = None,
             update_model_kwargs_fn: Optional[Callable[[dict, Any], dict]] = None,
             **model_kwargs) -> torch.Tensor:
    """Generate token sequence. The returned sequence is input_ids + generated_tokens.

    Args:
        model (nn.Module): model
        input_ids (torch.Tensor): input sequence
        max_length (int): max length of the returned sequence
        num_beams (int, optional): number of beams. Defaults to 1.
        do_sample (bool, optional): whether to do sample. Defaults to True.
        early_stopping (bool, optional): if True, the sequence length may be smaller than max_length due to finding eos. Defaults to False.
        eos_token_id (Optional[int], optional): end of sequence token id. Defaults to None.
        pad_token_id (Optional[int], optional): pad token id. Defaults to None.
        top_k (Optional[int], optional): the number of highest probability vocabulary tokens to keep for top-k-filtering. Defaults to None.
        top_p (Optional[float], optional): If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation. Defaults to None.
        temperature (Optional[float], optional): The value used to module the next token probabilities. Defaults to None.
        prepare_inputs_fn (Optional[Callable[[torch.Tensor, Any], dict]], optional): Function to preprocess model inputs. Arguments of this function should be input_ids and model_kwargs. Defaults to None.
        update_model_kwargs_fn (Optional[Callable[[dict, Any], dict]], optional): Function to update model_kwargs based on outputs. Arguments of this function should be outputs and model_kwargs. Defaults to None.
    """
    
    is_greedy_gen_mode = ((num_beams == 1) and do_sample is False)
    is_sample_gen_mode = ((num_beams == 1) and do_sample is True)
    is_beam_gen_mode = ((num_beams > 1) and do_sample is False)
    
    if is_greedy_gen_mode:
        # run greedy search
        raise NotImplementedError
    elif is_sample_gen_mode:
        # run sample
        return sample(model,
                      input_ids,
                      max_length,
                      early_stopping=early_stopping,
                      eos_token_id=eos_token_id,
                      pad_token_id=pad_token_id,
                      top_k=top_k,
                      top_p=top_p,
                      temperature=temperature,
                      prepare_inputs_fn=prepare_inputs_fn,
                      update_model_kwargs_fn=update_model_kwargs_fn,
                      **model_kwargs)
    elif is_beam_gen_mode:
        raise NotImplementedError
    else:
        raise ValueError("Unsupported generation mode")
