# ================================================================================
import os,shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.optim import Adam

from datasets import load_dataset # pip install datasets

import transformers
from transformers import pipeline
from transformers import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, pipeline
from transformers import Trainer, TrainingArguments, AutoModelWithLMHead
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import AutoTokenizer, BloomTokenizerFast
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer

from chatgpt.experience_maker import NaiveExperienceMaker
from chatgpt.experience_maker import NaiveExperienceMaker
from chatgpt.trainer.strategies import ColossalAIStrategy, DDPStrategy, NaiveStrategy
from chatgpt.dataset import RewardDataset
from chatgpt.models.bloom import BLOOMRM
from chatgpt.models.gpt import GPTRM
from chatgpt.models.opt import OPTRM
from chatgpt.trainer import RewardModelTrainer
# from ..base import RewardModel
from chatgpt.models.base import RewardModel, Actor
from chatgpt.models.bloom import BLOOMActor, BLOOMCritic
from chatgpt.models.gpt import GPTActor, GPTCritic
from chatgpt.models.opt import OPTActor, OPTCritic
from chatgpt.trainer import PPOTrainer

import loralib as lora
from colossalai.nn.optimizer import HybridAdam

from copy import deepcopy
import pandas as pd
import argparse
import copy
import logging
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import random

# ================================================================================
random.seed(230319)

# ================================================================================
# Configure variables

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
PROMPT_DICT = {
    "prompt_input": (
         "### Instruction:\n{prompt}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "### Instruction:\n{prompt}\n\n### Response:"
    ),
}

# ================================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default='./output_2_RM')
parser.add_argument('--data_path_2_RM', type=str, default='./data_kochatgpt/kochatgpt_2_RM.jsonl', 
                    help='https://huggingface.co/datasets/fka/awesome-chatgpt-prompts/blob/main/prompts.csv')
parser.add_argument('--strategy',
                    choices=['naive', 'ddp', 'colossalai_gemini', 'colossalai_zero2'],
                    default='naive')
parser.add_argument('--model', type=str, default='gpt2', choices=['gpt2', 'bloom', 'opt'])
parser.add_argument('--pretrain', type=str, default=None)
parser.add_argument('--dataset', type=str, default='Dahoas/rm-static')
parser.add_argument('--save_path', type=str, default='rm_ckpt.pth')
parser.add_argument('--max_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--lora_rank', type=int, default=0, help="low-rank adaptation matrices rank")
parser.add_argument('--max_len', type=int, default=512)  # wygo 추가

args = parser.parse_args(args=[])

# for test
args.max_epochs = 3
args.pretrain = 'skt/kogpt2-base-v2'  # pretrained 모델 가져오기
args.verbose = True

print(args)
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# ================================================================================
# configure strategy

# print('args.strategy',args.strategy)
# args.strategy naive

if args.strategy == 'naive':
    strategy = NaiveStrategy()
    # class NaiveStrategy(Strategy):
    # class Strategy(ABC):
elif args.strategy == 'ddp':
    strategy = DDPStrategy()
elif args.strategy == 'colossalai_gemini':
    strategy = ColossalAIStrategy(stage=3, placement_policy='cuda')
elif args.strategy == 'colossalai_zero2':
    strategy = ColossalAIStrategy(stage=2, placement_policy='cuda')
else:
    raise ValueError(f'Unsupported strategy "{args.strategy}"')

# ================================================================================
# customizing, https://github.com/hpcaitech/ColossalAI/blob/2e16f842a9e5b1fb54e7e41070e9d2bb5cd64d7c/applications/ChatGPT/chatgpt/nn/gpt_rm.py#L29

class GPTRM_custom(RewardModel):
    """
    GPT Reward model.
    Args:
        pretrained (str): Pretrained model name or path.
        config (GPT2Config): Model config.
        checkpoint (bool): Enable gradient checkpointing.
        lora_rank (int): Rank of the low-rank approximation.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(self,
                 pretrained: Optional[str] = None,
                 config: Optional[GPT2Config] = None,
                 checkpoint: bool = False,
                 lora_rank: int = 0,
                 lora_train_bias: str = 'none',
                 tokenizer=None) -> None:
        
        # print('pretrained',pretrained)
        # pretrained skt/kogpt2-base-v2

        if pretrained is not None:
            model = GPT2Model.from_pretrained(pretrained)
            model.resize_token_embeddings(len(tokenizer))  # wygo 추가!!!
        elif config is not None:
            model = GPT2Model(config)
        else:
            model = GPT2Model(GPT2Config())

        if checkpoint:
            model.gradient_checkpointing_enable()

        # model = model.resize_token_embeddings(len(tokenizer))

        # print('model',model)
        # GPT2Model(
        #   (wte): Embedding(51201, 768)
        #   (wpe): Embedding(1024, 768)
        #   (drop): Dropout(p=0.1, inplace=False)
        #   (h): ModuleList(
        #     (0): GPT2Block(
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

        # print('model.config.n_embd',model.config.n_embd)
        # model.config.n_embd 768
        value_head = nn.Linear(model.config.n_embd, 1)
        # print('value_head',value_head)
        # value_head Linear(in_features=768, out_features=1, bias=True)

        super().__init__(model, value_head, lora_rank, lora_train_bias)

        # 추가, 230421
        if pretrained is not None:
            self.model = model
            self.pretrained = pretrained

    # 추가, 230421, config.json을 생성하기 위해 추가
    def save_pretrained(self, dir):
        if self.pretrained is not None:
            self.model.save_pretrained(dir)

# ================================================================================
# configure model, tokenizer
with strategy.model_init_context():
    
    # print(args.model)
    # gpt2
    
    if args.model == 'gpt2':
        # print(args.pretrain)
        # skt/kogpt2-base-v2
        
        # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # tokenizer = AutoTokenizer.from_pretrained(args.pretrain)
        tokenizer = AutoTokenizer.from_pretrained(args.pretrain, padding_side="right", model_max_length=512)
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )
        tokenizer.pad_token = tokenizer.eos_token
        model = GPTRM_custom(pretrained=args.pretrain, lora_rank=args.lora_rank, tokenizer=tokenizer).cuda()
        # print('model',model)

    elif args.model == 'bloom':
        model = BLOOMRM(pretrained=args.pretrain, lora_rank=args.lora_rank).cuda()
        tokenizer = BloomTokenizerFast.from_pretrained(args.pretrain)

    elif args.model == 'opt':
        model = OPTRM(pretrained=args.pretrain, lora_rank=args.lora_rank).cuda()
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

    else:
        raise ValueError(f'Unsupported model "{args.model}"')

    # model.resize_token_embeddings(len(tokenizer))

# ================================================================================
# Make ranking data to "chosen", "rejetced" data

with open(args.data_path_2_RM, "r", encoding='utf-8-sig') as json_file:
    list_data_dict = json.load(json_file)
    if args.verbose:
        print('## data check ##')
        print((list_data_dict[0]))
# {'prompt': '번디는 자신이 탐정잡지, 범죄소설 그리고 성범죄 관련 실제 범죄 다큐멘터리들을 탐독했다고 누구에게 말했나?', 
#  'completion_0': 'Allow me to answer your question. I know that you are curious about me.', 
#  'completion_1': '번디는 다양한 인터뷰자들과 뉴스홍보 담당자들과의 면담 때 밝혔다.', 
#  'completion_2': '라이언에게 말했다.', 
#  'ranking': [2, 1, 0]}

# ================================================================================
total_data_ranking2chosen = []
for tmp in list_data_dict:
    one_data_ranking2chosen = []

    # data 1) 0 VS 1
    data = {}
    data['prompt'] = tmp['prompt']
    if tmp['ranking'][0] < tmp['ranking'][1]:
        data['chosen'] = tmp['completion_0']
        data['rejected'] = tmp['completion_1']
    else:
        data['chosen'] = tmp['completion_1']
        data['rejected'] = tmp['completion_0']
    one_data_ranking2chosen.append(data)

    # data 2) 0 VS 2
    data = {}
    data['prompt'] = tmp['prompt']
    if tmp['ranking'][0] < tmp['ranking'][2]:
        data['chosen'] = tmp['completion_0']
        data['rejected'] = tmp['completion_2']
    else:
        data['chosen'] = tmp['completion_2']
        data['rejected'] = tmp['completion_0']
    one_data_ranking2chosen.append(data)

    # data 1) 1 VS 2
    data = {}
    data['prompt'] = tmp['prompt']
    if tmp['ranking'][1] < tmp['ranking'][2]:
        data['chosen'] = tmp['completion_1']
        data['rejected'] = tmp['completion_2']
    else:
        data['chosen'] = tmp['completion_2']
        data['rejected'] = tmp['completion_1']
    one_data_ranking2chosen.append(data)

    total_data_ranking2chosen.extend(one_data_ranking2chosen)

# print('before data num: %d'%(len(list_data_dict)))
# print('after  data num: %d'%(len(total_data_ranking2chosen)))
# print('data example: \n%s'%total_data_ranking2chosen[0])
# print('data example: \n%s'%total_data_ranking2chosen[1])
# print('data example: \n%s'%total_data_ranking2chosen[2])
# before data num: 10220
# after  data num: 30660
# Data1: {'prompt': '번디는 자신이 탐정잡지, 범죄소설 그리고 성범죄 관련 실제 범죄 다큐멘터리들을 탐독했다고 누구에게 말했나?', 'chosen': '번디는 다양한 인터뷰자들과 뉴스홍보 담당자들과의 면담 때 밝혔다.', 'rejected': 'Allow me to answer your question. I know that you are curious about me.'}
# Data2: {'prompt': '번디는 자신이 탐정잡지, 범죄소설 그리고 성범죄 관련 실제 범죄 다큐멘터리들을 탐독했다고 누구에게 말했나?', 'chosen': '라이언에게 말했다.', 'rejected': 'Allow me to answer your question. I know that you are curious about me.'}
# Data3: {'prompt': '번디는 자신이 탐정잡지, 범죄소설 그리고 성범죄 관련 실제 범죄 다큐멘터리들을 탐독했다고 누구에게 말했나?', 'chosen': '라이언에게 말했다.', 'rejected': '번디는 다양한 인터뷰자들과 뉴스홍보 담당자들과의 면담 때 밝혔다.'}

# ================================================================================
# list_tmp = list(range(10))
random.shuffle(total_data_ranking2chosen)

# train_data = total_data_ranking2chosen[:-1000]  # 29000 학습
# eval_data = total_data_ranking2chosen[-1000:0]  # 1000개만 평가

train_data = total_data_ranking2chosen[:100]  # 29000 학습
eval_data = total_data_ranking2chosen[100:130]  # 1000개만 평가

train_dataset = RewardDataset(train_data, tokenizer, args.max_len)
eval_dataset = RewardDataset(eval_data, tokenizer, args.max_len)

# ================================================================================
# configure optimizer
if args.strategy.startswith('colossalai'):
    optim = HybridAdam(model.parameters(), lr=5e-5)
else:
    optim = Adam(model.parameters(), lr=5e-5)

# ================================================================================
# batch_size here is expected to be C(k,2), k means # response of each prompt
# be limited with the format of dataset 'Dahoas/rm-static', we'd better use batch_size as 1
trainer = RewardModelTrainer(model=model,
                             strategy=strategy,
                             optim=optim,
                             train_dataset=train_dataset,
                             eval_dataset=eval_dataset,
                             batch_size=args.batch_size,
                             max_epochs=args.max_epochs)
# class RewardModelTrainer(ABC):
# def __init__(

# ================================================================================
model_path = os.path.join(args.output_dir, 'RM.pt')
# print('model_path',model_path)
# model_path ./output_2_RM/RM.pt

state_dict = torch.load(model_path)
# print('state_dict',state_dict)
# OrderedDict([('model.wte.weight', tensor([[ 0.0282, -0.0369, -0.0061,  ..., -0.0248, -0.0068,  0.0108],
#         [ 0.0104,  0.0130,  0.0433,  ...,  0.0096, -0.0052, -0.0011],
#         [ 0.0314, -0.0164,  0.0178,  ..., -0.0054, -0.0480,  0.0040],
#         ...,
#         [-0.0206,  0.0045,  0.0241,  ...,  0.0630, -0.0192,  0.0099],
#         [-0.0199, -0.0382,  0.0268,  ...,  0.0299, -0.0154, -0.0423],
#         [-0.0245,  0.0167,  0.0274,  ...,  0.0083, -0.0163,  0.0284]],
#        device='cuda:0')), ('model.wpe.weight', tensor([[ 0.0225, -0.0101,  0.0035,  ...,  0.0084,  0.0046,  0.0032],
#         [ 0.0097, -0.0311,  0.0223,  ..., -0.0223, -0.0139, -0.0227],
#         [-0.0110, -0.0041,  0.0070,  ..., -0.0223, -0.0220,  0.0064],
#         ...,
#         [-0.0003, -0.0184, -0.0051,  ..., -0.0016,  0.0025,  0.0072],
#         [ 0.0009, -0.0152, -0.0034,  ..., -0.0009,  0.0021,  0.0082],
#         [ 0.0249,  0.0263,  0.0298,  ...,  0.0329, -0.0088,  0.0320]],
#         ...,  ('value_head.weight', tensor([[-3.3829e-02, -2.9420e-02,  5.1491e-03,  2.7133e-02, -2.0087e-03,
#           ...
#           2.8504e-02, -2.8202e-02,  8.0693e-03]], device='cuda:0')), ('value_head.bias', tensor([0.0271], device='cuda:0'))])

# ================================================================================
# train
trainer.fit(use_lora=args.lora_rank)

# print('model model',model)
# GPTRM_custom(
#   (model): GPT2Model(
#     (wte): Embedding(51201, 768)
#     (wpe): Embedding(1024, 768)
#     (drop): Dropout(p=0.1, inplace=False)
#     ...
#     (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
#   )
#   (value_head): Linear(in_features=768, out_features=1, bias=True)
# )

# save model checkpoint after fitting on only rank0
strategy.save_model(model, os.path.join(args.output_dir, 'RM.pt'), only_rank0=True)

# save optimizer checkpoint on all ranks
strategy.save_optimizer(optim,
                        os.path.join(args.output_dir, 'RM_optim_checkpoint_%d.pt' % (torch.cuda.current_device())),
                        only_rank0=False)

model.save_pretrained(args.output_dir)  # config.json 생성

# ================================================================================
# 보상모델 체크
def inference_RM(input_text='인공지능은 인공지능 입니다'):
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(
        torch.cuda.current_device())
    output = model(input_ids)
    output_reward = output.cpu().detach().numpy()[0]

    print('input: %s\nreward score: %.1f'%(input_text, output_reward))

    return output_reward

# input_text = '한국은 대한민국 입니다'
input_text = '인공지능은 인공지능 입니다'

output_reward = inference_RM(input_text=input_text)
# input: 인공지능은 인공지능 입니다
# reward score: -0.9