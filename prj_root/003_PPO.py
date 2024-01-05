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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

## clossalAI error 해결
os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'
os.environ['WORLD_SIZE'] = '2'
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '42043'

# data config
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context.\n"
        "아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n"
        "Write a response that appropriately completes the request.\n요청을 적절히 완료하는 응답을 작성하세요.\n\n"
        "### Instruction(명령어):\n{prompt}\n\n### Input(입력):\n{input}\n\n### Response(응답):"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task.\n"
        "아래는 작업을 설명하는 명령어입니다.\n\n"
        "Write a response that appropriately completes the request.\n명령어에 따른 요청을 적절히 완료하는 응답을 작성하세요.\n\n"
        "### Instruction(명령어):\n{prompt}\n\n### Response(응답):"
    ),
}

# ================================================================================
# define argment
parser = argparse.ArgumentParser()
parser.add_argument('--data_path_3_PPO', type=str, default='./data_kochatgpt/kochatgpt_3_PPO.jsonl')
parser.add_argument('--output_dir', type=str, default='./output_3_PPO')
parser.add_argument('--strategy',
                    choices=['naive', 'ddp', 'colossalai_gemini', 'colossalai_zero2'],
                    default='naive')
parser.add_argument('--model', type=str, default='gpt2', choices=['gpt2', 'bloom', 'opt'])
parser.add_argument('--pretrain', type=str, default=None)
parser.add_argument('--num_episodes', type=int, default=10)
parser.add_argument('--max_timesteps', type=int, default=3)
parser.add_argument('--update_timesteps', type=int, default=3)
parser.add_argument('--max_epochs', type=int, default=5)
parser.add_argument('--train_batch_size', type=int, default=8)
parser.add_argument('--lora_rank', type=int, default=0, help="low-rank adaptation matrices rank")
parser.add_argument('--max_length', type=int, default=250)
parser.add_argument('--random_prompts', type=bool, default=False)
args = parser.parse_args(args=[])

# for test
args.output_dir = './output_3_PPO'
args.pretrain = 'skt/kogpt2-base-v2'  # pretrained 모델 가져오기


## 이곳 수정!!
args.pretrain_actor = './output_1_SFT'  # SFT 모델 가져오기
args.pretrain_critic = './output_2_RM'  # RM 모델 가져오기
# args.pretrain_actor = args.pretrain
# args.pretrain_critic = args.pretrain

args.num_episodes = 1
args.max_epochs   = 1
args.train_batch_size = 1

print(args)
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# ================================================================================
# Configure strategy

# print('args.strategy',args.strategy)
# args.strategy naive

if args.strategy == 'naive':
    strategy = NaiveStrategy()
elif args.strategy == 'ddp':
    strategy = DDPStrategy()
elif args.strategy == 'colossalai_gemini':
    strategy = ColossalAIStrategy(stage=3, placement_policy='cuda')
elif args.strategy == 'colossalai_zero2':
    strategy = ColossalAIStrategy(stage=2, placement_policy='cuda')
else:
    raise ValueError(f'Unsupported strategy "{args.strategy}"')

# ================================================================================
with strategy.model_init_context():
    if args.model == 'gpt2':
        # print('args.pretrain_actor',args.pretrain_actor)
        # print('args.pretrain_actor',args.pretrain_critic)
        # args.pretrain_actor ./output_1_SFT
        # args.pretrain_actor ./output_2_RM

        actor = GPTActor(pretrained=args.pretrain_actor, lora_rank=args.lora_rank).to(torch.cuda.current_device())
        critic = GPTCritic(pretrained=args.pretrain_critic, lora_rank=args.lora_rank).to(torch.cuda.current_device())
        # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # tokenizer.pad_token = tokenizer.eos_token
        tokenizer = AutoTokenizer.from_pretrained(args.pretrain, padding_side="right", model_max_length=512)
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == 'bloom':
        actor = BLOOMActor(pretrained=args.pretrain, lora_rank=args.lora_rank).to(torch.cuda.current_device())
        critic = BLOOMCritic(pretrained=args.pretrain, lora_rank=args.lora_rank).to(torch.cuda.current_device())
        tokenizer = BloomTokenizerFast.from_pretrained(args.pretrain)
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == 'opt':
        actor = OPTActor(pretrained=args.pretrain, lora_rank=args.lora_rank).to(torch.cuda.current_device())
        critic = OPTCritic(pretrained=args.pretrain, lora_rank=args.lora_rank).to(torch.cuda.current_device())
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
    else:
        raise ValueError(f'Unsupported model "{args.model}"')

    # Model after performing SFT
    initial_model = deepcopy(actor)
    # print('initial_model',initial_model)
    # initial_model GPTActor(
    #   (model): GPT2LMHeadModel(
    #     (transformer): GPT2Model(
    #       (wte): Embedding(51200, 768)
    #       (wpe): Embedding(1024, 768)
    #       (drop): Dropout(p=0.1, inplace=False)
    #       (h): ModuleList(
    #         (0): GPT2Block(
    #           (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    #     ...
    #       (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    #     )
    #     (lm_head): Linear(in_features=768, out_features=51200, bias=False)
    #   )
    # )
    
    # Model after performing the reward model training
    reward_model = RewardModel(deepcopy(critic.model), deepcopy(critic.value_head)).to(torch.cuda.current_device())
    # print('reward_model',reward_model)
    # reward_model RewardModel(
    #   (model): GPT2Model(
    #     (wte): Embedding(51201, 768)
    #     (wpe): Embedding(1024, 768)
    #     (drop): Dropout(p=0.1, inplace=False)
    #     (h): ModuleList(
    #       (0): GPT2Block(
    #     ...
    #     (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    #   )
    #   (value_head): Linear(in_features=768, out_features=1, bias=True)
    # )

# ================================================================================
# Configure optimizer

# print('args.strategy',args.strategy)
# args.strategy naive

if args.strategy.startswith('colossalai'):
    actor_optim = HybridAdam(actor.parameters(), lr=5e-6)
    critic_optim = HybridAdam(critic.parameters(), lr=5e-6)
else:
    actor_optim = Adam(actor.parameters(), lr=5e-6)
    critic_optim = Adam(critic.parameters(), lr=5e-6)

# ================================================================================
# setting the models

# print('initial_model',initial_model)
# print('reward_model',reward_model)
# print('actor',actor)
# print('actor_optim',actor_optim)
# print('critic',critic)
# print('critic_optim',critic_optim)

(actor, actor_optim), (critic, critic_optim), reward_model, initial_model = strategy.prepare(
    (actor, actor_optim), (critic, critic_optim), reward_model, initial_model)

# ================================================================================
# Prepare data

with open(args.data_path_3_PPO, "r", encoding='utf-8-sig') as json_file:
    list_data_dict = json.load(json_file)
    list_prompt = [tmp['prompt'] for tmp in list_data_dict]
# print(list_prompt)

# ================================================================================
# Define tokenizer for question texts
def tokenize_fn(texts):
    batch = tokenizer(texts, return_tensors='pt', max_length=96, padding=True, truncation=True)
    return {k: v.cuda() for k, v in batch.items()}

# ================================================================================
# Configure trainer

trainer = PPOTrainer(strategy,
                     actor,
                     critic,
                     reward_model,
                     initial_model,
                     actor_optim,
                     critic_optim,
                     max_epochs=args.max_epochs,
                     train_batch_size=args.train_batch_size,
                     tokenizer=tokenize_fn,
                     max_length=128,
                     do_sample=True,
                     temperature=1.0,
                     top_k=50,
                     pad_token_id=tokenizer.pad_token_id,
                     eos_token_id=tokenizer.eos_token_id,
                     random_prompts=args.random_prompts)

# ================================================================================
# train

# print('args.num_episodes',args.num_episodes)
# print('args.max_timesteps',args.max_timesteps)
# print('args.update_timesteps',args.update_timesteps)
# args.num_episodes 1
# args.max_timesteps 3
# args.update_timesteps 3

trainer.fit(list_prompt,                             # 입력 prompt
            num_episodes=args.num_episodes,
            max_timesteps=args.max_timesteps,
            update_timesteps=args.update_timesteps)

# ================================================================================
# save model checkpoint after fitting on only rank0
strategy.save_model(actor, os.path.join(args.output_dir, 'actor.pt'), only_rank0=True)

# save optimizer checkpoint on all ranks
strategy.save_optimizer(actor_optim,
                        os.path.join(args.output_dir, 'actor_optim_checkpoint_%d.pt' % (torch.cuda.current_device())),
                        only_rank0=False)
