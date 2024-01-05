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
# Load pretrained GPT-2 model (Text completion model)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in list(state_dict.items())}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

# ================================================================================
# define argment
parser = argparse.ArgumentParser()
parser.add_argument('--data_path_1_SFT', type=str, default='./data_kochatgpt/kochatgpt_1_SFT.jsonl')
parser.add_argument('--model_name', type=str, default='gpt2', choices=['gpt2', 'bloom', 'opt'])
parser.add_argument('--max_epochs', type=int, default=2)
parser.add_argument('--train_batch_size', type=int, default=2)
parser.add_argument('--output_dir', type=str, default='./output_1_SFT')

args = parser.parse_args(args=[])

# for test
args.model_name = 'skt/kogpt2-base-v2'  # SK GPT2, https://github.com/SKT-AI/KoGPT2
# args.model_name = 'ajoublue-gpt2-base'  # 아주대, https://github.com/HeegyuKim/language-model

args.max_epochs = 2

print('args:\n',args)

# ================================================================================
# Pretrained tokenizer

tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                    bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                                                    pad_token='<pad>', mask_token='<mask>')
print('Tokenize the sentence by pretrained model (skt/kogpt2-base-v2): ',tokenizer.tokenize("안녕하세요. 한국어 GPT-2 입니다.😤:)l^o"))
# ['▁안녕', '하', '세', '요.', '▁한국어', '▁G', 'P', 'T', '-2', '▁입', '니다.', '😤', ':)', 'l^o']

# ================================================================================
# Pretrained GPT-2 model

model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
# print('model',model)
# GPT2LMHeadModel(
#   (transformer): GPT2Model(
#     (wte): Embedding(51200, 768)
#     (wpe): Embedding(1024, 768)
#     (drop): Dropout(p=0.1, inplace=False)
#     (h): ModuleList(
#     ...
#         (mlp): GPT2MLP(
#           (c_fc): Conv1D()
#           (c_proj): Conv1D()
#           (act): NewGELUActivation()
#           (dropout): Dropout(p=0.1, inplace=False)
#         )
#       )
#     )
#     (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
#   )
#   (lm_head): Linear(in_features=768, out_features=51200, bias=False)
# )

# Incompleted input text to GPT-2
text = '근육이 커지기 위해서는'

# Tokenize text
input_ids = tokenizer.encode(text, return_tensors='pt')
# print('input_ids',input_ids)
# input_ids tensor([[33245, 10114, 12748, 11357]])

# Generated text from GPT-2
# Question text token IDs + Answer text token IDs
gen_ids = model.generate(input_ids,
                         max_length=128,
                         repetition_penalty=2.0,
                         pad_token_id=tokenizer.pad_token_id,
                         eos_token_id=tokenizer.eos_token_id,
                         bos_token_id=tokenizer.bos_token_id,
                         use_cache=True)
# print('gen_ids',gen_ids)
# print('gen_ids',gen_ids.shape)
# gen_ids tensor([[33245, 10114, 12748, 11357, 23879, 39306,  9684,  7884, 10211, 15177,
#          26421,   387, 17339,  7889,  9908, 15768,  6903, 15386,  8146, 12923,
#           9228, 18651, 42600,  9564, 17764,  9033,  9199, 14441,  7335,  8704,
#          12557, 32030,  9510, 18595,  9025, 10571, 25741, 10599, 13229,  9508,
#           7965,  8425, 33102,  9122, 21240,  9801, 32106, 13579, 12442, 13235,
#          19430,  8022, 12972,  9566, 11178,  9554, 24873,  7198,  9391, 12486,
#           8711,  9346,  7071, 36736,  9693, 12006,  9038, 10279, 36122,  9960,
#           8405, 10826, 18988, 25998,  9292,  7671,  9465,  7489,  9277, 10137,
#           9677,  9248,  9912, 12834, 11488, 13417,  7407,  8428,  8137,  9430,
#          14222, 11356, 10061,  9885, 19265,  9377, 20305,  7991,  9178,  9648,
#           9133, 10021, 10138, 30315, 21833,  9362,  9301,  9685, 11584,  9447,
#          42129, 10124,  7532, 17932, 47123, 37544,  9355, 15632,  9124, 10536,
#          13530, 12204,  9184, 36152,  9673,  9788,  9029, 11764]])
# gen_ids torch.Size([1, 128])

generated = tokenizer.decode(gen_ids[0])
# print(generated)
# 근육이 커지기 위해서는 무엇보다 규칙적인 생활습관이 중요하다.
# 특히, 아침식사는 단백질과 비타민이 풍부한 과일과 채소를 많이 섭취하는 것이 좋다.
# 또한 하루 30분 이상 충분한 수면을 취하는 것도 도움이 된다.
# 아침 식사를 거르지 않고 규칙적으로 운동을 하면 혈액순환에 도움을 줄 뿐만 아니라 신진대사를 촉진해 체내 노폐물을 배출하고 혈압을 낮춰준다.
# 운동은 하루에 10분 정도만 하는 게 좋으며 운동 후에는 반드시 스트레칭을 통해 근육량을 늘리고 유연성을 높여야 한다.
# 운동 후 바로 잠자리에 드는 것은 피해야 하며 특히 아침에 일어나면 몸이 피곤해지기 때문에 무리하게 움직이면 오히려 역효과가 날 수도 있다.
# 운동을

# ================================================================================
# How to use pretrained GPT-2 by pipeline

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
generation_args = dict(
    num_beams=4,
    repetition_penalty=2.0,
    no_repeat_ngram_size=4,
    eos_token_id=375, # \n
    max_new_tokens=64,
    do_sample=True,
    top_k=50,
    early_stopping=True
)

# 0:, \n, 1: are needed
aa=generator(
    ["0 : **는 게임 좋아하니\n1 :",
     "0 : 어제 강남에서 살인사건 났대 ㅜㅜ 너무 무서워\n1 : 헐 왜? 무슨 일 있었어?\n0 : 사진보니까 막 피흘리는 사람있고 경찰들이 떠서 제압하고 난리도 아니었다던데??\n1 :",
     "0 : 자기야 어제는 나한테 왜 그랬어?\n1 : 뭔 일 있었어?\n0 : 어떻게 나한테 말도 없이 그럴 수 있어? 나 진짜 실망했어\n1 : "], **generation_args)

# ================================================================================
for one in aa:
    print(one[0]['generated_text'])
    print()

# 0 : **는 게임 좋아하니
# 1 : ***는 게임 못하는데


# 0 : 어제 강남에서 살인사건 났대 ㅜㅜ 너무 무서워
# 1 : 헐 왜? 무슨 일 있었어?
# 0 : 사진보니까 막 피흘리는 사람있고 경찰들이 떠서 제압하고 난리도 아니었다던데??
# 1 : 아까도 말했지만 이번에 또 살인을 저질러서 진짜 죄송합니다 ᄏᄏᄏ


# 0 : 자기야 어제는 나한테 왜 그랬어?
# 1 : 뭔 일 있었어?
# 0 : 어떻게 나한테 말도 없이 그럴 수 있어? 나 진짜 실망했어
# 1 : 뭘 했어?
# 2 : 뭘 하고 싶었어?
# 3 : 무슨 일이 있었어?
# 4 : 뭐라고 했어?
# 5 : 뭐라고 말했어?
# 6 : 뭐라고 말해?
# 7 : 뭐라고 말하지 않았어?
# 8 : 뭐라고 하지 않았어?
# 9 : 뭐라고 말한 거야