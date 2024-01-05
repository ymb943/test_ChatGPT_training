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

print(args)

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
# Load GPT-2 model which will be performed SFT

model = AutoModelForCausalLM.from_pretrained(args.model_name)
# print('model',model)
# GPT2LMHeadModel(
#   (transformer): GPT2Model(
#     (wte): Embedding(51200, 768)
#     (wpe): Embedding(1024, 768)
#     (drop): Dropout(p=0.1, inplace=False)
#     (h): ModuleList(
#       (0): GPT2Block(
#         (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
#         (attn): GPT2Attention(
#           (c_attn): Conv1D()
#           (c_proj): Conv1D()
#           (attn_dropout): Dropout(p=0.1, inplace=False)
#           (resid_dropout): Dropout(p=0.1, inplace=False)
#         )
#         ...
#     (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
#   )
#   (lm_head): Linear(in_features=768, out_features=51200, bias=False)
# )

tokenizer = transformers.AutoTokenizer.from_pretrained(
    args.model_name,
    padding_side="right",
    model_max_length=512,
)

tokenizer.add_special_tokens(
    {
        "eos_token": DEFAULT_EOS_TOKEN,
        "bos_token": DEFAULT_BOS_TOKEN,
        "unk_token": DEFAULT_UNK_TOKEN,
    }
)

tokenizer.pad_token = tokenizer.eos_token

# print(tokenizer)

# ================================================================================
class SFT_dataset(Dataset):
    '''SFT dataset by wygo'''
    def __init__(self, data_path_1_SFT: str, tokenizer: transformers.PreTrainedTokenizer, verbose=False):
        super(SFT_dataset, self).__init__()
        logging.warning("Loading data...")

        # ================================================================================
        pattern_instruction = 'prompt'  # instruction
        pattern_input = 'input'  # 내 데이터엔 input이 없다
        pattern_output = 'completion'  # output

        # ================================================================================
        # Load dataset for SFT
        
        # print(data_path_1_SFT)
        # ./data_kochatgpt/kochatgpt_1_SFT.jsonl

        # [
        #     {
        #         "prompt": "불고기용 고기 한우에요?",
        #         "completion": "'저는 인공지능 챗봇이며, 직접적으로 식품에 관한 정보를 가지고 있지 않습니다. 하지만 일반적으로 불고기용 고기는 한우, 쇠고기, 돼지고기 등 다양한 종류의 고기를 사용합니다. 하지만 한우는 대표적인 고급 육류로 알려져 있기 때문에, 한우를 사용하는 경우도 많습니다. 알러지나 개별 건강 상태에 따라 다를 수 있으니 충분한 정보 수집 후에 선택해 주시기 바랍니다.",
        #         "tokens": 193
        #     },...
        # ]
        
        with open(data_path_1_SFT, "r", encoding='utf-8-sig') as json_file:
            list_data_dict = json.load(json_file)
            if verbose:
                print('## data check ##')
                print((list_data_dict[0]))

        # ================================================================================
        # Generate formatted dataset
        
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        # print('prompt_input',prompt_input)
        # print('prompt_no_input',prompt_no_input)
        # ------------------------------------------------------
        # prompt_input
        # ### Instruction(명령어):
        # {prompt}

        # ### Input(입력):
        # {input}

        # ### Response(응답):
        
        # ------------------------------------------------------
        # prompt_no_input
        # ### Instruction(명령어):
        # {prompt}

        # ### Response(응답):

        # ================================================================================
        # Collects question texts
        
        sources = []
        for example in list_data_dict:
            # print('example',example)
            # {'prompt': '불고기용 고기 한우에요?', 
            #  'completion': "'저는 인공지능 챗봇이며, 직접적으로 식품에 관한 정보를 가지고 있지 않습니다. 하지만 일반적으로 불고기용 고기는 한우, 쇠고기, 돼지고기 등 다양한 종류의 고기를 사용합니다. 하지만 한우는 대표적인 고급 육류로 알려져 있기 때문에, 한우를 사용하는 경우도 많습니다. 알러지나 개별 건강 상태에 따라 다를 수 있으니 충분한 정보 수집 후에 선택해 주시기 바랍니다.", 
            #  'tokens': 193}

            # print('example.get(pattern_input, "")',example.get(pattern_input, ""))
            # ""
            
            if example.get(pattern_input, "") != "":
                tmp = prompt_input.format_map(example)
            else:
                tmp = prompt_no_input.format_map(example)
                # print('tmp',tmp)
                # ### Instruction:
                # 불고기용 고기 한우에요?

                # ### Response:
            
            sources.append(tmp)
        
        # print('sources',sources[:2])
        # sources ['### Instruction:\n불고기용 고기 한우에요?\n\n### Response:', '### Instruction:\n쓰던 앱이 유료로 전환됐어\n\n### Response:']
        
        # ================================================================================
        # Collects answer texts
        
        targets = []
        for example in list_data_dict:
            targets.append(f"{example[pattern_output]}{tokenizer.eos_token}")

        if verbose:
            idx = 0
            print((sources[idx]))
            print((targets[idx]))
            print("Tokenizing inputs... This may take some time...")

        # print('targets',targets[:2])
        # targets ["'저는 인공지능 챗봇이며, 직접적으로 식품에 관한 정보를 가지고 있지 않습니다. 하지만 일반적으로 불고기용 고기는 한우, 쇠고기, 돼지고기 등 다양한 종류의 고기를 사용합니다. 하지만 한우는 대표적인 고급 육류로 알려져 있기 때문에, 한우를 사용하는 경우도 많습니다. 알러지나 개별 건강 상태에 따라 다를 수 있으니 충분한 정보 수집 후에 선택해 주시기 바랍니다.</s>", "'어떤 앱인지 모르기 때문에 정확한 답변을 드리기 어렵습니다. 하지만, 일반적으로 유료 전환된 앱은 기존 무료 앱에서 추가적인 기능이나 서비스를 제공하는 경우가 많습니다. 이 경우, 유료 전환 이전 가격이 매우 저렴하거나 무료인 경우가 많으므로, 앱을 계속 사용하려면 추가적인 비용을 지불해야 할 수 있습니다. 하지만, 유료 전환 이후에는 기존 사용자 또는 새로운 사용자 모두가 추가 기능과 높은 품질의 서비스를 이용할 수 있게 됩니다. 따라서, 앱 개발자는 유료 모델로 전환함으로써 앱의 수익을 증가시키고 더 나은 서비스를 제공할 수 있습니다.</s>"]

        # ================================================================================

        examples = [s + '\n' +t for s, t in zip(sources, targets)]
        # print('examples',examples[0])
        # print('====================')
        # print('examples',examples[1])
        # print('====================')
        # print('examples',examples[2])

        # # ====================
        # ### Instruction:
        # 불고기용 고기 한우에요?

        # ### Response:
        # '저는 인공지능 챗봇이며, 직접적으로 식품에 관한 정보를 가지고 있지 않습니다. 하지만 일반적으로 불고기용 고기는 한우, 쇠고기, 돼지고기 등 다양한 종류의 고기를 사용합니다. 하지만 한우는 대표적인 고급 육류로 알려져 있기 때문에, 한우를 사용하는 경우도 많습니다. 알러지나 개별 건강 상태에 따라 다를 수 있으니 충분한 정보 수집 후에 선택해 주시기 바랍니다.</s>
        
        # # ====================
        # ### Instruction:
        # 쓰던 앱이 유료로 전환됐어

        # ### Response:
        # '어떤 앱인지 모르기 때문에 정확한 답변을 드리기 어렵습니다. 하지만, 일반적으로 유료 전환된 앱은 기존 무료 앱에서 추가적인 기능이나 서비스를 제공하는 경우가 많습니다. 이 경우, 유료 전환 이전 가격이 매우 저렴하거나 무료인 경우가 많으므로, 앱을 계속 사용하려면 추가적인 비용을 지불해야 할 수 있습니다. 하지만, 유료 전환 이후에는 기존 사용자 또는 새로운 사용자 모두가 추가 기능과 높은 품질의 서비스를 이용할 수 있게 됩니다. 따라서, 앱 개발자는 유료 모델로 전환함으로써 앱의 수익을 증가시키고 더 나은 서비스를 제공할 수 있습니다.</s>
        
        # # ====================
        # ### Instruction:
        # 여친이랑 다툼

        # ### Response:
        # '저는 인공지능이기 때문에 여친과의 다툼에 대해서는 조언을 드릴 수는 없습니다. 하지만 다투게 되면 상대방의 말을 경청하는 태도와 서로의 감정을 존중하는 것이 중요합니다. 논리적이고 차분한 대화를 통해 문제를 해결해보세요. 그리고 서로를 이해하고 서로의 의견을 수용하는 것이 중요합니다.</s>
        
        # ================================================================================
        # Perform tokenization

        sources_tokenized = self._tokenize_fn(sources, tokenizer)  # question 만
        examples_tokenized = self._tokenize_fn(examples, tokenizer)  # question + answer
        # print(sources_tokenized.keys())
        # dict_keys(['input_ids', 'labels', 'input_ids_lens', 'labels_lens'])
        # print(examples_tokenized.keys())
        # dict_keys(['input_ids', 'labels', 'input_ids_lens', 'labels_lens'])

        # print(sources_tokenized['input_ids'][13])
        # print(sources_tokenized['labels'][13])
        # print(sources_tokenized['input_ids_lens'][13])
        # print(sources_tokenized['labels_lens'][13])
        # tensor([  739, 378, 378, 378, 14659, 13394, 37091, 10651, 401, 375, 7071,  6866, 15933, 10348,  9098,  8245, 10030,  9677,  6872,  9017,  375, 378, 378, 378, 41951, 454,  9549, 20549, 401])
        # tensor([  739, 378, 378, 378, 14659, 13394, 37091, 10651, 401, 375, 7071,  6866, 15933, 10348,  9098,  8245, 10030,  9677,  6872,  9017,  375, 378, 378, 378, 41951, 454,  9549, 20549, 401])
        # 29
        # 29

        # print(examples_tokenized['input_ids'][13])
        # print(examples_tokenized['labels'][13])
        # print(examples_tokenized['input_ids_lens'][13])
        # print(examples_tokenized['labels_lens'][13])
        # tensor([  739, 378, 378, 378, 14659, 13394, 37091, 10651, 401, 375, 7071,  6866, 15933, 10348,  9098,  8245, 10030,  9677,  6872,  9017,  375, 378, 378, 378, 41951, 454,  9549, 20549, 401, 375,  382, 10072, 35673,  9122,  7162,  9199, 14444,  9154, 13362, 13154, 7281,  7481,  9025,  9080,  9362,  9558, 10276,  9033,  9306, 11683, 10345, 12399, 32240, 10795, 27540, 25287, 25226, 15806,  9018,  7521, 12191, 11219,  7788, 43248, 25741,  9199,  9036,  9340,  9597, 21154, 9323,  9026, 12504,  9939, 11667,  7788, 14240, 10106,  9314, 15933, 8570,  9823,  9277, 10599, 10586, 15517, 10341,  9025, 32987,  9650, 9362, 21031, 10043,  9077, 12181, 19870, 13204,  9394, 12610,  9036, 9919,  9134, 24352, 11683, 14981, 32240, 1])
        # tensor([  739, 378, 378, 378, 14659, 13394, 37091, 10651, 401, 375, 7071,  6866, 15933, 10348,  9098,  8245, 10030,  9677,  6872,  9017,  375, 378, 378, 378, 41951, 454,  9549, 20549, 401, 375,  382, 10072, 35673,  9122,  7162,  9199, 14444,  9154, 13362, 13154, 7281,  7481,  9025,  9080,  9362,  9558, 10276,  9033,  9306, 11683, 10345, 12399, 32240, 10795, 27540, 25287, 25226, 15806,  9018,  7521, 12191, 11219,  7788, 43248, 25741,  9199,  9036,  9340,  9597, 21154, 9323,  9026, 12504,  9939, 11667,  7788, 14240, 10106,  9314, 15933, 8570,  9823,  9277, 10599, 10586, 15517, 10341,  9025, 32987,  9650, 9362, 21031, 10043,  9077, 12181, 19870, 13204,  9394, 12610,  9036, 9919,  9134, 24352, 11683, 14981, 32240, 1])
        # 106
        # 106

        # ================================================================================
        # 입력은 source, 출력은 source+target 이지만 학습은 target 부분만

        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX  # source 부분은 -100으로 채운다

        data_dict = dict(input_ids=input_ids, labels=labels)
        # print(data_dict['input_ids'][13])      # Question text tokens + Answer text tokens
        # print(data_dict['labels'][13])         # Question text tokens (-100) + Answer text tokens
        # tensor([  739, 378, 378, 378, 14659, 13394, 37091, 10651, 401, 375, 7071,  6866, 15933, 10348,  9098,  8245, 10030,  9677,  6872,  9017,  375, 378, 378, 378, 41951, 454,  9549, 20549, 401, 375,  382, 10072, 35673,  9122,  7162,  9199, 14444,  9154, 13362, 13154, 7281,  7481,  9025,  9080,  9362,  9558, 10276,  9033,  9306, 11683, 10345, 12399, 32240, 10795, 27540, 25287, 25226, 15806,  9018,  7521, 12191, 11219,  7788, 43248, 25741,  9199,  9036,  9340,  9597, 21154, 9323,  9026, 12504,  9939, 11667,  7788, 14240, 10106,  9314, 15933, 8570,  9823,  9277, 10599, 10586, 15517, 10341,  9025, 32987,  9650, 9362, 21031, 10043,  9077, 12181, 19870, 13204,  9394, 12610,  9036, 9919,  9134, 24352, 11683, 14981, 32240, 1])
        # tensor([ -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100, -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100, -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100, 375,  382, 10072, 35673,  9122,  7162,  9199, 14444,  9154, 13362, 13154, 7281,  7481,  9025,  9080,  9362,  9558, 10276,  9033,  9306, 11683, 10345, 12399, 32240, 10795, 27540, 25287, 25226, 15806,  9018,  7521, 12191, 11219,  7788, 43248, 25741,  9199,  9036,  9340,  9597, 21154, 9323,  9026, 12504,  9939, 11667,  7788, 14240, 10106,  9314, 15933, 8570,  9823,  9277, 10599, 10586, 15517, 10341,  9025, 32987,  9650, 9362, 21031, 10043,  9077, 12181, 19870, 13204,  9394, 12610,  9036, 9919,  9134, 24352, 11683, 14981, 32240, 1])

        # ================================================================================
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        logging.warning("Loading data done!!: %d"%(len(self.labels)))

    def _tokenize_fn(self, strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
        """Tokenize a list of strings."""
        tokenized_list = [
            tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
            for text in strings
        ]
        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

# ================================================================================
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        # print('input_ids',input_ids)    # Question text token IDs + Answer text token IDs
        # print('labels',labels)          # Question text token IDs (replaced with -100) + Answer text token IDs
        # input_ids [tensor([  739,   378,   378,   378, 14659, 13394, 37091, 10651,   401,   375,
        #         7756,  6903, 21634,  9306, 19787,  7235, 13815, 15084,  7801, 12002,
        #           375,   378,   378,   378, 41951,   454,  9549, 20549,   401,   375,
        #           382, 25294, 50033,  9571, 12443, 11510, 15542, 13883,  8518,  9185,
        #         35119, 36568,  9030,  9366,  9242,  9556, 22125, 12011, 25856,   434,
        #           452, 30857, 50900,  9571,  9271,  8471,  9038,  9065, 13501, 11355,
        #         9355,  9226, 10820,  9677,  9496,  9046, 42305,   739,  8174,  9030,
        #         9882, 12650, 10458,  9826,  8711, 25856,   434,   452, 36045, 24708,
        #         9571, 15386,  9752,  6824, 29611, 19384, 32739,  9325, 11000, 10018,
        #         8022, 25856,  9046, 10400, 17624,  9313, 37428, 35119, 36568,  9030,
        #         9882,  9242,  9556,  9209,  9826, 37194,   434,   452, 34616, 31973,
        #         40827,  9571, 15386,  9752,   387, 21302,   387,  9410,  7671,  9225,
        #         29611, 14931,  9325,  9677, 16691,  9046, 12328, 35119,  7397, 17203,
        #         9206,  7605,  9030,  9882,  9242,  9556,  9209,  9826, 37194,   434,
        #           452, 37653, 22483, 23012,  9571, 41845, 27726, 34300, 14271,  9341,
        #         9847, 35943,  9677,  9248, 12095,  7395,  9325, 13229, 37671, 12521,
        #         11141,  7301,   387,  9230,  7556, 43068, 11244, 13503,   387, 27752,
        #         9030,  9882, 10133,  8185, 17053,  9209,  9826, 37194,     1])]
        # labels [tensor([ -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
        #         -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
        #         -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,   375,
        #           382, 25294, 50033,  9571, 12443, 11510, 15542, 13883,  8518,  9185,
        #         35119, 36568,  9030,  9366,  9242,  9556, 22125, 12011, 25856,   434,
        #           452, 30857, 50900,  9571,  9271,  8471,  9038,  9065, 13501, 11355,
        #         9355,  9226, 10820,  9677,  9496,  9046, 42305,   739,  8174,  9030,
        #         9882, 12650, 10458,  9826,  8711, 25856,   434,   452, 36045, 24708,
        #         9571, 15386,  9752,  6824, 29611, 19384, 32739,  9325, 11000, 10018,
        #         8022, 25856,  9046, 10400, 17624,  9313, 37428, 35119, 36568,  9030,
        #         9882,  9242,  9556,  9209,  9826, 37194,   434,   452, 34616, 31973,
        #         40827,  9571, 15386,  9752,   387, 21302,   387,  9410,  7671,  9225,
        #         29611, 14931,  9325,  9677, 16691,  9046, 12328, 35119,  7397, 17203,
        #         9206,  7605,  9030,  9882,  9242,  9556,  9209,  9826, 37194,   434,
        #           452, 37653, 22483, 23012,  9571, 41845, 27726, 34300, 14271,  9341,
        #         9847, 35943,  9677,  9248, 12095,  7395,  9325, 13229, 37671, 12521,
        #         11141,  7301,   387,  9230,  7556, 43068, 11244, 13503,   387, 27752,
        #         9030,  9882, 10133,  8185, 17053,  9209,  9826, 37194,     1])]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        # print('input_ids',input_ids)
        # print('input_ids',input_ids.shape)
        # input_ids tensor([[  739,   378,   378,   378, 14659, 13394, 37091, 10651,   401,   375,
        #           7756,  6903, 21634,  9306, 19787,  7235, 13815, 15084,  7801, 12002,
        #           375,   378,   378,   378, 41951,   454,  9549, 20549,   401,   375,
        #           382, 25294, 50033,  9571, 12443, 11510, 15542, 13883,  8518,  9185,
        #         35119, 36568,  9030,  9366,  9242,  9556, 22125, 12011, 25856,   434,
        #           452, 30857, 50900,  9571,  9271,  8471,  9038,  9065, 13501, 11355,
        #           9355,  9226, 10820,  9677,  9496,  9046, 42305,   739,  8174,  9030,
        #           9882, 12650, 10458,  9826,  8711, 25856,   434,   452, 36045, 24708,
        #           9571, 15386,  9752,  6824, 29611, 19384, 32739,  9325, 11000, 10018,
        #           8022, 25856,  9046, 10400, 17624,  9313, 37428, 35119, 36568,  9030,
        #           9882,  9242,  9556,  9209,  9826, 37194,   434,   452, 34616, 31973,
        #         40827,  9571, 15386,  9752,   387, 21302,   387,  9410,  7671,  9225,
        #         29611, 14931,  9325,  9677, 16691,  9046, 12328, 35119,  7397, 17203,
        #           9206,  7605,  9030,  9882,  9242,  9556,  9209,  9826, 37194,   434,
        #           452, 37653, 22483, 23012,  9571, 41845, 27726, 34300, 14271,  9341,
        #           9847, 35943,  9677,  9248, 12095,  7395,  9325, 13229, 37671, 12521,
        #         11141,  7301,   387,  9230,  7556, 43068, 11244, 13503,   387, 27752,
        #           9030,  9882, 10133,  8185, 17053,  9209,  9826, 37194,     1]])
        # input_ids torch.Size([1, 179])

        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        # print('labels',labels)
        # print('labels',labels.shape)
        # labels tensor([[ -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
        #           -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
        #           -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,   375,
        #           382, 25294, 50033,  9571, 12443, 11510, 15542, 13883,  8518,  9185,
        #         35119, 36568,  9030,  9366,  9242,  9556, 22125, 12011, 25856,   434,
        #           452, 30857, 50900,  9571,  9271,  8471,  9038,  9065, 13501, 11355,
        #           9355,  9226, 10820,  9677,  9496,  9046, 42305,   739,  8174,  9030,
        #           9882, 12650, 10458,  9826,  8711, 25856,   434,   452, 36045, 24708,
        #           9571, 15386,  9752,  6824, 29611, 19384, 32739,  9325, 11000, 10018,
        #           8022, 25856,  9046, 10400, 17624,  9313, 37428, 35119, 36568,  9030,
        #           9882,  9242,  9556,  9209,  9826, 37194,   434,   452, 34616, 31973,
        #         40827,  9571, 15386,  9752,   387, 21302,   387,  9410,  7671,  9225,
        #         29611, 14931,  9325,  9677, 16691,  9046, 12328, 35119,  7397, 17203,
        #           9206,  7605,  9030,  9882,  9242,  9556,  9209,  9826, 37194,   434,
        #           452, 37653, 22483, 23012,  9571, 41845, 27726, 34300, 14271,  9341,
        #           9847, 35943,  9677,  9248, 12095,  7395,  9325, 13229, 37671, 12521,
        #         11141,  7301,   387,  9230,  7556, 43068, 11244, 13503,   387, 27752,
        #           9030,  9882, 10133,  8185, 17053,  9209,  9826, 37194,     1]])
        # labels torch.Size([1, 179])

        attn_mask=input_ids.ne(self.tokenizer.pad_token_id)
        # print('attn_mask',attn_mask)
        # print('attn_mask',attn_mask.shape)
        # attn_mask tensor([[ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #           True,  True,  True,  True,  True,  True,  True,  True, False]])
        # attn_mask torch.Size([1, 179])

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

# ================================================================================
# print('args.data_path_1_SFT',args.data_path_1_SFT)
# args.data_path_1_SFT ./data_kochatgpt/kochatgpt_1_SFT.jsonl

train_dataset = SFT_dataset(data_path_1_SFT=args.data_path_1_SFT, tokenizer=tokenizer)
eval_dataset  = None  # eval은 안함
data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

# check
# print('input : %s'%train_dataset.input_ids[0])
# print('output: %s'%train_dataset.labels[0])

# ================================================================================
# Perform SFT

# training_args : https://github.com/Beomi/KoAlpaca/blob/main/train.sh 참고
training_args = TrainingArguments(
    output_dir="./test",            # The output directory
    overwrite_output_dir=True,      # overwrite the content of the output directory
    num_train_epochs=1,             # number of training epochs
    # per_device_train_batch_size=4,  # batch size for training
    per_device_train_batch_size=1,  # batch size for training
    per_device_eval_batch_size=4,   # batch size for evaluation
    eval_steps = 3,                 # Number of update steps between two evaluations.
    save_steps=500,                 # after # steps model is saved
    warmup_steps=5,                 # number of warmup steps for learning rate scheduler
    prediction_loss_only=True)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset)

# trainer.train()
# trainer.save_state()
# safe_save_model_for_hf_trainer(trainer=trainer, output_dir=args.output_dir)

# ================================================================================
# Inference test by SFT performed model

generator = pipeline('text-generation', model=args.output_dir, tokenizer=tokenizer)
# generator = pipeline('text-generation', model=model.cpu(), tokenizer=tokenizer, config={'max_length':800})

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

# Question texts
list_prompt = ['컴퓨터가 발명된 연도는?',
               '인공지능에 대해 설명해줘']
list_prompt = [PROMPT_DICT['prompt_no_input'].format_map({'prompt' : tmp}) for tmp in list_prompt]
# print('list_prompt',list_prompt)
# list_prompt ['### Instruction:\n컴퓨터가 발명된 연도는?\n\n### Response:', '### Instruction:\n인공지능에 대해 설명해줘\n\n### Response:']

# Answer texts
list_result = generator(list_prompt, **generation_args)
for prompt, result in zip(list_prompt, list_result):
    print('\n','#'*70)
    print(('completion: %s'%(result[0]['generated_text'])))

#  ######################################################################
# ### Instruction:
# 컴퓨터가 발명된 연도는?

# ### Response:'컴퓨터가 발명 된 연도는 1930년입니다. 컴퓨터는 20세기 초반부터 20세기 초반까지 발전했습니다. 컴퓨터는 다양한 분야에서 사용되기 시작했습니다. 예를 들면, 스마트폰, 태블릿, TV, 노트북 등 다양한 분야에서 사용되었습니다. 컴퓨터가 발명한 연도는 다음과 같습니다.\n\n1. PC: 19세기 후반부터

#  ######################################################################
# ### Instruction:
# 인공지능에 대해 설명해줘

# ### Response:'저는 인공지능 언어모델로써 인간의 질문에 대한 답변을 제공할 수 없습니다. 하지만 인공지능은 다양한 분야에서 활용되고 있습니다. 예를 들면, 로봇, 컴퓨터, 소프트웨어 등 다양한 분야에서 활용됩니다.\n\n또한, 인공지능 기술은 여러 분야에서 발전하고 있으며, 이를 통해 많은 사람들이 새로운 경험을 쌓을 수 있습니다.
