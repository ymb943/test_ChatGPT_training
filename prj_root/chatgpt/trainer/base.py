from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from chatgpt.experience_maker import Experience, ExperienceMaker
from chatgpt.replay_buffer import ReplayBuffer
from torch import Tensor
from torch.utils.data import DistributedSampler
from tqdm import tqdm

from .callbacks import Callback
from .strategies import Strategy
from .utils import is_rank_0


class Trainer(ABC):
    """
        Base class for rlhf trainers.

    Args:
        strategy (Strategy):the strategy to use for training
        experience_maker (ExperienceMaker): the experience maker to use for produce experience to fullfill replay buffer
        replay_buffer (ReplayBuffer): the replay buffer to use for training
        experience_batch_size (int, defaults to 8): the batch size to use for experience generation
        max_epochs (int, defaults to 1): the number of epochs of training process
        tokenizer (Callable, optional): the tokenizer to use for tokenizing the input
        sample_replay_buffer (bool, defaults to False): whether to sample from replay buffer
        data_loader_pin_memory (bool, defaults to True): whether to pin memory for data loader
        callbacks (List[Callback], defaults to []): the callbacks to call during training process
        generate_kwargs (dict, optional): the kwargs to use while model generating
    """

    def __init__(self,
                 strategy: Strategy,
                 experience_maker: ExperienceMaker,
                 replay_buffer: ReplayBuffer,
                 experience_batch_size: int = 8,
                 max_epochs: int = 1,
                 tokenizer: Optional[Callable[[Any], dict]] = None,
                 sample_replay_buffer: bool = False,
                 dataloader_pin_memory: bool = True,
                 callbacks: List[Callback] = [],
                 **generate_kwargs) -> None:
        super().__init__()
        self.strategy = strategy
        self.experience_maker = experience_maker
        self.replay_buffer = replay_buffer
        self.experience_batch_size = experience_batch_size
        self.max_epochs = max_epochs
        self.tokenizer = tokenizer
        self.generate_kwargs = generate_kwargs
        self.sample_replay_buffer = sample_replay_buffer
        self.dataloader_pin_memory = dataloader_pin_memory
        self.callbacks = callbacks

    @abstractmethod
    def training_step(self, experience: Experience) -> Dict[str, Any]:
        pass

    def _make_experience(self, inputs: Union[Tensor, Dict[str, Tensor]]) -> Experience:
        # print('self.experience_maker.make_experience',self.experience_maker.make_experience)
        # <bound method NaiveExperienceMaker.make_experience of <chatgpt.experience_maker.naive.NaiveExperienceMaker object at 0x7fecac3f9880>>

        if isinstance(inputs, Tensor):
            return self.experience_maker.make_experience(inputs, **self.generate_kwargs)
        elif isinstance(inputs, dict):
            return self.experience_maker.make_experience(**inputs, **self.generate_kwargs)
        else:
            raise ValueError(f'Unsupported input type "{type(inputs)}"')

    def _sample_prompts(self, prompts) -> list:
        indices = list(range(len(prompts)))
        sampled_indices = self.strategy.experience_sampler.choice(indices, self.experience_batch_size, replace=False)
        return [prompts[i] for i in sampled_indices]

    def _learn(self):
        
        # print('self.sample_replay_buffer',self.sample_replay_buffer)
        # self.sample_replay_buffer False
        
        # replay buffer may be empty at first, we should rebuild at each training
        if not self.sample_replay_buffer:
            dataloader = self.strategy.setup_dataloader(self.replay_buffer, self.dataloader_pin_memory)
            device = torch.cuda.current_device()
            # print('dataloader',dataloader)
            # print('device',device)
            # dataloader <torch.utils.data.dataloader.DataLoader object at 0x7fbed1f3ff70>
            # device 0
        
        if self.sample_replay_buffer:
            pbar = tqdm(range(self.max_epochs), desc='Train epoch', disable=not is_rank_0())
            for _ in pbar:
                experience = self.replay_buffer.sample()
                metrics = self.training_step(experience)
                pbar.set_postfix(metrics)
        else:
            for epoch in range(self.max_epochs):
                self._on_learn_epoch_start(epoch)
                
                # ================================================================================
                # print('isinstance(dataloader.sampler, DistributedSampler)',isinstance(dataloader.sampler, DistributedSampler))
                # isinstance(dataloader.sampler, DistributedSampler) False

                if isinstance(dataloader.sampler, DistributedSampler):
                    dataloader.sampler.set_epoch(epoch)
                
                # ================================================================================
                pbar = tqdm(dataloader, desc=f'Train epoch [{epoch+1}/{self.max_epochs}]', disable=not is_rank_0())
                
                for experience in pbar: # From DataLoader containing replay_buffer of experiences
                    self._on_learn_batch_start()
                    experience.to_device(device)
                    metrics = self.training_step(experience)
                    # print('metrics',metrics)
                    # metrics {'actor_loss': 0.0, 'critic_loss': 0.005122666247189045}
                    self._on_learn_batch_end(metrics, experience)
                    pbar.set_postfix(metrics)
                self._on_learn_epoch_end(epoch)

    def fit(self, prompts, num_episodes: int = 50000, max_timesteps: int = 500, update_timesteps: int = 5000) -> None:
        time = 0
        
        # print('self.strategy.setup_sampler',self.strategy.setup_sampler)
        # <bound method Strategy.setup_sampler of <chatgpt.trainer.strategies.naive.NaiveStrategy object at 0x7fd6728158e0>>

        sampler = self.strategy.setup_sampler(prompts)
        self._on_fit_start()
        
        # print('num_episodes',num_episodes)
        # num_episodes 1
              
        for episode in range(num_episodes):
            self._on_episode_start(episode)

            # print('max_timesteps',max_timesteps)
            # max_timesteps 3
            
            for timestep in tqdm(range(max_timesteps),
                                 desc=f'Episode [{episode+1}/{num_episodes}]',
                                 disable=not is_rank_0()):
                time += 1
                
                # print('self.experience_batch_size',self.experience_batch_size)
                # self.experience_batch_size 8
                # print('sampler.sample',sampler.sample)
                # <bound method DistributedSampler.sample of <chatgpt.trainer.strategies.sampler.DistributedSampler object at 0x7fd672815700>>
                rand_prompts = sampler.sample(self.experience_batch_size,self.random_prompts)
                # print('rand_prompts',rand_prompts)
                # ['다해서 얼마에요?', '깐 쪽파 있어요?', '초등학생은 추가 비용 있어요?', '기독교대한감리회에서 전래 당시에 조선의 독립을 위해 노력했으며 아펜젤러 선교사가 소속된 교단은?', '이런 조끼는 안에 뭐 들었어요?', '더 답답한 오늘', '알바하면서 썸 많이 타?', '사이마 운하가 건설되던 때에 라펜란타와 비푸리는 어느 나라 땅이였어']
                
                # print('self.tokenizer',self.tokenizer)
                # self.tokenizer <function tokenize_fn at 0x7f54c5c37670>

                if self.tokenizer is not None:
                    inputs = self.tokenizer(rand_prompts)
                else:
                    inputs = rand_prompts
                # print('inputs',inputs)
                # {'input_ids': tensor([[ 9054,  9867, 11474,  8022,  8084,   406,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1],
                #                       [40053, 12789,  8615,  9341,  8084,   406,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1],
                #                       [19785, 12380,  8135, 13119, 13456,  9341,  8084,   406,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1],
                #                       [15215,  9913, 26999, 11886, 15706, 16860, 13833, 16608,  9207, 10805, 10560,  9050,  8645,  8199,  7397, 12749,  9559, 26526,  9099, 11862,406],
                #                       [10165,  9059,  7048,  7162, 11064, 46651,  9625,  8017,  8006,  8084,   406,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1],
                #                       [ 9267, 12817,  7192,  8704, 10070,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1],
                #                       [34790,  9448,   739,  7920,  9564,  9421,   406,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1],
                #                       [ 9435,  7487,  9273, 15723, 10006, 15624, 11435,  9755,  8645,  7374, 32974,  9072,  8671,  9409, 10203,  9602, 22366,  8041,  8006,     1,     1]], device='cuda:0'), 
                #  'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                #                            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                #                            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                #                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                #                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                #                            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                #                            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                #                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]],device='cuda:0')}
                                          
                # print('inputs',inputs['input_ids'].shape)
                # print('inputs',inputs['attention_mask'].shape)
                # inputs torch.Size([8, 21])
                # inputs torch.Size([8, 21])

                # ================================================================================
                self._on_make_experience_start()
                experience = self._make_experience(inputs)
                # print('experience',experience)
                # Experience(
                #   sequences=tensor([[ 9054,  9867, 11474,  ...,     1,     1,     1],
                #                     [40053, 12789,  8615,  ...,     1,     1,     1],
                #                     [19785, 12380,  8135,  ...,     1,     1,     1],
                #                     ...,
                #                     [ 9267, 12817,  7192,  ...,     1,     1,     1],
                #                     [34790,  9448,   739,  ...,  9144, 10846,  9022],
                #                     [ 9435,  7487,  9273,  ...,     1,     1,     1]], device='cuda:0'), 
                #   action_log_probs=tensor([[-6.3748, -0.3850, -5.2702,  ..., -5.2411, -4.6299, -4.6946],
                #                            [-2.4233, -0.1502, -0.0315,  ..., -5.3831, -4.7305, -4.7086],
                #                            [-2.9520, -1.2308, -1.8419,  ..., -6.3464, -5.6861, -5.7310],
                #                            ...,
                #                            [-4.2715, -0.8264, -0.1740,  ..., -7.6705, -7.0685, -7.0264],
                #                            [-2.4117, -0.5632, -1.3264,  ..., -0.1638, -0.3451, -0.3214],
                #                            [-5.4037, -1.6125, -5.6595,  ..., -6.4581, -6.0071, -5.7032]], device='cuda:0'), 
                #   values=tensor([0.2177, 0.1005, 0.2401,  ..., 0.2930, 0.1571, 0.0807], device='cuda:0'), 
                #   reward=tensor([0.2177, 0.1005, 0.2401,  ..., 0.2930, 0.1571, 0.0807], device='cuda:0'), 
                #   advantages=tensor([[0.],[0.],[0.],...,[0.],[0.],[0.]], device='cuda:0'), 
                #   attention_mask=tensor([[1, 1, 1,  ..., 0, 0, 0],
                #                          [1, 1, 1,  ..., 0, 0, 0],
                #                          [1, 1, 1,  ..., 0, 0, 0],
                #                          ...,
                #                          [1, 1, 1,  ..., 0, 0, 0],
                #                          [1, 1, 1,  ..., 1, 1, 1],
                #                          [1, 1, 1,  ..., 0, 0, 0]], device='cuda:0'), 
                #   action_mask=tensor([[ True,  True, False,  ..., False, False, False],
                #                       [ True,  True,  True,  ..., False, False, False],
                #                       [ True,  True,  True,  ..., False, False, False],
                #                       ...,
                #                       [ True,  True,  True,  ..., False, False, False],
                #                       [ True,  True,  True,  ...,  True,  True,  True],
                #                       [ True,  True, False,  ..., False, False, False]], device='cuda:0'))

                # ================================================================================
                # print("experience.sequences.shape",experience.sequences.shape)
                # for one in range(experience.sequences.shape[0]):
                #   temp_torch_tensor=experience.sequences[one,:]
                #   print('temp_torch_tensor[temp_torch_tensor!=1]',temp_torch_tensor[temp_torch_tensor!=1].shape)
                # print("experience.action_log_probs.shape",experience.action_log_probs.shape)
                # print("experience.attention_mask.shape",experience.attention_mask.shape)
                # print("experience.action_mask.shape",experience.action_mask.shape)
                
                # experience.sequences.shape torch.Size([8, 128]) # max_length
                # temp_torch_tensor[temp_torch_tensor!=1] torch.Size([7])
                # temp_torch_tensor[temp_torch_tensor!=1] torch.Size([16])
                # temp_torch_tensor[temp_torch_tensor!=1] torch.Size([18])
                # temp_torch_tensor[temp_torch_tensor!=1] torch.Size([21])
                # temp_torch_tensor[temp_torch_tensor!=1] torch.Size([20])
                # temp_torch_tensor[temp_torch_tensor!=1] torch.Size([75])
                # temp_torch_tensor[temp_torch_tensor!=1] torch.Size([47])
                # temp_torch_tensor[temp_torch_tensor!=1] torch.Size([59])
                # experience.action_log_probs.shape torch.Size([8, 107])
                # experience.attention_mask.shape torch.Size([8, 128])
                # experience.action_mask.shape torch.Size([8, 107])

                self._on_make_experience_end(experience)
                
                # ================================================================================
                # Store experience into replay buffer

                self.replay_buffer.append(experience)
                
                # ================================================================================
                # Store experience, and sometimes perform learning step, then clear the replay buffer of experience

                if time % update_timesteps == 0:
                    self._learn()
                    self.replay_buffer.clear()
            
            self._on_episode_end(episode)
        
        self._on_fit_end()

    # TODO(ver217): maybe simplify these code using context
    def _on_fit_start(self) -> None:
        for callback in self.callbacks:
            callback.on_fit_start()

    def _on_fit_end(self) -> None:
        for callback in self.callbacks:
            callback.on_fit_end()

    def _on_episode_start(self, episode: int) -> None:
        for callback in self.callbacks:
            callback.on_episode_start(episode)

    def _on_episode_end(self, episode: int) -> None:
        for callback in self.callbacks:
            callback.on_episode_end(episode)

    def _on_make_experience_start(self) -> None:
        for callback in self.callbacks:
            callback.on_make_experience_start()

    def _on_make_experience_end(self, experience: Experience) -> None:
        for callback in self.callbacks:
            callback.on_make_experience_end(experience)

    def _on_learn_epoch_start(self, epoch: int) -> None:
        for callback in self.callbacks:
            callback.on_learn_epoch_start(epoch)

    def _on_learn_epoch_end(self, epoch: int) -> None:
        for callback in self.callbacks:
            callback.on_learn_epoch_end(epoch)

    def _on_learn_batch_start(self) -> None:
        for callback in self.callbacks:
            callback.on_learn_batch_start()

    def _on_learn_batch_end(self, metrics: dict, experience: Experience) -> None:
        for callback in self.callbacks:
            callback.on_learn_batch_end(metrics, experience)
