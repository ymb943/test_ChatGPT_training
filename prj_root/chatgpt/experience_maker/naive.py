import torch
from chatgpt.models.utils import compute_reward, normalize

from .base import Experience, ExperienceMaker


class NaiveExperienceMaker(ExperienceMaker):
    """
    Naive experience maker.
    """

    @torch.no_grad()
    def make_experience(self, input_ids: torch.Tensor, **generate_kwargs) -> Experience:
        
        # ================================================================================
        # Configure as evaluation mode because this is just experience generation process

        self.actor.eval()
        self.critic.eval()
        self.initial_model.eval()
        self.reward_model.eval()

        # ================================================================================
        # print('self.actor.generate',type(self.actor.generate))
        # self.actor.generate <class 'method'>
        # print('self.actor.generate',self.actor.generate)

        # /data/hdd2/user/ympark/2023/12/01_chatgpt/prj_root/chatgpt/models/base/actor.py
        sequences, attention_mask, action_mask = self.actor.generate(input_ids,
                                                                     return_action_mask=True,
                                                                     **generate_kwargs)
        
        # print('sequences',sequences.shape)
        # print('attention_mask',attention_mask.shape)
        # print('action_mask',action_mask.shape)
        # sequences torch.Size([8, 128])
        # attention_mask torch.Size([8, 128])
        # action_mask torch.Size([8, 107])
        
        # ================================================================================
        num_actions = action_mask.size(1)
        # print('num_actions',num_actions)
        # num_actions 107

        # torch.set_printoptions(profile='default')
        torch.set_printoptions(threshold=6)
        
        # ================================================================================
        # print('self.actor',self.actor)
        # GPTActor(
        #   (model): GPT2LMHeadModel(
        #     (transformer): GPT2Model(
        #       (wte): Embedding(51200, 768)
        #       (wpe): Embedding(1024, 768)
        #       (drop): Dropout(p=0.1, inplace=False)
        #       (h): ModuleList(
        #       ...
        #     (lm_head): Linear(in_features=768, out_features=51200, bias=False)
        #   )
        # )

        # Traiable Actor model (same structure with SFT trained initial_model)
        action_log_probs = self.actor(sequences, num_actions, attention_mask)
        # print('action_log_probs',action_log_probs)
        # print('action_log_probs',action_log_probs.shape)
        # tensor([[-6.3748, -0.3850, -5.2702,  ..., -5.2411, -4.6299, -4.6946],
        #         [-2.4233, -0.1502, -0.0315,  ..., -5.3831, -4.7305, -4.7086],
        #         [-2.9520, -1.2308, -1.8419,  ..., -6.3464, -5.6861, -5.7310],
        #         ...,
        #         [-4.2715, -0.8264, -0.1740,  ..., -7.6705, -7.0685, -7.0264],
        #         [-2.4117, -0.5632, -1.3264,  ..., -0.1638, -0.3451, -0.3214],
        #         [-5.4037, -1.6125, -5.6595,  ..., -6.4581, -6.0071, -5.7032]], device='cuda:0')
        # action_log_probs torch.Size([8, 107])
        
        # ================================================================================
        # print('self.initial_model',self.initial_model)
        # GPTActor(
        #   (model): GPT2LMHeadModel(
        #     (transformer): GPT2Model(
        #       (wte): Embedding(51200, 768)
        #       (wpe): Embedding(1024, 768)
        #       (drop): Dropout(p=0.1, inplace=False)
        #       (h): ModuleList(
        #       ...
        #     (lm_head): Linear(in_features=768, out_features=51200, bias=False)
        #   )
        # )
        
        # Untraiable inital model which had been trained by SFT
        base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)
        # print('base_action_log_probs',base_action_log_probs)
        # print('base_action_log_probss',base_action_log_probs.shape)
        # tensor([[-6.3748, -0.3850, -5.2702,  ..., -5.2411, -4.6299, -4.6946],
        #         [-2.4233, -0.1502, -0.0315,  ..., -5.3831, -4.7305, -4.7086],
        #         [-2.9520, -1.2308, -1.8419,  ..., -6.3464, -5.6861, -5.7310],
        #         ...,
        #         [-4.2715, -0.8264, -0.1740,  ..., -7.6705, -7.0685, -7.0264],
        #         [-2.4117, -0.5632, -1.3264,  ..., -0.1638, -0.3451, -0.3214],
        #         [-5.4037, -1.6125, -5.6595,  ..., -6.4581, -6.0071, -5.7032]], device='cuda:0')
        # base_action_log_probss torch.Size([8, 107])

        # ================================================================================
        # Use trainable critic model which criticizes Actor output "sequences"

        # print('self.critic',self.critic)
        # self.critic GPTCritic(
        #   (model): GPT2Model(
        #     (wte): Embedding(51201, 768)
        #     (wpe): Embedding(1024, 768)
        #     (drop): Dropout(p=0.1, inplace=False)
        #     (h): ModuleList(
        #   ...
        #   (value_head): Linear(in_features=768, out_features=1, bias=True)
        # )

        value = self.critic(sequences, action_mask, attention_mask)
        # print('value',value)
        # print('value',value.shape)
        # value tensor([0.2177, 0.1005, 0.2401,  ..., 0.2930, 0.1571, 0.0807], device='cuda:0')
        # value torch.Size([8])
        
        # ================================================================================
        # Untraiable reward model which had been trained by ranking labels

        # print('self.reward_model',self.reward_model)

        # RewardModel(
        #   (model): GPT2Model(
        #     (wte): Embedding(51201, 768)
        #     (wpe): Embedding(1024, 768)
        #     (drop): Dropout(p=0.1, inplace=False)
        #     (h): ModuleList(
        #    ...
        #   (value_head): Linear(in_features=768, out_features=1, bias=True)
        # )

        r = self.reward_model(sequences, attention_mask)
        # print('r',r)
        # print('r',r.shape)
        # r tensor([0.2177, 0.1005, 0.2401,  ..., 0.2930, 0.1571, 0.0807], device='cuda:0')
        # r torch.Size([8])

        # ================================================================================
        # Final reward for PPO training
        
        reward = compute_reward(r, self.kl_coef, action_log_probs, base_action_log_probs, action_mask=action_mask)
        # print('reward',reward)
        # print('reward',reward.shape)
        # reward tensor([0.2177, 0.1005, 0.2401,  ..., 0.2930, 0.1571, 0.0807], device='cuda:0')
        # reward torch.Size([8])

        # ================================================================================
        advantage = reward - value
        # print('advantage',advantage)
        # advantage tensor([0., 0., 0.,  ..., 0., 0., 0.], device='cuda:0')

        # TODO(ver217): maybe normalize adv
        if advantage.ndim == 1:
            advantage = advantage.unsqueeze(-1)

        return Experience(sequences, action_log_probs, value, reward, advantage, attention_mask, action_mask)
