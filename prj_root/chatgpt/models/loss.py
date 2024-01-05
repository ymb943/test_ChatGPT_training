from typing import Optional

import torch
import torch.nn as nn

from .utils import masked_mean


class GPTLMLoss(nn.Module):
    """
    GPT Language Model Loss
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


class PolicyLoss(nn.Module):
    """
    Policy Loss for PPO
    """

    def __init__(self, clip_eps: float = 0.2) -> None:
        super().__init__()
        self.clip_eps = clip_eps

    def forward(self,
                log_probs: torch.Tensor,
                old_log_probs: torch.Tensor,
                advantages: torch.Tensor,
                action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # sequences : Question text with pads + answer text with pads, [8, 128]
        # action_log_probs : from Actor model, [8, 107]
        # base_action_log_probs : from initial model, [8, 107]
        # attention_mask : True at the location where tokens exist at question text, [8, 21]
        # action_mask : "question text + answer text length (128)" - "question text (21)" = 107, True at the location of no pad, [8, 107]
        
        ratio = (log_probs - old_log_probs).exp()
        # print('ratio',ratio)
        # print('ratio',ratio.shape)
        # ratio tensor([[0.8039, 0.7946, 0.9961,  ..., 0.1283, 0.0735, 0.9634]], device='cuda:0', grad_fn=<ExpBackward0>)
        # ratio torch.Size([1, 107])

        # print('advantages',advantages)
        # print('advantages',advantages.shape)
        # advantages tensor([[0.]], device='cuda:0')
        # advantages torch.Size([1, 1])
        surr1 = ratio * advantages
        # print('surr1',surr1)
        # print('surr1',surr1.shape)
        # surr1 tensor([[0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', grad_fn=<MulBackward0>)
        # surr1 torch.Size([1, 107])

        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        # print('surr2',surr2)
        # print('surr2',surr2.shape)
        # surr2 tensor([[0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', grad_fn=<MulBackward0>)
        # surr2 torch.Size([1, 107])

        loss = -torch.min(surr1, surr2)
        # print('loss',loss)
        # print('loss',loss.shape)
        # loss tensor([[-0., -0., -0.,  ..., -0., -0., -0.]], device='cuda:0', grad_fn=<NegBackward0>)
        # loss torch.Size([1, 107])

        # print('action_mask',action_mask)
        # print('action_mask',action_mask.shape)
        # action_mask tensor([[ True,  True,  True,  ..., False, False, False]], device='cuda:0')
        # action_mask torch.Size([1, 107])
        if action_mask is not None:
            loss = masked_mean(loss, action_mask)
            # print('loss',loss)
            # print('loss',loss.shape)
            # loss tensor([0.], device='cuda:0', grad_fn=<DivBackward0>)
            # loss torch.Size([1])

        loss = loss.mean()
        # print('loss',loss)
        # print('loss',loss.shape)
        # loss tensor(0., device='cuda:0', grad_fn=<MeanBackward0>)
        # loss torch.Size([])

        return loss


class ValueLoss(nn.Module):
    """
    Value Loss for PPO
    """

    def __init__(self, clip_eps: float = 0.4) -> None:
        super().__init__()
        self.clip_eps = clip_eps

    def forward(self,
                values: torch.Tensor,
                old_values: torch.Tensor,
                reward: torch.Tensor,
                action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        # values,
        # experience.values,
        # experience.reward,
        # action_mask=experience.action_mask
        
        values_clipped = old_values + (values - old_values).clamp(-self.clip_eps, self.clip_eps)
        # print('values_clipped',values_clipped)
        # print('values_clipped',values_clipped.shape)
        # values_clipped tensor([0.0289], device='cuda:0', grad_fn=<AddBackward0>)
        # values_clipped torch.Size([1])
        
        surr1 = (values_clipped - reward)**2
        # print('surr1',surr1)
        # print('surr1',surr1.shape)
        # surr1 tensor([0.0051], device='cuda:0', grad_fn=<PowBackward0>)
        # surr1 torch.Size([1])
        
        surr2 = (values - reward)**2
        # print('surr2',surr2)
        # print('surr2',surr2.shape)
        # surr2 tensor([0.0051], device='cuda:0', grad_fn=<PowBackward0>)
        # surr2 torch.Size([1])
        
        loss = torch.max(surr1, surr2)
        # print('loss',loss)
        # print('loss',loss.shape)
        # loss tensor([0.0051], device='cuda:0', grad_fn=<MaximumBackward0>)
        # loss torch.Size([1])
        
        loss = loss.mean()
        # print('loss',loss)
        # print('loss',loss.shape)
        # loss tensor(0.0051, device='cuda:0', grad_fn=<MeanBackward0>)
        # loss torch.Size([])
        
        return loss


class PPOPtxActorLoss(nn.Module):
    """
    To Do:

    PPO-ptx Actor Loss
    """

    def __init__(self, policy_clip_eps: float = 0.2, pretrain_coef: float = 0.0, pretrain_loss_fn=GPTLMLoss()) -> None:
        super().__init__()
        self.pretrain_coef = pretrain_coef
        self.policy_loss_fn = PolicyLoss(clip_eps=policy_clip_eps)
        self.pretrain_loss_fn = pretrain_loss_fn

    def forward(self,
                log_probs: torch.Tensor,
                old_log_probs: torch.Tensor,
                advantages: torch.Tensor,
                lm_logits: torch.Tensor,
                lm_input_ids: torch.Tensor,
                action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        policy_loss = self.policy_loss_fn(log_probs, old_log_probs, advantages, action_mask=action_mask)
        lm_loss = self.pretrain_loss_fn(lm_logits, lm_input_ids)
        return policy_loss + self.pretrain_coef * lm_loss


class PairWiseLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    """

    def forward(self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(chosen_reward - reject_reward)
        log_probs = torch.log(probs)
        loss = -log_probs.mean()
        return loss
