# -*- coding: utf-8 -*-
"""
PPO + 可适配扩散动作先验（PET with LoRA）+ 价值引导（VG）+ 双近端监控（policy-KL / prior-KL）
严格 on-policy 实现要点：
  - PPO 的所有梯度仅基于 on-policy 批次 D_on（来自 env runner 或 robomimic 的 sampler）
  - 先验与价值引导路径一律 stop_gradient：不向 psi（先验）或 φ（Q头）回传
  - 先验仅作为 proposal generator 与软锚定（prior-KL），从不替代策略或参与 PPO 反传
  - PET 仅更新先验中的 LoRA 低秩参数，形成 prior 侧的“近端小步”
"""

import copy
import math
import random
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

import gym
import d4rl  # 仅用于构造环境；不再用于离线数据

import robomimic.models.obs_nets as ObsNets
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
from robomimic.algo import register_algo_factory_func, PolicyAlgo, ValueAlgo

# ------------------------------- 工具函数 -------------------------------

def gaussian_kl(mu1, log_std1, mu2, log_std2, eps=1e-8):
    """KL[ N(mu1, std1) || N(mu2, std2) ]（按最后一维求和）"""
    var1 = (log_std1.exp() + eps) ** 2
    var2 = (log_std2.exp() + eps) ** 2
    kl = 0.5 * ( (var1 / (var2 + eps)) + ((mu2 - mu1) ** 2) / (var2 + eps) - 1.0 + (log_std2 - log_std1) * 2.0 )
    return kl.sum(dim=-1, keepdim=True)


def detach_all(x):
    if isinstance(x, (list, tuple)):
        return [detach_all(t) for t in x]
    if isinstance(x, dict):
        return {k: detach_all(v) for k, v in x.items()}
    if torch.is_tensor(x):
        return x.detach()
    return x


def softmax_stable(x, dim=-1):
    x = x - x.max(dim=dim, keepdim=True)[0]
    return torch.softmax(x, dim=dim)


# ------------------------------- LoRA 低秩适配器 -------------------------------

class LoRALinear(nn.Module):
    """
    在线阶段仅训练 A/B 两个低秩参数，主权重 W 冻结。
    y = x W^T + scale * x (B A)^T
    """
    def __init__(self, in_features, out_features, r=8, bias=True, scale=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.scale = scale

        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)  # 冻结
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False) if bias else None

        # LoRA 参数
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_A)
        nn.init.zeros_(self.lora_B)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        base = torch.nn.functional.linear(x, self.weight, self.bias)
        lora = torch.nn.functional.linear(x, torch.matmul(self.lora_B, self.lora_A)) * self.scale
        return base + lora


def lora_mlp(in_dim, hidden_dims, out_dim, r=8, scale=1.0, act=nn.ReLU):
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers += [LoRALinear(prev, h, r=r, scale=scale), act()]
        prev = h
    layers += [LoRALinear(prev, out_dim, r=r, scale=scale)]
    return nn.Sequential(*layers)


# ------------------------------- 网络定义：Actor / Critic(Q) / Prior -------------------------------

class ActorNetwork(nn.Module):
    """Gaussian policy head: πθ(a|s) = N(μθ, diag(σθ^2))"""
    def __init__(self, obs_shapes, ac_dim, mlp_layer_dims, encoder_kwargs):
        super().__init__()
        self.encoder = ObsNets.obs_encoder_factory(obs_shapes, encoder_kwargs=encoder_kwargs)
        fea_dim = self.encoder.output_shape()[0]
        self.mlp = nn.Sequential(
            nn.Linear(fea_dim, mlp_layer_dims[0]), nn.ReLU(),
            nn.Linear(mlp_layer_dims[0], mlp_layer_dims[1]), nn.ReLU(),
        )
        self.mu = nn.Linear(mlp_layer_dims[1], ac_dim)
        self.log_std = nn.Linear(mlp_layer_dims[1], ac_dim)

    def forward(self, obs_dict):
        h = self.encoder(obs_dict)
        h = self.mlp(h)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(-5.0, 2.0)
        return mu, log_std

    def dist(self, obs_dict):
        mu, log_std = self.forward(obs_dict)
        return Normal(mu, log_std.exp())


class CriticQNetwork(nn.Module):
    """
    共享编码器：V(s) + 轻量 Q(s,a)，Q 用一步 TD 目标训练，仅用于 VG 排序/引导。
    """
    def __init__(self, obs_shapes, ac_dim, mlp_layer_dims, encoder_kwargs):
        super().__init__()
        self.encoder = ObsNets.obs_encoder_factory(obs_shapes, encoder_kwargs=encoder_kwargs)
        fea_dim = self.encoder.output_shape()[0]
        self.v_mlp = nn.Sequential(
            nn.Linear(fea_dim, mlp_layer_dims[0]), nn.ReLU(),
            nn.Linear(mlp_layer_dims[0], 1),
        )
        self.q_mlp = nn.Sequential(
            nn.Linear(fea_dim + ac_dim, mlp_layer_dims[0]), nn.ReLU(),
            nn.Linear(mlp_layer_dims[0], 1),
        )

    def V(self, obs_dict):
        h = self.encoder(obs_dict)
        return self.v_mlp(h)

    def Q(self, obs_dict, actions):
        h = self.encoder(obs_dict)
        ha = torch.cat([h, actions], dim=-1)
        return self.q_mlp(ha)


class DiffusionActionPrior(nn.Module):
    """
    可适配的动作先验：
      - 默认实现：编码器 + LoRA-MLP + 高斯头（作为“扩散的高斯代理”），支持 prior-KL/采样
      - 若需要接 synther 的扩散采样器，可以在 sample() 内替换为扩散过程（不回传梯度）
    PET：仅训练 LoRA 低秩参数；其余参数冻结。
    """
    def __init__(self, obs_shapes, ac_dim, hidden=(256, 256), encoder_kwargs=None,
                 lora_rank=8, lora_scale=1.0, device="cuda"):
        super().__init__()
        self.device = device
        self.encoder = ObsNets.obs_encoder_factory(obs_shapes, encoder_kwargs=encoder_kwargs)
        fea_dim = self.encoder.output_shape()[0]
        # 冻结编码器主干（先验仅 PET LoRA）
        for p in self.encoder.parameters():
            p.requires_grad = False
        # LoRA-MLP 作为先验的小适配器
        self.mlp = lora_mlp(fea_dim, list(hidden), hidden[-1], r=lora_rank, scale=lora_scale, act=nn.ReLU)
        self.mu = LoRALinear(hidden[-1], ac_dim, r=lora_rank, scale=lora_scale)
        self.log_std = LoRALinear(hidden[-1], ac_dim, r=lora_rank, scale=lora_scale)

        # 便于监控、计算 prior-KL 的“高斯代理头”
        self._pet_params = [p for n, p in self.named_parameters() if ("lora_" in n)]

        # 可选：如需挂接真正的扩散模型，可在此初始化；本实现以高斯代理为主。
        self.use_external_diffusion = False

    @property
    def pet_parameters(self):
        return self._pet_params  # 仅 LoRA 参数

    def forward(self, obs_dict):
        with torch.no_grad():   # 编码器冻结且不需要梯度
            h = self.encoder(obs_dict)
        h2 = self.mlp(h)        # 仅 LoRA 可训练
        mu = self.mu(h2)
        log_std = self.log_std(h2).clamp(-5.0, 2.0)
        return mu, log_std

    @torch.no_grad()
    def sample(self, obs_dict, K=10):
        """在给定 obs 上采样 K 个动作（无梯度路径） -> (B, K, A)"""
        mu, log_std = self.forward(obs_dict)
        std = log_std.exp()
        B, A = mu.shape
        mu = mu.unsqueeze(1).expand(B, K, A)
        std = std.unsqueeze(1).expand(B, K, A)
        return Normal(mu, std).sample()

# ------------------------------- PPO 主算法 -------------------------------

@register_algo_factory_func("ppo")
def algo_config_to_class(algo_config, obs_config, global_config, obs_key_shapes, ac_dim, device):
    return PPO(algo_config=algo_config, obs_config=obs_config, global_config=global_config,
               obs_key_shapes=obs_key_shapes, ac_dim=ac_dim, device=device), {}


class PPO(PolicyAlgo, ValueAlgo):
    """
    关键开关：
      - use_vg         : 是否启用价值引导（能量重加权/梯度引导）
      - use_grad_guid  : 是否启用“过程内梯度引导”
      - use_pet        : 是否在线 PET（仅 LoRA 参数）
      - pet_freq       : PET 更新频率（每多少个 actor 更新后做一次 PET）
      - k_proposals    : 每个状态的候选数 K
      - aux_ratio      : 候选在一个更新中的使用比例（<=20%）
      - lambda_kl / lambda_aux : 软先验 KL 与辅助模仿权重（建议 1e-3~1e-2）
    """
    def __init__(
        self,
        algo_config, obs_config, global_config, obs_key_shapes, ac_dim, device,
        use_vg=True, use_grad_guid=True, use_pet=True,
        pet_freq=10, k_proposals=10, aux_ratio=0.2,
        beta_init=0.0, beta_final=1.0, beta_warmup=0.3,
        alpha_max=0.30,  # 梯度引导步长上限
    ):
        PolicyAlgo.__init__(self, algo_config=algo_config, obs_config=obs_config, global_config=global_config,
                            obs_key_shapes=obs_key_shapes, ac_dim=ac_dim, device=device)
        ValueAlgo.__init__(self, algo_config=algo_config, obs_config=obs_config, global_config=global_config,
                           obs_key_shapes=obs_key_shapes, ac_dim=ac_dim, device=device)

        self.device = device
        self.obs_shapes = obs_key_shapes
        self.ac_dim = ac_dim

        # ----------------- 网络 -----------------
        self._create_networks(obs_config)

        # 优化器
        self.optimizer_actor = optim.Adam(self.nets['actor'].parameters(),
                                          lr=algo_config.optim_params.actor.learning_rate.initial)
        self.optimizer_critic = optim.Adam(self.nets['critic'].parameters(),
                                           lr=algo_config.optim_params.critic.learning_rate.initial)

        # 先验 PET 优化器（仅 LoRA 参数）
        self.use_pet = use_pet
        pet_params = list(self.nets['prior'].pet_parameters)
        self.pet_optimizer = optim.Adam(pet_params, lr=1e-5) if (use_pet and len(pet_params) > 0) else None

        # PPO & GAE
        self.eps_clip = algo_config.eps_clip
        self.gamma = algo_config.discount
        self.lamda = algo_config.lamda
        self.ppo_update_steps = algo_config.ppo_update_steps

        # VG 相关
        self.use_vg = use_vg
        self.use_grad_guid = use_grad_guid
        self.k_proposals = k_proposals
        self.aux_ratio = aux_ratio
        self.beta_init, self.beta_final = beta_init, beta_final
        self.beta_warmup = beta_warmup
        self.alpha_max = alpha_max

        self.pet_freq = pet_freq
        self.iter_count = 0

        # 旧策略/旧先验（用于 KL 监控与 PPO 比率）
        self.actor_old = copy.deepcopy(self.nets['actor']).eval()
        self.prior_old = copy.deepcopy(self.nets['prior']).eval()
        for p in self.prior_old.parameters():
            p.requires_grad = False

        # 监控
        self.last_log = OrderedDict()

    # ----------------- 构建网络 -----------------
    def _create_networks(self, obs_config):
        self.nets = nn.ModuleDict()
        encoder_kwargs = self._get_encoder_kwargs(obs_config)
        # actor
        self.nets['actor'] = ActorNetwork(
            obs_shapes=self.obs_shapes, ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor.layer_dims,
            encoder_kwargs=encoder_kwargs
        )
        # critic + Q
        self.nets['critic'] = CriticQNetwork(
            obs_shapes=self.obs_shapes, ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.critic.layer_dims,
            encoder_kwargs=encoder_kwargs
        )
        # prior（LoRA-Adapter）
        self.nets['prior'] = DiffusionActionPrior(
            obs_shapes=self.obs_shapes, ac_dim=self.ac_dim,
            hidden=(256, 256), encoder_kwargs=encoder_kwargs,
            lora_rank=8, lora_scale=1.0, device=self.device
        )
        self.nets = self.nets.float().to(self.device)

    def _get_encoder_kwargs(self, obs_config):
        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(obs_config.encoder)
        if 'obs' not in encoder_kwargs:
            encoder_kwargs['obs'] = {}
        encoder_kwargs['obs']['feature_activation'] = nn.ReLU
        return encoder_kwargs

    # ----------------- 训练主循环（严格 on-policy） -----------------
    def train(self, num_epochs, batch_size):
        """
        说明：
          - 假设 robomimic 的外部 runner/sampler 会提供 on-policy batch
          - 如果你们自有 runner，也只需确保 sample_batch 返回的是当前策略 rollouts
        """
        for epoch in range(num_epochs):
            # 1) 采样 on-policy 批次（外部采样器）并预处理
            batch = self.sample_batch(batch_size)               # == D_on
            batch = self.process_batch_for_training(batch)      # to device/float
            # 2) 单次迭代的 PPO + prior-KL + aux-BC（VG proposals）
            log = self._train_on_batch_onpolicy(batch, epoch)
            # 3) 可选：PET（仅 LoRA），频率 pet_freq
            if self.use_pet and (self.iter_count % self.pet_freq == 0):
                self._pet_update_on_onpolicy(batch)
            # 4) 维护旧快照与监控
            self.actor_old.load_state_dict(self.nets['actor'].state_dict())
            self.prior_old.load_state_dict(self.nets['prior'].state_dict())
            self.last_log = log
            self.iter_count += 1

    # ----------------- 单批次训练 -----------------
    def _train_on_batch_onpolicy(self, batch, epoch):
        obs, actions, rewards, dones, next_obs = batch['obs'], batch['actions'], batch['rewards'], batch['dones'], batch['next_obs']

        # 旧策略 log-prob（冻结）
        with torch.no_grad():
            mu_old, ls_old = self.actor_old(obs)
            dist_old = Normal(mu_old, ls_old.exp())
            old_log_probs = dist_old.log_prob(actions).sum(-1, keepdim=True)

        # GAE 优势
        advantages, returns = self._compute_gae(obs, actions, rewards, dones, next_obs)

        # β 退火（能量重加权）
        beta = self._anneal_beta(epoch, self.beta_init, self.beta_final, self.beta_warmup)

        # 多个 PPO 小步
        for _ in range(self.ppo_update_steps):
            # 策略分布
            mu_new, ls_new = self.nets['actor'](obs)
            dist_new = Normal(mu_new, ls_new.exp())
            log_probs = dist_new.log_prob(actions).sum(-1, keepdim=True)
            ratios = torch.exp(log_probs - old_log_probs)

            # PPO clipped loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            ppo_loss = -torch.min(surr1, surr2).mean()

            # 软先验 KL（对 prior 断梯度；仅同一在线状态）
            with torch.no_grad():
                mu_p, ls_p = self.nets['prior'](obs)   # no grad -> 不回传到 psi
            prior_kl = gaussian_kl(mu_new, ls_new, mu_p, ls_p).mean()

            # 辅助模仿（VG proposals，仅同一在线状态；对 proposals detach）
            aux_loss = torch.tensor(0.0, device=self.device)
            if self.use_vg and self.aux_ratio > 0:
                aux_obs, aux_actions = self._vg_proposals(obs, K=self.k_proposals, beta=beta,
                                                          use_grad=self.use_grad_guid, alpha=self.alpha_max)
                if aux_obs is not None:
                    mu_aux, ls_aux = self.nets['actor'](aux_obs)
                    dist_aux = Normal(mu_aux, ls_aux.exp())
                    aux_loss = - dist_aux.log_prob(aux_actions).sum(-1).mean()

            # Critic loss：V 的 TD；Q 的一步 TD
            v_pred = self.nets['critic'].V(obs)
            critic_v_loss = nn.MSELoss()(v_pred, returns)
            with torch.no_grad():
                v_next = self.nets['critic'].V(next_obs)
            q_pred = self.nets['critic'].Q(obs, actions)
            q_target = rewards + self.gamma * v_next * (1 - dones)
            critic_q_loss = nn.MSELoss()(q_pred, q_target)
            critic_loss = critic_v_loss + critic_q_loss

            # 组合损失（权重小）
            lam_kl = getattr(self.algo_config, "lambda_kl", 5e-3)
            lam_aux = getattr(self.algo_config, "lambda_aux", 1e-2)
            actor_loss = ppo_loss + lam_kl * prior_kl + lam_aux * aux_loss

            # ========== 反传与泄漏监控 ==========
            # 先清梯度
            self.optimizer_actor.zero_grad(set_to_none=True)
            self.optimizer_critic.zero_grad(set_to_none=True)

            # 仅 actor 反传；先验与 Q 头因 no_grad / freeze，不应产生梯度
            actor_loss.backward(retain_graph=True)
            leak_psi = self._grad_norm(self.nets['prior'])
            leak_q = self._grad_norm_q_head()  # 只统计 critic 的 Q 支路
            torch.nn.utils.clip_grad_norm_(self.nets['actor'].parameters(),
                                           max_norm=self.algo_config.max_grad_norm)
            self.optimizer_actor.step()

            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.nets['critic'].parameters(),
                                           max_norm=self.algo_config.max_grad_norm)
            self.optimizer_critic.step()

        # 监控 KL（双近端）
        with torch.no_grad():
            mu_old_new, ls_old_new = self.actor_old(obs)
            kl_policy = gaussian_kl(mu_new, ls_new, mu_old_new, ls_old_new).mean().item()
            mu_p_cur, ls_p_cur = self.nets['prior'](obs)
            mu_p_old, ls_p_old = self.prior_old(obs)
            kl_prior = gaussian_kl(mu_p_cur, ls_p_cur, mu_p_old, ls_p_old).mean().item()

        loss_log = OrderedDict()
        loss_log["Actor/PPO"] = ppo_loss.item()
        loss_log["Actor/PriorKL"] = prior_kl.item()
        loss_log["Actor/AuxBC"] = aux_loss.item() if torch.is_tensor(aux_loss) else 0.0
        loss_log["Critic/Loss"] = critic_loss.item()
        loss_log["Monitor/KL_policy"] = kl_policy
        loss_log["Monitor/KL_prior"] = kl_prior
        loss_log["Monitor/LeakGradPsi"] = leak_psi
        loss_log["Monitor/LeakGradQ"] = leak_q
        return loss_log

    # ----------------- PET：仅 LoRA 参数的小步更新（仅用 D_on） -----------------
    def _pet_update_on_onpolicy(self, batch):
        if (self.pet_optimizer is None) or (len(self.nets['prior'].pet_parameters) == 0):
            return
        obs, actions = batch['obs'], batch['actions']
        # 最大似然：-log p_psi(a|s)（只更新 LoRA 参数）
        mu_p, ls_p = self.nets['prior'](obs)  # 前向只涉及 LoRA；编码器冻结
        dist_p = Normal(mu_p, ls_p.exp())
        pet_loss = - dist_p.log_prob(actions).sum(-1).mean()

        self.pet_optimizer.zero_grad(set_to_none=True)
        pet_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.nets['prior'].pet_parameters, max_norm=1.0)
        self.pet_optimizer.step()

    # ----------------- VG：能量重加权 + 可选梯度引导 -----------------
    def _vg_proposals(self, obs_dict, K=10, beta=1.0, use_grad=True, alpha=0.30):
        """
        返回 (obs_rep, actions_rep)：
          - obs_rep:   [M, ...] 重复展开后的状态
          - actions_rep: [M, A] 选中的高价值候选
        所有 proposal 均为 no_grad / detach 的常量，不向 prior 或 Q 头回传。
        """
        B = next(iter(obs_dict.values())).shape[0]
        # 1) 从先验采样 K 个候选（无梯度）
        with torch.no_grad():
            a0 = self.nets['prior'].sample(obs_dict, K=K)  # [B, K, A]

        # 2) 可选：梯度引导：对 Q(s,a) w.r.t a 的梯度做一步更新（不对 Q 头回传）
        if use_grad:
            # 冻结 Q 头参数；仅对 a 求导
            for p in self.nets['critic'].parameters():
                p.requires_grad_(False)
            a_flat = a0.reshape(B * K, -1).detach().requires_grad_(True)
            # 复制状态
            obs_rep = {k: v.unsqueeze(1).expand(-1, K, *v.shape[1:]).reshape(B * K, *v.shape[1:]) for k, v in obs_dict.items()}
            q = self.nets['critic'].Q(obs_rep, a_flat).sum()  # 标量
            grad_a = torch.autograd.grad(q, a_flat, retain_graph=False, create_graph=False)[0]
            a_ref = (a_flat + alpha * grad_a).detach()
            a = a_ref.reshape(B, K, -1)
            # 解除冻结（供后续 TD 更新）
            for p in self.nets['critic'].parameters():
                p.requires_grad_(True)
        else:
            a = a0  # (B, K, A)

        # 3) 能量重加权 w_i ∝ exp(β Q̂(s, a_i))（不回传）
        with torch.no_grad():
            obs_rep = {k: v.unsqueeze(1).expand(-1, K, *v.shape[1:]).reshape(B * K, *v.shape[1:]) for k, v in obs_dict.items()}
            q_vals = self.nets['critic'].Q(obs_rep, a.reshape(B * K, -1)).reshape(B, K, 1)
            w = softmax_stable(beta * q_vals, dim=1)  # [B, K, 1]

            # 选取每个状态 m = ceil(aux_ratio * K) 个候选
            m = max(1, int(math.ceil(self.aux_ratio * K)))
            idx = torch.multinomial(w.squeeze(-1), num_samples=m, replacement=True)  # [B, m]
            # 收集对应的动作
            gather_idx = idx.unsqueeze(-1).expand(B, m, a.shape[-1])
            a_sel = a.gather(dim=1, index=gather_idx)  # [B, m, A]
            # 展平
            a_sel = a_sel.reshape(B * m, -1)
            obs_sel = {k: v.unsqueeze(1).expand(-1, m, *v.shape[1:]).reshape(B * m, *v.shape[1:]) for k, v in obs_dict.items()}

        return obs_sel, a_sel

    # ----------------- GAE / Returns -----------------
    def _compute_gae(self, obs, actions, rewards, dones, next_obs):
        """
        GAE-Lambda, critic 使用 V(s)
        """
        with torch.no_grad():
            values = self.nets['critic'].V(obs).squeeze(-1)           # [T]
            next_values = self.nets['critic'].V(next_obs).squeeze(-1) # [T]
        rewards = rewards.squeeze(-1)
        dones = dones.squeeze(-1)
        T = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lamda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = (advantages + values).unsqueeze(-1)
        return advantages.unsqueeze(-1), returns

    def _anneal_beta(self, epoch, beta0, beta1, warmup_ratio):
        # 线性退火：前 warmup_ratio * total_epochs 内从 beta0 -> beta1
        # 若不知道总 epoch，可用经验：前 30% 迭代退火；此处用 self.iter_count 代替
        p = min(1.0, max(0.0, (self.iter_count + 1) / max(1.0, self.ppo_update_steps) / max(1.0, warmup_ratio*100)))
        return beta0 + (beta1 - beta0) * p

    # ----------------- 监控与工具 -----------------
    def _grad_norm(self, module: nn.Module):
        total = 0.0
        has_grad = False
        for p in module.parameters():
            if p.grad is not None:
                total += (p.grad.detach().float() ** 2).sum().item()
                has_grad = True
        if not has_grad:
            return 0.0
        return math.sqrt(max(total, 0.0))

    def _grad_norm_q_head(self):
        # 仅统计 critic 的梯度（如果错误地回传了，会非零）
        return self._grad_norm(self.nets['critic'])

    # ----------------- robomimic 框架兼容的辅助 -----------------
    def process_batch_for_training(self, batch):
        input_batch = {
            'obs': {k: batch['obs'][k].squeeze(1) for k in batch['obs'] if batch['obs'][k] is not None},
            'next_obs': {k: batch['next_obs'][k].squeeze(1) for k in batch['next_obs'] if batch['next_obs'][k] is not None},
            'actions': batch['actions'].squeeze(1) if batch['actions'] is not None else None,
            'rewards': batch['rewards'].squeeze() if batch['rewards'] is not None else None,
            'dones': batch['dones'].squeeze() if batch['dones'] is not None else None,
        }
        self._check_for_nan_in_data(input_batch)
        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def _check_for_nan_in_data(self, data):
        if isinstance(data, dict):
            for k, v in data.items():
                if v is None: continue
                if isinstance(v, dict):
                    self._check_for_nan_in_data(v)
                elif torch.isnan(v).any() or torch.isinf(v).any():
                    raise ValueError(f"NaN/Inf detected in {k}")
        elif torch.is_tensor(data):
            if torch.isnan(data).any() or torch.isinf(data).any():
                raise ValueError("NaN/Inf detected in tensor")

    # robomimic 的 API：日志
    def log_info(self, info=None):
        if info is None:
            info = self.last_log if hasattr(self, "last_log") else {}
        return info

    def set_train(self):
        self.nets.train()

    def set_eval(self):
        self.nets.eval()

# ------------------------------- 运行入口 -------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    import argparse
    from robomimic.utils import train_utils as TrainUtils

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="robomimic 配置文件路径")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    config = TrainUtils.load_config(args.config)
    device = torch.device(args.device)

    algo, _ = algo_config_to_class(config.algo, config.obs, config.global_config,
                                   config.obs_key_shapes, config.ac_dim, device)

    # 由 robomimic 的 training harness 负责 runner/sampler 与 epoch 循环
    # 这里仅展示如何直接调用：
    algo.train(num_epochs=config.train.num_epochs, batch_size=config.train.batch_size)


if __name__ == "__main__":
    main()
