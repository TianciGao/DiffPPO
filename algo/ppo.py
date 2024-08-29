import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import OrderedDict
import robomimic.models.obs_nets as ObsNets
import robomimic.models.value_nets as ValueNets
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
from robomimic.algo import register_algo_factory_func, PolicyAlgo, ValueAlgo
import random
import numpy as np
import gin
from synther.diffusion.utils import construct_diffusion_model, make_inputs, split_diffusion_samples
from synther.diffusion.train_diffuser import SimpleDiffusionGenerator, Trainer
import gym
import d4rl

@register_algo_factory_func("ppo")
def algo_config_to_class(algo_config, obs_config, global_config, obs_key_shapes, ac_dim, device):
    return PPO(algo_config=algo_config, obs_config=obs_config, global_config=global_config,
               obs_key_shapes=obs_key_shapes, ac_dim=ac_dim, device=device), {}

class PPO(PolicyAlgo, ValueAlgo):
    def __init__(self, algo_config, obs_config, global_config, obs_key_shapes, ac_dim, device,
                 virtual_trajectory_freq=1, num_virtual_trajectories=10, diffusion_update_freq=5):
        PolicyAlgo.__init__(self, algo_config=algo_config, obs_config=obs_config, global_config=global_config,
                            obs_key_shapes=obs_key_shapes, ac_dim=ac_dim, device=device)
        self._create_networks()
        self.optimizer_actor = optim.Adam(self.nets['actor'].parameters(),
                                          lr=self.algo_config.optim_params.actor.learning_rate.initial)
        self.optimizer_critic = optim.Adam(self.nets['critic'].parameters(),
                                           lr=self.algo_config.optim_params.critic.learning_rate.initial)
        self.eps_clip = self.algo_config.eps_clip
        self.gamma = self.algo_config.discount
        self.lamda = self.algo_config.lamda
        self.ppo_update_steps = self.algo_config.ppo_update_steps
        self.virtual_trajectory_freq = virtual_trajectory_freq
        self.num_virtual_trajectories = num_virtual_trajectories
        self.diffusion_update_freq = diffusion_update_freq
        self.current_epoch = 0

        # SynthER 相关初始化
        if algo_config.synth_er.enabled:
            gin.parse_config_files_and_bindings(algo_config.synth_er.gin_config_files, [])
            self.diffusion_model = self._initialize_diffusion_model()
            self.diffusion_trainer = Trainer(self.diffusion_model, None, results_folder=algo_config.synth_er.results_folder)
            self.synth_er_generator = SimpleDiffusionGenerator(
                env=gym.make('hopper-medium-v2'),
                ema_model=self.diffusion_trainer.ema.ema_model,
                num_sample_steps=algo_config.synth_er.num_sample_steps,
                sample_batch_size=algo_config.synth_er.sample_batch_size
            )
            self.diffusion_dataset = []

    def _initialize_diffusion_model(self):
        env = gym.make('hopper-medium-v2')
        inputs = make_inputs(env)
        inputs = torch.from_numpy(inputs).float()
        return construct_diffusion_model(inputs=inputs)

    def _create_networks(self):
        self.nets = nn.ModuleDict()
        self._create_actor()
        self._create_critic()
        self.nets = self.nets.float().to(self.device)
        self._initialize_weights(method='xavier_normal')

    def _initialize_weights(self, method='xavier_normal'):
        for m in self.nets.modules():
            if isinstance(m, nn.Linear):
                if method == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight)
                elif method == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight)
                elif method == 'kaiming_normal':
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                elif method == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif method == 'orthogonal':
                    nn.init.orthogonal_(m.weight)
                else:
                    raise ValueError(f"Unknown initialization method: {method}")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

    def _create_actor(self):
        actor_class = ActorNetwork
        actor_args = {
            'obs_shapes': self.obs_shapes,
            'goal_shapes': self.goal_shapes,
            'ac_dim': self.ac_dim,
            'mlp_layer_dims': self.algo_config.actor.layer_dims,
            'encoder_kwargs': self._get_encoder_kwargs()
        }
        self.nets['actor'] = actor_class(**actor_args)

    def _create_critic(self):
        critic_class = ValueNets.ValueNetwork
        critic_args = {
            'obs_shapes': self.obs_shapes,
            'goal_shapes': self.goal_shapes,
            'mlp_layer_dims': self.algo_config.critic.layer_dims,
            'encoder_kwargs': self._get_encoder_kwargs()
        }
        self.nets['critic'] = critic_class(**critic_args)

    def _get_encoder_kwargs(self):
        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder)
        if 'obs' not in encoder_kwargs:
            encoder_kwargs['obs'] = {}
        encoder_kwargs['obs']['feature_activation'] = nn.ReLU
        return encoder_kwargs

    def generate_trajectory(self, initial_obs, max_steps=1000):
        obs = initial_obs
        trajectory = []
        for _ in range(max_steps):
            obs_tensor = TensorUtils.to_tensor(obs).unsqueeze(0).to(self.device)
            mu, log_std = self.nets['actor'](obs_tensor)
            std = log_std.exp()
            dist = Normal(mu, std)
            action = dist.sample().squeeze(0).cpu().numpy()

            next_obs, reward, done, _ = self.env.step(action)
            incremental_reward = self.improved_incremental_reward_function(obs, action, next_obs, reward, done)
            trajectory.append((obs, action, incremental_reward, next_obs, done))

            if done:
                break

            obs = next_obs

        return trajectory

    def improved_incremental_reward_function(self, obs, action, next_obs, reward, done):
        incremental_reward = reward
        action_penalty = np.sum(np.abs(action))
        incremental_reward -= 0.01 * action_penalty
        if hasattr(self, 'last_action'):
            action_diff = np.sum(np.abs(action - self.last_action))
            incremental_reward -= 0.01 * action_diff
        self.last_action = action
        return incremental_reward

    def generate_and_optimize_virtual_trajectory(self, max_steps=1000):
        obs = self.env.reset()
        virtual_trajectory = []
        real_trajectory = []

        for _ in range(max_steps):
            obs_tensor = TensorUtils.to_tensor(obs).unsqueeze(0).to(self.device)
            mu, log_std = self.nets['actor'](obs_tensor)
            std = log_std.exp()
            dist = Normal(mu, std)
            action = dist.sample().squeeze(0).cpu().numpy()

            next_obs, reward, done, _ = self.env.step(action)
            incremental_reward = self.improved_incremental_reward_function(obs, action, next_obs, reward, done)
            virtual_trajectory.append((obs, action, incremental_reward, next_obs, done))

            real_obs, real_action, real_reward, real_next_obs, real_done = self.dataset.sample()
            real_trajectory.append((real_obs, real_action, real_reward, real_next_obs, real_done))
            if len(real_trajectory) >= max_steps:
                break

            if done:
                break

            obs = next_obs

        bc_optimizer = BCOptimizer(self.nets['actor'])
        bc_optimizer.optimize(virtual_trajectory, real_trajectory)
        return virtual_trajectory

    def generate_virtual_trajectories(self, num_trajectories, max_steps=1000):
        virtual_trajectories = []
        for _ in range(num_trajectories):
            trajectory = self.generate_and_optimize_virtual_trajectory(max_steps)
            virtual_trajectories.append(trajectory)
        return virtual_trajectories

    def add_virtual_trajectories_to_data(self, virtual_trajectories):
        for trajectory in virtual_trajectories:
            for step in trajectory:
                obs, action, reward, next_obs, done = step
                self.dataset.add(obs, action, reward, next_obs, done)

    def train(self, num_epochs, batch_size):
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            batch = self.sample_batch(batch_size)
            processed_batch = self.process_batch_for_training(batch)
            self.train_on_batch(processed_batch, epoch)

            if epoch % self.virtual_trajectory_freq == 0:
                virtual_trajectories = self.generate_virtual_trajectories(self.num_virtual_trajectories)
                self.add_virtual_trajectories_to_data(virtual_trajectories)

            self.log_info(epoch)

            # 在训练过程中生成和评估扩散曲线
            if epoch % 10 == 0:  # 每10个epoch生成一次扩散曲线
                observations, actions, rewards, next_observations, terminals = self.synth_er_generator.sample(
                    num_samples=100000)
                # 在此处使用生成的样本进行策略优化或其他评估
                # 例如，可以使用这些样本更新PPO的经验池，或直接用于策略的评估和改进

            # 定期更新扩散模型
            if epoch % self.diffusion_update_freq == 0 and epoch > 0:
                self.update_diffusion_model()

    def process_batch_for_training(self, batch):
        input_batch = {
            'obs': {k: batch['obs'][k].squeeze(1) for k in batch['obs'] if batch['obs'][k] is not None},
            'next_obs': {k: batch['next_obs'][k].squeeze(1) for k in batch['next_obs'] if
                         batch['next_obs'][k] is not None},
            'goal_obs': batch.get('goal_obs', None),
            'actions': batch['actions'].squeeze(1) if batch['actions'] is not None else None,
            'rewards': batch['rewards'].squeeze() if batch['rewards'] is not None else None,
            'dones': batch['dones'].squeeze() if batch['dones'] is not None else None,
        }
        self._check_for_nan_in_data(input_batch)
        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def train_on_batch(self, batch, epoch, validate=False):
        processed_batch = self.process_batch_for_training(batch)
        obs, actions, rewards, dones, next_obs = processed_batch['obs'], processed_batch['actions'], processed_batch[
            'rewards'], processed_batch['dones'], processed_batch['next_obs']
        old_log_probs = self._get_log_prob(obs, actions).detach()
        improved_actions = actions

        for _ in range(self.ppo_update_steps):
            log_probs = self._get_log_prob(obs, improved_actions)
            ratios = torch.exp(log_probs - old_log_probs + 1e-10)
            advantages = self._compute_advantages(rewards, dones, obs, next_obs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self._compute_critic_loss(obs, rewards, dones, next_obs)

            if not validate:
                self.optimizer_actor.zero_grad()
                actor_loss.backward(retain_graph=True)
                self._check_for_nan(self.nets['actor'], name="actor")
                torch.nn.utils.clip_grad_norm_(self.nets['actor'].parameters(), max_norm=self.algo_config.max_grad_norm)
                self.optimizer_actor.step()

                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.nets['critic'].parameters(),
                                               max_norm=self.algo_config.max_grad_norm)
                self.optimizer_critic.step()

        return {'actor_loss': actor_loss.item(), 'critic_loss': critic_loss.item()}

    def _get_log_prob(self, obs, actions):
        mu, log_std = self.nets['actor'](obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        log_probs = dist.log_prob(actions).sum(-1, keepdim=True)
        return log_probs

    def _compute_advantages(self, rewards, dones, obs, next_obs):
        values = self.nets['critic'](obs).squeeze().detach()
        next_values = self.nets['critic'](next_obs).squeeze().detach()
        advantages = torch.zeros_like(rewards)
        advantage = 0
        for t in reversed(range(len(rewards))):
            td_error = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            advantage = td_error + self.gamma * self.lamda * (1 - dones[t]) * advantage
            advantages[t] = advantage
        return advantages

    def _compute_critic_loss(self, obs, rewards, dones, next_obs):
        values = self.nets['critic'](obs).squeeze()
        targets = rewards + self.gamma * self.nets['critic'](next_obs).squeeze() * (1 - dones)
        return nn.MSELoss()(values, targets)

    def _check_for_nan(self, module, name):
        for param_name, param in module.named_parameters():
            if param.grad is None:
                print(f"Warning: Gradient for {name}.{param_name} is None")
            elif torch.isnan(param.grad).any():
                print(f"NaN detected in gradients of {name}.{param_name}")
                print(f"grad: {param.grad}")
                raise ValueError(f"NaN detected in gradients of {name}.{param_name}")

    def _check_for_nan_in_data(self, data):
        if isinstance(data, dict):
            for key, value in data.items():
                if value is None:
                    continue
                elif isinstance(value, dict):
                    self._check_for_nan_in_data(value)
                elif torch.isnan(value).any() or torch.isinf(value).any():
                    print(f"NaN or Inf detected in {key}")
                    print(f"value: {value}")
                    raise ValueError(f"NaN or Inf detected in {key}")
        elif isinstance(data, torch.Tensor):
            if torch.isnan(data).any() or torch.isinf(data).any():
                print("NaN or Inf detected in tensor")
                raise ValueError("NaN or Inf detected in tensor")

    def log_info(self, info):
        loss_log = OrderedDict()
        loss_log["Actor/Loss"] = info["actor_loss"]
        loss_log["Critic/Loss"] = info["critic_loss"]
        return loss_log

    def set_train(self):
        self.nets.train()

    def set_eval(self):
        self.nets.eval()

    def on_epoch_end(self, epoch):
        if self.lr_schedulers["critic"] is not None:
            for lr_sc in self.lr_schedulers["critic"]:
                if lr_sc is not None:
                    lr_sc.step()
        if self.lr_schedulers["actor"] is not None:
            self.lr_schedulers["actor"].step()

    def get_action(self, obs_dict, goal_dict=None):
        assert not self.nets.training
        obs = TensorUtils.to_tensor(obs_dict)
        with torch.no_grad():
            mu, log_std = self.nets['actor'](obs)
            std = log_std.exp()
            dist = Normal(mu, std)
            action = dist.sample()
        return action

    def get_state_value(self, obs_dict, goal_dict=None):
        assert not self.nets.training
        obs = TensorUtils.to_tensor(obs_dict)
        with torch.no_grad():
            value = self.nets['critic'](obs)
        return value

    def update_diffusion_model(self):
        print("Updating diffusion model...")
        # 更新扩散模型的训练数据集
        dataset = d4rl.qlearning_dataset(gym.make('hopper-medium-v2'))
        inputs = torch.from_numpy(make_inputs(gym.make('hopper-medium-v2'))).float()
        self.diffusion_trainer.dataset = torch.utils.data.TensorDataset(inputs)

        # 训练扩散模型
        self.diffusion_trainer.train()

        # 更新SimpleDiffusionGenerator
        self.synth_er_generator.diffusion = self.diffusion_trainer.ema.ema_model

class ActorNetwork(nn.Module):
    def __init__(self, obs_shapes, goal_shapes, ac_dim, mlp_layer_dims, encoder_kwargs):
        super(ActorNetwork, self).__init__()
        self.encoder = ObsNets.obs_encoder_factory(obs_shapes, encoder_kwargs=encoder_kwargs)
        output_shape = self.encoder.output_shape()
        self.mlp = nn.Sequential(
            nn.Linear(output_shape[0], mlp_layer_dims[0]),
            nn.ReLU(),
            nn.Linear(mlp_layer_dims[0], mlp_layer_dims[1]),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(mlp_layer_dims[1], ac_dim)
        self.log_std_layer = nn.Linear(mlp_layer_dims[1], ac_dim)

    def forward(self, obs_dict, goal_dict=None):
        h = self.encoder(obs_dict)
        h = self.mlp(h)
        mu = self.mu_layer(h)
        log_std = self.log_std_layer(h)
        return mu, log_std

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
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument("--device", type=str, default="cuda:0", help="device to use for training")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument("--init_method", type=str, default='xavier_normal', help="weight initialization method")
    parser.add_argument("--virtual_trajectory_freq", type=int, default=1,
                        help="frequency of generating virtual trajectories")
    parser.add_argument("--num_virtual_trajectories", type=int, default=10,
                        help="number of virtual trajectories to generate")
    parser.add_argument("--diffusion_update_freq", type=int, default=3,
                        help="frequency of updating diffusion model")
    args = parser.parse_args()

    set_seed(args.seed)

    config = TrainUtils.load_config(args.config)
    device = torch.device(args.device)

    ppo = PPO(algo_config=config.algo, obs_config=config.obs, global_config=config.global_config,
              obs_key_shapes=config.obs_key_shapes, ac_dim=config.ac_dim, device=device,
              virtual_trajectory_freq=args.virtual_trajectory_freq,
              num_virtual_trajectories=args.num_virtual_trajectories,
              diffusion_update_freq=args.diffusion_update_freq)

    ppo._initialize_weights(method=args.init_method)

    ppo.train(num_epochs=config.train.num_epochs, batch_size=config.train.batch_size)

if __name__ == "__main__":
    main()
