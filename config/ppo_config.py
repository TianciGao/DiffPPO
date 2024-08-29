# import torch.nn as nn
# from robomimic.config.base_config import BaseConfig
#
# class PPOConfig(BaseConfig):
#     ALGO_NAME = "ppo"
#
#     def __init__(self, dict_to_load=None):
#         super(PPOConfig, self).__init__(dict_to_load=dict_to_load)
#
#     def experiment_config(self):
#         super().experiment_config()
#         self.experiment.name = "ppo_experiment"
#         self.experiment.validate = False
#         self.experiment.logging.terminal_output_to_txt = True
#         self.experiment.logging.log_tb = True
#         self.experiment.logging.log_wandb = False
#         self.experiment.logging.wandb_proj_name = "debug"
#         self.experiment.save.enabled = True
#         self.experiment.save.every_n_seconds = None
#         self.experiment.save.every_n_epochs = 20
#         self.experiment.save.epochs = []
#         self.experiment.save.on_best_validation = False
#         self.experiment.save.on_best_rollout_return = True
#         self.experiment.save.on_best_rollout_success_rate = False
#         self.experiment.epoch_every_n_steps = 5000
#         self.experiment.validation_epoch_every_n_steps = 10
#         self.experiment.env = None
#         self.experiment.additional_envs = None
#         self.experiment.render = False
#         self.experiment.render_video = False
#         self.experiment.keep_all_videos = False
#         self.experiment.video_skip = 5
#         self.experiment.rollout.enabled = True
#         self.experiment.rollout.n = 50
#         self.experiment.rollout.horizon = 1000
#         self.experiment.rollout.rate = 1
#         self.experiment.rollout.warmstart = 0
#         self.experiment.rollout.terminate_on_success = True
#
#     def algo_config(self):
#         self.algo.optim_params.actor.learning_rate.initial = 3e-4
#         self.algo.optim_params.actor.learning_rate.decay_factor = 0.1
#         self.algo.optim_params.actor.learning_rate.epoch_schedule = []
#         self.algo.optim_params.actor.regularization.L2 = 0.00
#         self.algo.optim_params.actor.start_epoch = -1
#         self.algo.optim_params.actor.end_epoch = -1
#
#         self.algo.optim_params.critic.learning_rate.initial = 3e-4
#         self.algo.optim_params.critic.learning_rate.decay_factor = 0.1
#         self.algo.optim_params.critic.learning_rate.epoch_schedule = []
#         self.algo.optim_params.critic.regularization.L2 = 0.00
#         self.algo.optim_params.critic.start_epoch = -1
#         self.algo.optim_params.critic.end_epoch = -1
#
#         self.algo.discount = 0.99
#         self.algo.eps_clip = 0.2
#         self.algo.ppo_update_steps = 4
#         self.algo.lamda = 0.95
#         self.algo.gae_lambda = 0.95
#         self.algo.critic.use_huber = False
#         self.algo.critic.max_gradient_norm = None
#         self.algo.critic.layer_dims = (300, 400)
#         self.algo.actor.enabled = True
#         self.algo.actor.layer_dims = (300, 400)
#
#         self.algo.value_loss_coef = 0.5
#         self.algo.entropy_coef = 0.01
#         self.algo.max_grad_norm = 0.5
#
#         # Add encoder_kwargs configuration
#         self.algo.encoder_kwargs = {
#             "obs": {
#                 "core_class": "ObservationEncoder",
#                 "feature_activation": "ReLU"  # Use a string reference instead of direct callable
#             }
#         }
#
#         # Add bc_loss_weight configuration
#         self.algo.bc_loss_weight = 0.1
#
#     def train_config(self):
#         super().train_config()
#         self.train.data = "/home/tianci/mimicgen_project/robomimic/datasets/d4rl/converted/walker2d_medium_expert_v2.hdf5"
#         self.train.output_dir = "../ppo_trained_models/d4rl/ppo/walker2d-medium-expert-v2/trained_models"
#         self.train.num_data_workers = 0
#         self.train.hdf5_cache_mode = "all"
#         self.train.hdf5_use_swmr = True
#         self.train.hdf5_load_next_obs = True
#         self.train.hdf5_normalize_obs = False
#         self.train.hdf5_filter_key = None
#         self.train.hdf5_validation_filter_key = None
#         self.train.seq_length = 1
#         self.train.pad_seq_length = True
#         self.train.frame_stack = 1
#         self.train.pad_frame_stack = True
#         self.train.dataset_keys = ["actions", "rewards", "dones"]
#         self.train.goal_mode = None
#         self.train.cuda = True
#         self.train.batch_size = 256
#         self.train.num_epochs = 200
#         self.train.seed = 1
#
#     def observation_config(self):
#         super().observation_config()
#         self.observation.modalities.obs.low_dim = ["flat"]
#         self.observation.modalities.obs.rgb = []
#         self.observation.modalities.obs.depth = []
#         self.observation.modalities.obs.scan = []
#         self.observation.modalities.goal.low_dim = []
#         self.observation.modalities.goal.rgb = []
#         self.observation.modalities.goal.depth = []
#         self.observation.modalities.goal.scan = []
#         self.observation.encoder.low_dim.core_class = None
#         self.observation.encoder.low_dim.core_kwargs = {}
#         self.observation.encoder.rgb.core_class = "VisualCore"
#         self.observation.encoder.rgb.core_kwargs = {}
#         self.observation.encoder.depth.core_class = "VisualCore"
#         self.observation.encoder.depth.core_kwargs = {}
#         self.observation.encoder.scan.core_class = "ScanCore"
#         self.observation.encoder.scan.core_kwargs = {}
#
#     def meta_config(self):
#         super().meta_config()
#         self.meta.hp_base_config_file = None
#         self.meta.hp_keys = []
#         self.meta.hp_values = []

import torch.nn as nn
from robomimic.config.base_config import BaseConfig, Config

class SynthERConfig(Config):
    def __init__(self):
        super().__init__()
        self.enabled = False
        self.gin_config_files = []
        self.results_folder = "./results"
        self.num_sample_steps = 128
        self.sample_batch_size = 100000
        self.save_samples = True
        self.save_num_samples = 5000000
        self.save_file_name = "5m_samples.npz"

class PPOConfig(BaseConfig):
    ALGO_NAME = "ppo"

    def __init__(self, dict_to_load=None):
        super(PPOConfig, self).__init__(dict_to_load=dict_to_load)

    def experiment_config(self):
        super().experiment_config()
        self.experiment.name = "ppo_experiment"
        self.experiment.validate = False
        self.experiment.logging.terminal_output_to_txt = True
        self.experiment.logging.log_tb = True
        self.experiment.logging.log_wandb = False
        self.experiment.logging.wandb_proj_name = "debug"
        self.experiment.save.enabled = True
        self.experiment.save.every_n_seconds = None
        self.experiment.save.every_n_epochs = 20
        self.experiment.save.epochs = []
        self.experiment.save.on_best_validation = False
        self.experiment.save.on_best_rollout_return = True
        self.experiment.save.on_best_rollout_success_rate = False
        self.experiment.epoch_every_n_steps = 5000
        self.experiment.validation_epoch_every_n_steps = 10
        self.experiment.env = None
        self.experiment.additional_envs = None
        self.experiment.render = False
        self.experiment.render_video = False
        self.experiment.keep_all_videos = False
        self.experiment.video_skip = 5
        self.experiment.rollout.enabled = True
        self.experiment.rollout.n = 50
        self.experiment.rollout.horizon = 1000
        self.experiment.rollout.rate = 1
        self.experiment.rollout.warmstart = 0
        self.experiment.rollout.terminate_on_success = True

    def algo_config(self):
        super().algo_config()
        self.algo.optim_params.actor.learning_rate.initial = 3e-4
        self.algo.optim_params.actor.learning_rate.decay_factor = 0.1
        self.algo.optim_params.actor.learning_rate.epoch_schedule = []
        self.algo.optim_params.actor.regularization.L2 = 0.00
        self.algo.optim_params.actor.start_epoch = -1
        self.algo.optim_params.actor.end_epoch = -1

        self.algo.optim_params.critic.learning_rate.initial = 3e-4
        self.algo.optim_params.critic.learning_rate.decay_factor = 0.1
        self.algo.optim_params.critic.learning_rate.epoch_schedule = []
        self.algo.optim_params.critic.regularization.L2 = 0.00
        self.algo.optim_params.critic.start_epoch = -1
        self.algo.optim_params.critic.end_epoch = -1

        self.algo.discount = 0.99
        self.algo.eps_clip = 0.2
        self.algo.ppo_update_steps = 4
        self.algo.lamda = 0.95
        self.algo.gae_lambda = 0.95
        self.algo.critic.use_huber = False
        self.algo.critic.max_gradient_norm = None
        self.algo.critic.layer_dims = (300, 400)
        self.algo.actor.enabled = True
        self.algo.actor.layer_dims = (300, 400)

        self.algo.value_loss_coef = 0.5
        self.algo.entropy_coef = 0.01
        self.algo.max_grad_norm = 0.5

        self.algo.encoder_kwargs = {
            "obs": {
                "core_class": "ObservationEncoder",
                "feature_activation": "ReLU"
            }
        }

        self.algo.bc_loss_weight = 0.1

        self.algo.synth_er = SynthERConfig()
        self.algo.synth_er.enabled = True
        self.algo.synth_er.gin_config_files = ["/home/tianci/mimicgen_project/robomimic/config/resmlp_denoiser.gin"]
        self.algo.synth_er.results_folder = "./results"
        self.algo.synth_er.num_sample_steps = 128
        self.algo.synth_er.sample_batch_size = 100000
        self.algo.synth_er.save_samples = True
        self.algo.synth_er.save_num_samples = 5000000
        self.algo.synth_er.save_file_name = "5m_samples.npz"

    def train_config(self):
        super().train_config()
        self.train.data = "/home/tianci/mimicgen_project/robomimic/datasets/d4rl/converted/walker2d_medium_expert_v2.hdf5"
        self.train.output_dir = "../ppo_trained_models/d4rl/ppo/walker2d-medium-expert-v2/trained_models"
        self.train.num_data_workers = 0
        self.train.hdf5_cache_mode = "all"
        self.train.hdf5_use_swmr = True
        self.train.hdf5_load_next_obs = True
        self.train.hdf5_normalize_obs = False
        self.train.hdf5_filter_key = None
        self.train.hdf5_validation_filter_key = None
        self.train.seq_length = 1
        self.train.pad_seq_length = True
        self.train.frame_stack = 1
        self.train.pad_frame_stack = True
        self.train.dataset_keys = ["actions", "rewards", "dones"]
        self.train.goal_mode = None
        self.train.cuda = True
        self.train.batch_size = 256
        self.train.num_epochs = 200
        self.train.seed = 1

    def observation_config(self):
        super().observation_config()
        self.observation.modalities.obs.low_dim = ["flat"]
        self.observation.modalities.obs.rgb = []
        self.observation.modalities.obs.depth = []
        self.observation.modalities.obs.scan = []
        self.observation.modalities.goal.low_dim = []
        self.observation.modalities.goal.rgb = []
        self.observation.modalities.goal.depth = []
        self.observation.modalities.goal.scan = []
        self.observation.encoder.low_dim.core_class = None
        self.observation.encoder.low_dim.core_kwargs = {}
        self.observation.encoder.rgb.core_class = "VisualCore"
        self.observation.encoder.rgb.core_kwargs = {}
        self.observation.encoder.depth.core_class = "VisualCore"
        self.observation.encoder.depth.core_kwargs = {}
        self.observation.encoder.scan.core_class = "ScanCore"
        self.observation.encoder.scan.core_kwargs = {}

    def meta_config(self):
        super().meta_config()
        self.meta.hp_base_config_file = None
        self.meta.hp_keys = []
        self.meta.hp_values = []


