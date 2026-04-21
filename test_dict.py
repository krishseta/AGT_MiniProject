import ray
from ray.rllib.algorithms.ppo import PPOConfig
from env.micro4x_env import Micro4XEnv
from training.ppo_baseline import SingleAgentWrapper
from env.action_mask_model import GridWiseActionMaskModel
from ray.rllib.models import ModelCatalog

import pprint

ModelCatalog.register_custom_model('grid_action_mask', GridWiseActionMaskModel)
ray.init(ignore_reinit_error=True)

config = (
    PPOConfig()
    .environment(SingleAgentWrapper, env_config={'max_turns': 100})
    .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
    .framework('torch')
    .training(train_batch_size=256, minibatch_size=64, num_epochs=2, model={'custom_model': 'grid_action_mask', 'custom_model_config': {'grid_h': 16, 'grid_w': 16, 'num_obs_channels': 7, 'num_action_types': 31}})
)
algo = config.build_algo()

print("Training iter 1")
res = algo.train()
env_runners = res.get('env_runners', {})
print("reward_mean:", env_runners.get('episode_reward_mean'))
print("reward_max:", env_runners.get('episode_reward_max'))

print("Training iter 2")
res = algo.train()
env_runners = res.get('env_runners', {})
print("reward_mean:", env_runners.get('episode_reward_mean'))

ray.shutdown()
