import ray
from training.ppo_baseline import SingleAgentWrapper
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from env.action_mask_model import GridWiseActionMaskModel

ModelCatalog.register_custom_model("grid_action_mask", GridWiseActionMaskModel)
ray.init(ignore_reinit_error=True, num_cpus=2)
config = PPOConfig().environment(SingleAgentWrapper, env_config={"seed": 42}).api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False).framework("torch").training(train_batch_size=200, minibatch_size=50, num_epochs=1, model={"custom_model": "grid_action_mask", "custom_model_config": {"grid_h":16, "grid_w":16, "num_obs_channels":7, "num_action_types":31}})
algo = config.build_algo()
res = algo.train()

print('episode_reward_mean:', res.get('episode_reward_mean'))
print('episode_len_mean:', res.get('episode_len_mean'))

if 'env_runners' in res:
    print('ENV RUNNERS episode_reward_mean:', res['env_runners'].get('episode_reward_mean'))
    print('ENV RUNNERS keys:', res['env_runners'].keys())
