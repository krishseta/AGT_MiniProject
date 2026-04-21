import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override

FLOAT_MIN = -1e9


class GridWiseActionMaskModel(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        nn.Module.__init__(self)

        custom = model_config.get("custom_model_config", {})
        self.grid_h = custom.get("grid_h", 24)
        self.grid_w = custom.get("grid_w", 24)
        num_channels = custom.get("num_obs_channels", 7)
        self.num_actions = custom.get("num_action_types", 31)

        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 128, 1),
            nn.ReLU(),
        )

        self.policy_head = nn.Conv2d(128, self.num_actions, 1)

        self.value_pool = nn.AdaptiveAvgPool2d(1)
        self.value_head = nn.Linear(128, 1)

        self._features = None
        self._value = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]["observation"].float()
        action_mask = input_dict["obs"]["action_mask"].float()

        x = obs.permute(0, 3, 1, 2)
        self._features = self.encoder(x)

        logits = self.policy_head(self._features)
        logits = logits.permute(0, 2, 3, 1).reshape(
            -1, self.grid_h * self.grid_w * self.num_actions
        )

        inf_mask = torch.clamp(
            torch.log(action_mask), min=FLOAT_MIN
        )
        masked_logits = logits + inf_mask

        pooled = self.value_pool(self._features).squeeze(-1).squeeze(-1)
        self._value = self.value_head(pooled)

        return masked_logits, state

    @override(TorchModelV2)
    def value_function(self):
        return self._value.squeeze(-1)
