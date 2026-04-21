"""Quick test to validate the MultiAgentEnv wrapper works end-to-end."""
import ray
import numpy as np

ray.init(num_cpus=2, num_gpus=0, ignore_reinit_error=True)

from training.rllib_wrapper import Micro4XMultiAgentEnv

env = Micro4XMultiAgentEnv({"grid_h": 24, "grid_w": 24})
obs, info = env.reset()
print("Reset OK. Agents:", list(obs.keys()))

for step in range(10):
    acts = {a: env.action_space.sample() for a in obs}
    obs, r, t, tr, i = env.step(acts)
    
    print(f"Step {step}: agents={list(obs.keys())}, __all__term={t['__all__']}, __all__trunc={tr['__all__']}")
    
    for a in obs:
        o = obs[a]
        valid = env.observation_space.contains(o)
        if not valid:
            print(f"  {a} INVALID OBS!")
            print(f"    obs dtype={o['observation'].dtype} range=[{np.min(o['observation'])}, {np.max(o['observation'])}]")
            print(f"    mask dtype={o['action_mask'].dtype} range=[{np.min(o['action_mask'])}, {np.max(o['action_mask'])}]")
    
    if t["__all__"] or tr["__all__"]:
        print("Game ended!")
        break

ray.shutdown()
print("All OK!")
