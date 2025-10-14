import numpy as np
from envs.crlite_env import CRLiteEnv

env = CRLiteEnv(seed=123)
obs, info = env.reset()
assert obs.shape == (96, 96, 3), obs.shape
mask = info.get("action_mask", np.ones(9, dtype=bool))
total = 0.0
for step in range(50):
    valid = np.flatnonzero(mask)
    a = int(np.random.choice(valid)) if len(valid) else 0
    obs, r, term, trunc, info = env.step(a)
    total += r
    mask = info.get("action_mask", np.ones(9, dtype=bool))
    if term or trunc:
        break
print(f"OK: steps={step+1}, return={total:.2f}, result={info.get('result')}")
