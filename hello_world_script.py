import numpy as np
import time

from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSScriptVecEnv

n_envs = 1

try:
    env = MicroRTSScriptVecEnv(
        ai2s=[microrts_ai.coacAI for _ in range(n_envs)],
        max_steps=2000,
        render_theme=2,
        map_path="maps/16x16/basesWorkers16x16.xml",
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
    )

    env.action_space.seed(0)
    env.reset()
    for i in range(10000):
        env.render()
        time.sleep(0.01)
        action = [env.action_space.sample() for _ in range(n_envs)]
        next_obs, reward, done, info = env.step(action)
        if done.any():
            print(reward)
    env.close()
except Exception as e:
    e.printStackTrace()