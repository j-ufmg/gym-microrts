import numpy as np
import time

from stable_baselines3.common.vec_env import DummyVecEnv

from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSScriptVecEnv

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

env = MicroRTSScriptVecEnv(
    ai2s=[microrts_ai.coacAI],
    max_steps=2000,
    render_theme=2,
    map_path="maps/16x16/basesWorkers16x16.xml",
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
)

env = DummyVecEnv([lambda: env])

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
model.save("ppo_script")

model = PPO.load("ppo_script")

obs = env.reset()

'''for i in range(5_000):
    env.render()
    time.sleep(0.01)

    action = model.predict(obs[0])
    next_obs, reward, done, info = env.step(action)'''

env.close()
