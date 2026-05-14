import gymnasium as gym
import highway_env
import time

env = gym.make(
    "intersection-v1",
    render_mode="human"
)

obs, info = env.reset()

for _ in range(500):

    action = env.action_space.sample()

    obs, reward, done, truncated, info = env.step(action)

    env.render()

    time.sleep(0.03)

    if done or truncated:
        obs, info = env.reset()

env.close()