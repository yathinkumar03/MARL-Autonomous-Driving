import gymnasium as gym
import highway_env

import numpy as np
import torch
import time

from dqn_agent import Agent


# -----------------------------------
# SAME CONFIG AS TRAINING
# -----------------------------------

config = {

    "observation": {
        "type": "Kinematics"
    },

    "action": {
        "type": "DiscreteMetaAction"
    },

    "vehicles_count": 20,

    "duration": 40,

    "initial_vehicle_count": 10,

    "spawn_probability": 0.6,

    "screen_width": 1000,

    "screen_height": 700
}


env = gym.make(
    "intersection-v1",
    render_mode="human",
    config=config
)


# -----------------------------------
# INITIAL OBSERVATION
# -----------------------------------

obs, info = env.reset()

state_size = np.array(obs).flatten().shape[0]

print("State Size:", state_size)

action_size = 5


# -----------------------------------
# LOAD AGENT
# -----------------------------------

agent = Agent(
    state_size,
    action_size
)


# -----------------------------------
# LOAD TRAINED MODEL
# -----------------------------------

agent.model.load_state_dict(
    torch.load(
        "dqn_model.pth",
        map_location=torch.device("cpu")
    )
)

agent.model.eval()


# Disable Exploration

agent.epsilon = 0


# -----------------------------------
# TESTING LOOP
# -----------------------------------

episodes = 5


for episode in range(episodes):

    state, info = env.reset()

    state = np.array(state).flatten()

    done = False

    total_reward = 0

    while not done:

        env.render()

        # Predict Best Action

        action = agent.act(state)

        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        next_state = np.array(next_state).flatten()

        state = next_state

        total_reward += reward

        time.sleep(0.03)

    print(
        f"Test Episode: {episode + 1}, "
        f"Reward: {total_reward:.2f}"
    )


env.close()