import gymnasium as gym
import highway_env

import numpy as np
import torch
import pygame
import time

from dqn_agent import Agent

from plots import (
    plot_rewards,
    plot_collisions,
    plot_lane_violations,
    plot_avg_speed,
    plot_front_distance
)


# -----------------------------------
# Initialize pygame
# -----------------------------------

pygame.init()

font = pygame.font.SysFont(
    "Arial",
    30
)


# -----------------------------------
# Environment Configuration
# -----------------------------------

config = {

    "observation": {
        "type": "Kinematics"
    },

    "action": {

        "type": "MultiAgentAction",

        "action_config": {

            "type": "DiscreteMetaAction"
        }
    },

    "vehicles_count": 20,

    "controlled_vehicles": 3,

    "duration": 40,

    "screen_width": 1000,

    "screen_height": 700
}


# -----------------------------------
# Create Environment
# -----------------------------------

env = gym.make(
    "intersection-v1",
    render_mode="human",
    config=config
)


# -----------------------------------
# Initial Observation
# -----------------------------------

obs, info = env.reset()

state_size = np.array(
    obs[0]
).flatten().shape[0]

action_size = 5

NUM_AGENTS = 3


# -----------------------------------
# Load Agents
# -----------------------------------

agents = []

for i in range(NUM_AGENTS):

    agent = Agent(
        state_size,
        action_size
    )

    agent.model.load_state_dict(

        torch.load(
            f"dqn_agent_{i}.pth"
        )
    )

    # Disable exploration

    agent.epsilon = 0

    agents.append(agent)

print(
    "\nAll trained MARL agents loaded successfully!"
)


# -----------------------------------
# Evaluation Parameters
# -----------------------------------

episodes = 10


# -----------------------------------
# Per-Agent Metrics
# -----------------------------------

agent_rewards = [

    [],
    [],
    []
]

agent_collisions = [

    [],
    [],
    []
]

agent_lane_violations = [

    [],
    [],
    []
]

agent_avg_speeds = [

    [],
    [],
    []
]

agent_front_distances = [

    [],
    [],
    []
]


# -----------------------------------
# Evaluation Loop
# -----------------------------------

for episode in range(episodes):

    states, info = env.reset()

    states = [

        np.array(state).flatten()

        for state in states
    ]

    agent_total_rewards = [

        0,
        0,
        0
    ]

    agent_collision_counts = [

        0,
        0,
        0
    ]

    agent_lane_counts = [

        0,
        0,
        0
    ]

    agent_speed_sums = [

        0,
        0,
        0
    ]

    agent_front_distance_sums = [

        0,
        0,
        0
    ]

    step_count = 0

    done = False

    while not done:

        env.render()

        # -----------------------------------
        # Generate Actions
        # -----------------------------------

        actions = []

        for i in range(NUM_AGENTS):

            action = agents[i].act(
                states[i]
            )

            actions.append(action)

        # -----------------------------------
        # Environment Step
        # -----------------------------------

        next_states, rewards_step, terminated, truncated, info = env.step(
            tuple(actions)
        )

        done = terminated or truncated

        next_states = [

            np.array(state).flatten()

            for state in next_states
        ]

        # -----------------------------------
        # Per-Agent Evaluation
        # -----------------------------------

        for i in range(NUM_AGENTS):

            reward = rewards_step

            speed = info.get(
                "speed",
                0
            )

            front_distance = info.get(
                "distance",
                100
            )

            agent_speed_sums[i] += speed

            agent_front_distance_sums[i] += (
                front_distance
            )

            step_count += 1

            # Collision

            if info.get("crashed"):

                agent_collision_counts[i] += 1

                reward -= 50

            # Lane Violation

            if info.get("offroad"):

                agent_lane_counts[i] += 1

                reward -= 20

            # Speed Reward

            reward += speed * 0.01

            agent_total_rewards[i] += reward

        states = next_states

        time.sleep(0.02)

    # -----------------------------------
    # Store Metrics
    # -----------------------------------

    for i in range(NUM_AGENTS):

        avg_speed = 0

        if step_count > 0:

            avg_speed = (
                agent_speed_sums[i]
                / step_count
            )

        avg_front_distance = 0

        if step_count > 0:

            avg_front_distance = (
                agent_front_distance_sums[i]
                / step_count
            )

        agent_rewards[i].append(
            agent_total_rewards[i]
        )

        agent_collisions[i].append(
            agent_collision_counts[i]
        )

        agent_lane_violations[i].append(
            agent_lane_counts[i]
        )

        agent_avg_speeds[i].append(
            avg_speed
        )

        agent_front_distances[i].append(
            avg_front_distance
        )

    # -----------------------------------
    # Episode Summary
    # -----------------------------------

    print(
        f"\nEvaluation Episode {episode + 1}"
    )

    for i in range(NUM_AGENTS):

        print(

            f"Agent {i + 1} | "
            f"Reward: {agent_total_rewards[i]:.2f} | "
            f"Collisions: {agent_collision_counts[i]} | "
            f"Lane Violations: {agent_lane_counts[i]}"
        )


# -----------------------------------
# Close Environment
# -----------------------------------

env.close()

pygame.quit()


# -----------------------------------
# Generate Evaluation Plots
# -----------------------------------
# -----------------------------------
# Generate Evaluation Plots
# -----------------------------------

for i in range(NUM_AGENTS):

    avg_rewards = moving_average(
        agent_rewards[i]
    )

    # -----------------------------------
    # Reward Plot
    # -----------------------------------

    plot_rewards(

        agent_rewards[i],

        i + 1
    )

    # -----------------------------------
    # Moving Average Plot
    # -----------------------------------

    plot_moving_average(

        avg_rewards,

        i + 1
    )

    # -----------------------------------
    # Collision Plot
    # -----------------------------------

    plot_collisions(

        agent_collisions[i],

        i + 1
    )

    # -----------------------------------
    # Lane Violation Plot
    # -----------------------------------

    plot_lane_violations(

        agent_lane_violations[i],

        i + 1
    )

    # -----------------------------------
    # Speed Plot
    # -----------------------------------

    plot_avg_speed(

        agent_avg_speeds[i],

        i + 1
    )

    # -----------------------------------
    # Front Distance Plot
    # -----------------------------------

    plot_front_distance(

        agent_front_distances[i],

        i + 1
    )

print(
    "\nEvaluation plots generated successfully!"
)