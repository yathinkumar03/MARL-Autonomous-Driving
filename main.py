import gymnasium as gym
import highway_env

import numpy as np
import time
import torch
import cv2

from dqn_agent import Agent

import json
# -----------------------------------
# Load Dynamic Config
# -----------------------------------

with open(

    "config.json",

    "r"

) as f:

    config_data = json.load(f)

environment_config = {

    "episodes":
        config_data["episodes"],

    "vehicles_count":
        config_data["vehicles_count"],

    "controlled_vehicles":
        config_data["controlled_vehicles"],

    "duration":
        config_data["duration"],

    "spawn_probability":
        config_data["spawn_probability"],

    "screen_width": 1000,

    "screen_height": 700
}

agent_configs = config_data["agents"]

from plots import (
    plot_rewards,
    plot_moving_average,
    plot_collisions,
    plot_lane_violations,
    plot_avg_speed,
    plot_front_distance
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

    "vehicles_count":
        environment_config["vehicles_count"],

    "controlled_vehicles":
        environment_config["controlled_vehicles"],

    "duration":
        environment_config["duration"],

    "initial_vehicle_count": 10,

    "spawn_probability":
        environment_config["spawn_probability"],

    "screen_width":
        environment_config["screen_width"],

    "screen_height":
        environment_config["screen_height"]
}


# -----------------------------------
# Create Environment
# -----------------------------------

env = gym.make(

    "intersection-v1",

    render_mode="rgb_array",

    config=config
)


# -----------------------------------
# Initial Observation
# -----------------------------------

obs, info = env.reset()

print(
    "Controlled Vehicles:",
    len(env.unwrapped.controlled_vehicles)
)

state_size = np.array(
    obs[0]
).flatten().shape[0]

print("\nSTATE SIZE =", state_size)

action_size = 5


# -----------------------------------
# Multi-Agent RL Agents
# -----------------------------------
# -----------------------------------
# Multi-Agent RL Agents
# -----------------------------------

NUM_AGENTS = 3

agents = []

for i in range(NUM_AGENTS):

    agents.append(

        Agent(

            state_size,

            action_size,

            learning_rate=
            agent_configs[i]["learning_rate"],

            gamma=
            agent_configs[i]["gamma"],

            epsilon=
            agent_configs[i]["epsilon"],

            epsilon_decay=
            agent_configs[i]["epsilon_decay"]
        )
    )

print(
    f"{NUM_AGENTS} DQN Agents Initialized"
)


# -----------------------------------
# Training Parameters
# -----------------------------------

episodes = environment_config["episodes"]


# -----------------------------------
# Per-Agent Metrics Storage
# -----------------------------------

agent_rewards = [[], [], []]

agent_collisions = [[], [], []]

agent_lane_violations = [[], [], []]

agent_avg_speeds = [[], [], []]

agent_front_distances = [[], [], []]


# -----------------------------------
# Moving Average Function
# -----------------------------------

def moving_average(data, window_size=10):

    averages = []

    for i in range(len(data)):

        start = max(0, i - window_size + 1)

        window = data[start:i + 1]

        averages.append(
            sum(window) / len(window)
        )

    return averages


# -----------------------------------
# Training Loop
# -----------------------------------

for episode in range(episodes):

    states, info = env.reset()

    states = [

        np.array(state).flatten()

        for state in states
    ]

    agent_total_rewards = [0, 0, 0]

    agent_collision_counts = [0, 0, 0]

    agent_lane_counts = [0, 0, 0]

    agent_speed_sums = [0, 0, 0]

    agent_front_distance_sums = [0, 0, 0]

    step_count = 0

    done = False

    signal_timer = 0

    traffic_signal = "GREEN"

    while not done:

        # -----------------------------------
        # Render RGB Frame
        # -----------------------------------

        frame = env.render()

        frame = np.ascontiguousarray(frame)

        # -----------------------------------
        # Traffic Signal Logic
        # -----------------------------------

        signal_timer += 1

        if signal_timer % 200 < 100:

            traffic_signal = "GREEN"

        else:

            traffic_signal = "RED"

        # -----------------------------------
        # Traffic Signal Overlay
        # -----------------------------------

        signal_text = f"Signal: {traffic_signal}"

        if traffic_signal == "RED":

            color = (255, 0, 0)

        else:

            color = (0, 255, 0)

        cv2.rectangle(

            frame,

            (10, 10),

            (280, 80),

            (0, 0, 0),

            -1
        )

        cv2.putText(

            frame,

            signal_text,

            (20, 55),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            color,

            2,

            cv2.LINE_AA
        )

        # -----------------------------------
        # Multi-Agent Actions
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
        # Process Agents
        # -----------------------------------

        for i in range(NUM_AGENTS):

            agent_config = agent_configs[i]

            reward = rewards_step

            speed = info.get(
                "speed",
                0
            )

            agent_speed_sums[i] += speed

            # -----------------------------------
            # Front Distance
            # -----------------------------------

            front_distance = 100

            if "distance" in info:

                front_distance = info["distance"]

            agent_front_distance_sums[i] += (
                front_distance
            )

            # -----------------------------------
            # Safe Distance Reward
            # -----------------------------------

            if front_distance < 15:

                reward -= 10

            elif 15 <= front_distance <= 40:

                reward += agent_config[
                    "safe_distance_reward"
                ]

            # -----------------------------------
            # Collision Penalty
            # -----------------------------------

            if info.get("crashed"):

                reward += agent_config[
                    "collision_penalty"
                ]

                agent_collision_counts[i] += 1

            # -----------------------------------
            # Lane Penalty
            # -----------------------------------

            if info.get("offroad"):

                reward += agent_config[
                    "lane_penalty"
                ]

                agent_lane_counts[i] += 1

            # -----------------------------------
            # Speed Reward
            # -----------------------------------

            reward += speed * 0.01

            # -----------------------------------
            # Traffic Signal Reward
            # -----------------------------------

            if traffic_signal == "RED":

                if speed > 10:

                    reward -= 5

                else:

                    reward += 1

            if traffic_signal == "GREEN":

                reward += agent_config[
                    "signal_reward"
                ]

            # -----------------------------------
            # Overspeed Penalty
            # -----------------------------------

            if speed > 25:

                reward += agent_config[
                    "overspeed_penalty"
                ]

            # -----------------------------------
            # Emergency Braking
            # -----------------------------------

            if speed > 35:

                reward += agent_config[
                    "emergency_brake_penalty"
                ]

            # -----------------------------------
            # Store Experience
            # -----------------------------------

            agents[i].remember(

                states[i],

                actions[i],

                reward,

                next_states[i],

                done
            )

            agent_total_rewards[i] += reward

        # -----------------------------------
        # Update States
        # -----------------------------------

        states = next_states

        # -----------------------------------
        # Train Agents
        # -----------------------------------

        for i in range(NUM_AGENTS):

            agents[i].replay()

        step_count += 1

        time.sleep(0.02)

    # -----------------------------------
    # Update Target Networks
    # -----------------------------------

    for i in range(NUM_AGENTS):

        agents[i].update_target()

    # -----------------------------------
    # Store Metrics
    # -----------------------------------

    for i in range(NUM_AGENTS):

        average_speed = 0

        if step_count > 0:

            average_speed = (

                agent_speed_sums[i]
                / step_count
            )

        average_front_distance = 0

        if step_count > 0:

            average_front_distance = (

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
            average_speed
        )

        agent_front_distances[i].append(
            average_front_distance
        )

    # -----------------------------------
    # Episode Summary
    # -----------------------------------

    print(f"\nEpisode {episode + 1}")

    for i in range(NUM_AGENTS):

        print(

            f"Agent {i + 1} | "
            f"Reward: {agent_total_rewards[i]:.2f} | "
            f"Collisions: {agent_collision_counts[i]} | "
            f"Lane Violations: {agent_lane_counts[i]} | "
            f"Epsilon: {agents[i].epsilon:.3f}"
        )


# -----------------------------------
# Close Environment
# -----------------------------------

env.close()


# -----------------------------------
# Save Models
# -----------------------------------

for i in range(NUM_AGENTS):

    torch.save(

        agents[i].model.state_dict(),

        f"dqn_agent_{i+1}.pth"
    )

print(
    "\nAll MARL models saved successfully!"
)


# -----------------------------------
# Generate Plots
# -----------------------------------

for i in range(NUM_AGENTS):

    avg_rewards = moving_average(
        agent_rewards[i]
    )

    plot_rewards(

        agent_rewards[i],

        i + 1
    )

    plot_moving_average(

        avg_rewards,

        i + 1
    )

    plot_collisions(

        agent_collisions[i],

        i + 1
    )

    plot_lane_violations(

        agent_lane_violations[i],

        i + 1
    )

    plot_avg_speed(

        agent_avg_speeds[i],

        i + 1
    )

    plot_front_distance(

        agent_front_distances[i],

        i + 1
    )

print("\nAll agent plots generated successfully!")