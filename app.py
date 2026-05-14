import sys
import streamlit as st
import subprocess
import os
import json


# -----------------------------------
# Streamlit Page Config
# -----------------------------------

st.set_page_config(

    page_title="MARL Autonomous Driving",

    layout="wide"
)


# -----------------------------------
# Title
# -----------------------------------

st.title(
    "Multi-Agent Reinforcement Learning Autonomous Driving"
)

st.markdown(
    "Decentralized MARL Autonomous Driving Dashboard"
)


# -----------------------------------
# Global Environment Configuration
# -----------------------------------

st.sidebar.header(
    "Global Environment Configuration"
)

episodes = st.sidebar.slider(

    "Episodes",

    1,

    500,

    50
)

vehicles_count = st.sidebar.slider(

    "Traffic Vehicles",

    5,

    50,

    20
)

controlled_vehicles = st.sidebar.slider(

    "Controlled Vehicles",

    1,

    5,

    3
)

spawn_probability = st.sidebar.slider(

    "Spawn Probability",

    0.1,

    1.0,

    0.6
)

duration = st.sidebar.slider(

    "Environment Duration",

    10,

    100,

    40
)


# -----------------------------------
# Agent 1 Configuration
# -----------------------------------

st.sidebar.header("Agent 1 Configuration")

agent1_learning_rate = st.sidebar.number_input(

    "Agent1 Learning Rate",

    value=0.001,

    format="%.5f"
)

agent1_gamma = st.sidebar.slider(

    "Agent1 Gamma",

    0.1,

    1.0,

    0.99
)

agent1_collision_penalty = st.sidebar.slider(

    "Agent1 Collision Penalty",

    -100,

    0,

    -50
)

agent1_lane_penalty = st.sidebar.slider(

    "Agent1 Lane Penalty",

    -50,

    0,

    -20
)

agent1_speed_reward = st.sidebar.slider(

    "Agent1 Speed Reward",

    0,

    20,

    2
)


# -----------------------------------
# Agent 2 Configuration
# -----------------------------------

st.sidebar.header("Agent 2 Configuration")

agent2_learning_rate = st.sidebar.number_input(

    "Agent2 Learning Rate",

    value=0.0005,

    format="%.5f"
)

agent2_gamma = st.sidebar.slider(

    "Agent2 Gamma",

    0.1,

    1.0,

    0.95
)

agent2_collision_penalty = st.sidebar.slider(

    "Agent2 Collision Penalty",

    -100,

    0,

    -40
)

agent2_lane_penalty = st.sidebar.slider(

    "Agent2 Lane Penalty",

    -50,

    0,

    -15
)

agent2_speed_reward = st.sidebar.slider(

    "Agent2 Speed Reward",

    0,

    20,

    3
)


# -----------------------------------
# Agent 3 Configuration
# -----------------------------------

st.sidebar.header("Agent 3 Configuration")

agent3_learning_rate = st.sidebar.number_input(

    "Agent3 Learning Rate",

    value=0.0008,

    format="%.5f"
)

agent3_gamma = st.sidebar.slider(

    "Agent3 Gamma",

    0.1,

    1.0,

    0.97
)

agent3_collision_penalty = st.sidebar.slider(

    "Agent3 Collision Penalty",

    -100,

    0,

    -60
)

agent3_lane_penalty = st.sidebar.slider(

    "Agent3 Lane Penalty",

    -50,

    0,

    -25
)

agent3_speed_reward = st.sidebar.slider(

    "Agent3 Speed Reward",

    0,

    20,

    2
)


# -----------------------------------
# Main Controls
# -----------------------------------

col1, col2 = st.columns(2)

with col1:

    train_button = st.button(
        "Start MARL Training"
    )

with col2:

    evaluate_button = st.button(
        "Run Evaluation"
    )


# -----------------------------------
# Live Simulation Section
# -----------------------------------

st.header(
    "Live MARL Simulation"
)

simulation_placeholder = st.empty()


# -----------------------------------
# Live Metrics
# -----------------------------------

st.header(
    "Live Training Metrics"
)

metrics_placeholder = st.empty()


# -----------------------------------
# Start Training
# -----------------------------------

if train_button:

    st.success(
        "MARL Training Started"
    )

    env_vars = os.environ.copy()

    # -----------------------------------
    # Global Parameters
    # -----------------------------------

    env_vars["EPISODES"] = str(
        episodes
    )

    env_vars["VEHICLES_COUNT"] = str(
        vehicles_count
    )

    env_vars["CONTROLLED_VEHICLES"] = str(
        controlled_vehicles
    )

    env_vars["SPAWN_PROBABILITY"] = str(
        spawn_probability
    )

    env_vars["DURATION"] = str(
        duration
    )

    # -----------------------------------
    # Agent 1 Parameters
    # -----------------------------------

    env_vars["AGENT1_LR"] = str(
        agent1_learning_rate
    )

    env_vars["AGENT1_GAMMA"] = str(
        agent1_gamma
    )

    env_vars["AGENT1_COLLISION"] = str(
        agent1_collision_penalty
    )

    env_vars["AGENT1_LANE"] = str(
        agent1_lane_penalty
    )

    env_vars["AGENT1_SPEED"] = str(
        agent1_speed_reward
    )

    # -----------------------------------
    # Agent 2 Parameters
    # -----------------------------------

    env_vars["AGENT2_LR"] = str(
        agent2_learning_rate
    )

    env_vars["AGENT2_GAMMA"] = str(
        agent2_gamma
    )

    env_vars["AGENT2_COLLISION"] = str(
        agent2_collision_penalty
    )

    env_vars["AGENT2_LANE"] = str(
        agent2_lane_penalty
    )

    env_vars["AGENT2_SPEED"] = str(
        agent2_speed_reward
    )

    # -----------------------------------
    # Agent 3 Parameters
    # -----------------------------------

    env_vars["AGENT3_LR"] = str(
        agent3_learning_rate
    )

    env_vars["AGENT3_GAMMA"] = str(
        agent3_gamma
    )

    env_vars["AGENT3_COLLISION"] = str(
        agent3_collision_penalty
    )

    env_vars["AGENT3_LANE"] = str(
        agent3_lane_penalty
    )

    env_vars["AGENT3_SPEED"] = str(
        agent3_speed_reward
    )

    # -----------------------------------
    # Launch Training
    # -----------------------------------
        # -----------------------------------
    # Save Dynamic Config
    # -----------------------------------

    config_data = {

        "episodes": episodes,

        "vehicles_count": vehicles_count,

        "controlled_vehicles": controlled_vehicles,

        "spawn_probability": spawn_probability,

        "duration": duration,

        "agents": [

    {

        "learning_rate": agent1_learning_rate,

        "gamma": agent1_gamma,

        "epsilon": 1.0,

        "epsilon_decay": 0.995,

        "collision_penalty": agent1_collision_penalty,

        "lane_penalty": agent1_lane_penalty,

        "speed_reward": agent1_speed_reward,

        "signal_reward": 2,

        "safe_distance_reward": 3,

        "overspeed_penalty": -4,

        "emergency_brake_penalty": -10
    },

    {

        "learning_rate": agent2_learning_rate,

        "gamma": agent2_gamma,

        "epsilon": 1.0,

        "epsilon_decay": 0.994,

        "collision_penalty": agent2_collision_penalty,

        "lane_penalty": agent2_lane_penalty,

        "speed_reward": agent2_speed_reward,

        "signal_reward": 2,

        "safe_distance_reward": 4,

        "overspeed_penalty": -5,

        "emergency_brake_penalty": -8
    },

    {

        "learning_rate": agent3_learning_rate,

        "gamma": agent3_gamma,

        "epsilon": 1.0,

        "epsilon_decay": 0.996,

        "collision_penalty": agent3_collision_penalty,

        "lane_penalty": agent3_lane_penalty,

        "speed_reward": agent3_speed_reward,

        "signal_reward": 3,

        "safe_distance_reward": 5,

        "overspeed_penalty": -6,

        "emergency_brake_penalty": -12
    }
]
        
    }

    with open(

        "config.json",

        "w"

    ) as f:

        json.dump(

            config_data,

            f,

            indent=4
        )
    process = subprocess.Popen(

    [sys.executable, "main.py"],

    env=env_vars
)

    metrics_placeholder.info(
        "Training in progress..."
    )

    process.wait()

    st.success(
        "Training Completed"
    )


# -----------------------------------
# Run Evaluation
# -----------------------------------

if evaluate_button:

    st.success(
        "Evaluation Started"
    )

    subprocess.run(

    [sys.executable, "evaluate.py"]
)

    st.success(
        "Evaluation Completed"
    )


# ===================================
# AGENT 1 RESULTS
# ===================================

st.header("Agent 1 Analysis")

if os.path.exists("reward_plot_agent1.png"):

    st.image("reward_plot_agent1.png")

if os.path.exists("moving_average_agent1.png"):

    st.image("moving_average_agent1.png")

if os.path.exists("collision_plot_agent1.png"):

    st.image("collision_plot_agent1.png")

if os.path.exists("lane_plot_agent1.png"):

    st.image("lane_plot_agent1.png")

if os.path.exists("speed_plot_agent1.png"):

    st.image("speed_plot_agent1.png")

if os.path.exists("front_distance_agent1.png"):

    st.image("front_distance_agent1.png")


# ===================================
# AGENT 2 RESULTS
# ===================================

st.header("Agent 2 Analysis")

if os.path.exists("reward_plot_agent2.png"):

    st.image("reward_plot_agent2.png")

if os.path.exists("moving_average_agent2.png"):

    st.image("moving_average_agent2.png")

if os.path.exists("collision_plot_agent2.png"):

    st.image("collision_plot_agent2.png")

if os.path.exists("lane_plot_agent2.png"):

    st.image("lane_plot_agent2.png")

if os.path.exists("speed_plot_agent2.png"):

    st.image("speed_plot_agent2.png")

if os.path.exists("front_distance_agent2.png"):

    st.image("front_distance_agent2.png")


# ===================================
# AGENT 3 RESULTS
# ===================================

st.header("Agent 3 Analysis")

if os.path.exists("reward_plot_agent3.png"):

    st.image("reward_plot_agent3.png")

if os.path.exists("moving_average_agent3.png"):

    st.image("moving_average_agent3.png")

if os.path.exists("collision_plot_agent3.png"):

    st.image("collision_plot_agent3.png")

if os.path.exists("lane_plot_agent3.png"):

    st.image("lane_plot_agent3.png")

if os.path.exists("speed_plot_agent3.png"):

    st.image("speed_plot_agent3.png")

if os.path.exists("front_distance_agent3.png"):

    st.image("front_distance_agent3.png")


# -----------------------------------
# Evaluation Metrics
# -----------------------------------

st.header(
    "Evaluation Metrics"
)

st.table(

    {

        "Metric": [

            "Average Reward",

            "Collision Rate",

            "Lane Violations",

            "Average Speed",

            "Safe Distance"
        ],

        "Description": [

            "Overall learning performance",

            "Driving safety",

            "Lane discipline",

            "Driving stability",

            "Obstacle awareness"
        ]
    }
)


# -----------------------------------
# MARL Agent Comparison
# -----------------------------------

st.header(
    "MARL Agent Comparison"
)

st.table(

    {

        "Agent": [

            "Agent 1",

            "Agent 2",

            "Agent 3"
        ],

        "Status": [

            "Active",

            "Active",

            "Active"
        ]
    }
)


# -----------------------------------
# Footer
# -----------------------------------

st.markdown("---")

st.markdown(
    "Decentralized Multi-Agent Reinforcement Learning Dashboard"
)
