import os


# -----------------------------------
# Environment Configuration
# -----------------------------------

environment_config = {

    "episodes": int(
        os.getenv("EPISODES", 50)
    ),

    "vehicles_count": int(
        os.getenv("VEHICLES_COUNT", 20)
    ),

    "controlled_vehicles": int(
        os.getenv("CONTROLLED_VEHICLES", 3)
    ),

    "duration": int(
        os.getenv("DURATION", 40)
    ),

    "spawn_probability": float(
        os.getenv("SPAWN_PROBABILITY", 0.6)
    ),

    "screen_width": 1000,

    "screen_height": 700
}


# -----------------------------------
# Agent 1 Config
# -----------------------------------

agent1_config = {

    "learning_rate": float(
        os.getenv("AGENT1_LR", 0.001)
    ),

    "gamma": float(
        os.getenv("AGENT1_GAMMA", 0.99)
    ),

    "epsilon": 1.0,

    "epsilon_decay": 0.995,

    "collision_penalty": int(
        os.getenv("AGENT1_COLLISION", -50)
    ),

    "lane_penalty": int(
        os.getenv("AGENT1_LANE", -20)
    ),

    "safe_distance_reward": 3,

    "speed_reward": int(
        os.getenv("AGENT1_SPEED", 2)
    ),

    "signal_reward": 2,

    "overspeed_penalty": -4,

    "emergency_brake_penalty": -10
}


# -----------------------------------
# Agent 2 Config
# -----------------------------------

agent2_config = {

    "learning_rate": float(
        os.getenv("AGENT2_LR", 0.0005)
    ),

    "gamma": float(
        os.getenv("AGENT2_GAMMA", 0.95)
    ),

    "epsilon": 1.0,

    "epsilon_decay": 0.994,

    "collision_penalty": int(
        os.getenv("AGENT2_COLLISION", -40)
    ),

    "lane_penalty": int(
        os.getenv("AGENT2_LANE", -15)
    ),

    "safe_distance_reward": 4,

    "speed_reward": int(
        os.getenv("AGENT2_SPEED", 3)
    ),

    "signal_reward": 2,

    "overspeed_penalty": -5,

    "emergency_brake_penalty": -8
}


# -----------------------------------
# Agent 3 Config
# -----------------------------------

agent3_config = {

    "learning_rate": float(
        os.getenv("AGENT3_LR", 0.0008)
    ),

    "gamma": float(
        os.getenv("AGENT3_GAMMA", 0.97)
    ),

    "epsilon": 1.0,

    "epsilon_decay": 0.996,

    "collision_penalty": int(
        os.getenv("AGENT3_COLLISION", -60)
    ),

    "lane_penalty": int(
        os.getenv("AGENT3_LANE", -25)
    ),

    "safe_distance_reward": 5,

    "speed_reward": int(
        os.getenv("AGENT3_SPEED", 2)
    ),

    "signal_reward": 3,

    "overspeed_penalty": -6,

    "emergency_brake_penalty": -12
}


# -----------------------------------
# Combined Configs
# -----------------------------------

agent_configs = [

    agent1_config,

    agent2_config,

    agent3_config
]