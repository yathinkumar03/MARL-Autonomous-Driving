import matplotlib.pyplot as plt


# -----------------------------------
# Reward Plot
# -----------------------------------

def plot_rewards(rewards, agent_id):

    plt.figure(figsize=(10, 5))

    plt.plot(

        rewards,

        label=f"Agent {agent_id} Reward"
    )

    plt.xlabel("Episodes")

    plt.ylabel("Total Reward")

    plt.title(
        f"Agent {agent_id} Reward vs Episodes"
    )

    plt.legend()

    plt.grid()

    plt.savefig(

        f"reward_plot_agent{agent_id}.png"
    )

    plt.close()


# -----------------------------------
# Moving Average Plot
# -----------------------------------

def plot_moving_average(
    avg_rewards,
    agent_id
):

    plt.figure(figsize=(10, 5))

    plt.plot(

        avg_rewards,

        label=f"Agent {agent_id} Moving Average"
    )

    plt.xlabel("Episodes")

    plt.ylabel("Average Reward")

    plt.title(
        f"Agent {agent_id} Moving Average"
    )

    plt.legend()

    plt.grid()

    plt.savefig(

        f"moving_average_agent{agent_id}.png"
    )

    plt.close()


# -----------------------------------
# Collision Plot
# -----------------------------------

def plot_collisions(
    collisions,
    agent_id
):

    plt.figure(figsize=(10, 5))

    plt.plot(

        collisions,

        label=f"Agent {agent_id} Collisions"
    )

    plt.xlabel("Episodes")

    plt.ylabel("Collision Count")

    plt.title(
        f"Agent {agent_id} Collision Analysis"
    )

    plt.legend()

    plt.grid()

    plt.savefig(

        f"collision_plot_agent{agent_id}.png"
    )

    plt.close()


# -----------------------------------
# Lane Violation Plot
# -----------------------------------

def plot_lane_violations(
    lane_violations,
    agent_id
):

    plt.figure(figsize=(10, 5))

    plt.plot(

        lane_violations,

        label=f"Agent {agent_id} Lane Violations"
    )

    plt.xlabel("Episodes")

    plt.ylabel("Lane Violations")

    plt.title(
        f"Agent {agent_id} Lane Discipline"
    )

    plt.legend()

    plt.grid()

    plt.savefig(

        f"lane_plot_agent{agent_id}.png"
    )

    plt.close()


# -----------------------------------
# Average Speed Plot
# -----------------------------------

def plot_avg_speed(
    avg_speeds,
    agent_id
):

    plt.figure(figsize=(10, 5))

    plt.plot(

        avg_speeds,

        label=f"Agent {agent_id} Avg Speed"
    )

    plt.xlabel("Episodes")

    plt.ylabel("Average Speed")

    plt.title(
        f"Agent {agent_id} Speed Analysis"
    )

    plt.legend()

    plt.grid()

    plt.savefig(

        f"speed_plot_agent{agent_id}.png"
    )

    plt.close()


# -----------------------------------
# Front Distance Plot
# -----------------------------------

def plot_front_distance(
    front_distances,
    agent_id
):

    plt.figure(figsize=(10, 5))

    plt.plot(

        front_distances,

        label=f"Agent {agent_id} Front Distance"
    )

    plt.xlabel("Episodes")

    plt.ylabel("Distance")

    plt.title(
        f"Agent {agent_id} Front Distance"
    )

    plt.legend()

    plt.grid()

    plt.savefig(

        f"front_distance_agent{agent_id}.png"
    )

    plt.close()