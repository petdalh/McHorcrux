import numpy as np
from dotenv import load_dotenv
from pathlib import Path
import matplotlib.pyplot as plt
from numpy_core.rl_training.colregs_gym import ColregsGym
from numpy_core.controllers.adaptive_seakeeping import (
    MRACShipController,
    heading_to_goal,
)
from ddpg.agent import Agent
import os

# Remember setting PYTHONPATH in .env file
load_dotenv()
home = os.getenv("PYTHONPATH")
if not home:
    raise EnvironmentError("PYTHONPATH not set in environment variables.")

from numpy_core.rl_training.reachable_sets import preload_reachable_sets, select_reachable_set

dt=0.08

def plot_learning_curve(scores, filename):
    """
    Plots the total reward per episode and a running average.
    """
    plt.figure(figsize=(10, 5))
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 10):(i + 1)])

    plt.plot(scores, label="Score", alpha=0.3)
    plt.plot(running_avg, label="Running Average (10 eps)", color='red')
    plt.title("Learning Progress")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()


# Environment factory
def make_head_on_env(dt=0.08, render_on=False):
    env = ColregsGym(
        dt=dt,
        grid_width=25,
        grid_height=30,
        render_on=render_on,
        final_plot=True,
    )

    start_pos = (1.5, 7.5, 0.0)
    wave_conditions = (0.05, 1.5, 0)

    env.set_head_on_task(
        start_position=start_pos,
        wave_conditions=wave_conditions,
        separation=30,
        target_speed=0.3,
        goal_ahead_distance=25.0,
        collision_radius=1.0,
        simtime=150.0,
    )
    return env


# State encoding for RL
def state_to_obs(state: dict) -> np.ndarray:
    """
    Build the observation vector for the RL agent.
    Example: [n, e, psi, u, v, r, g_n, g_e, tgt_n, tgt_e]
    """
    eta = state["eta"]
    nu = state["nu"]
    g_n, g_e, _ = state["goal"]

    if "target_eta" in state and state["target_eta"] is not None:
        tgt = state["target_eta"][:2]
    else:
        tgt = np.array([0.0, 0.0])

    return np.concatenate([eta[:2], [eta[-1]], nu, [g_n, g_e], tgt])


def decode_action_to_refs(action: np.ndarray, state: dict):
    """
    Map RL action in [-1,1]^2 -> (psi_d, u_d).
    """
    a_h, a_u = action  # both assumed in [-1, 1]

    n, e = state["eta"][:2]
    g_n, g_e, _ = state["goal"]
    psi_goal = heading_to_goal(n, e, g_n, g_e)

    # Heading deviation allowed from the straight-line-to-goal direction
    dpsi_max = np.deg2rad(45.0)
    psi_d = psi_goal + a_h * dpsi_max

    # Desired speed in [0, u_max]
    u_max = 1.0
    u_d = 0.5 * (a_u + 1.0) * u_max  # map [-1,1] -> [0, u_max]

    return psi_d, u_d


# Training loop, just set episodes to a low number for quick tests
# and for checking the simulation behavior.
def train(num_episodes=300, dt=0.08):
    # env just for agent init (no rendering)
    init_env = make_head_on_env(dt=dt, render_on=False)
    init_env.v_min, init_env.v_max = -1.0, 1.0
    init_env.r_min, init_env.r_max = -1.0, 1.0

    init_state = init_env.get_state()
    init_obs = state_to_obs(init_state)
    obs_dim = init_obs.shape[0]

    score_history = []

    ellipsoids_Ab_dicts = preload_reachable_sets("voyager")

    agent = Agent(
        alpha=0.000025,
        beta=0.00025,
        input_dims=obs_dim,
        tau=0.01,
        env=init_env,
        n_actions=2,
        chkpt_dir=str(Path(home) / "McHorcrux" / "checkpoints" / "ddpg",)
    )

    if len(
            os.listdir(
                str(Path(home) / "McHorcrux" / "checkpoints" / "ddpg"))
    ) != 0:
        print("Loading existing models...")
        agent.load_models()

    env = None

    for episode in range(num_episodes):
        # Render only the episode
        render_this = (episode == num_episodes - 1)

        env = make_head_on_env(dt=dt, render_on=render_this)
        env.v_min, env.v_max = -1.0, 1.0
        env.r_min, env.r_max = -1.0, 1.0

        controller = MRACShipController(
            dt=dt)  # fresh controller per episode (optional)

        state = env.get_state()
        obs = state_to_obs(state)
        done = False
        score = 0.0

        ellipsoids_Ab_dict = select_reachable_set(ellipsoids_Ab_dicts, env.head_on_speed)

        while not done:
            action_rl = agent.choose_action(obs)
            psi_d, u_d = decode_action_to_refs(action_rl, state)
            tau = controller.compute_action_minimal(state, psi_d, u_d)

            new_state, done, info, env_reward = env.step(tau)

            reward = env_reward
            if done:
                reason = info.get("reason", "")
                if reason == "head_on_collision":
                    reward -= 50.0
                elif reason == "goal_reached":
                    reward += 50.0
                elif reason == "head_on_time_limit":
                    reward -= 10.0

            new_obs = state_to_obs(new_state)
            agent.remember(obs, action_rl, reward, new_obs, int(done))
            agent.learn()

            obs = new_obs
            state = new_state
            score += reward
            score_history.append(score)

        score_history.append(score)

        print(f"Episode {episode}, return {score:.2f}")
        agent.save_models()

    # After training: plot last episode trajectory
    env.plot_trajectory()

    plot_learning_curve(score_history, "learning_progress.png")


if __name__ == "__main__":
    train()
