import numpy as np
from dotenv import load_dotenv
from pathlib import Path
import matplotlib.pyplot as plt
from mchorcrux.numpy_core.rl_training.colregs_gym import ColregsGym
from mchorcrux.numpy_core.controllers.adaptive_seakeeping import (
    MRACShipController,
    heading_to_goal,
)
from mchorcrux.ddpg.agent import Agent
from pacSTL.colregs_spec import CrossingSpec
import os

# Remember setting PYTHONPATH in .env file
load_dotenv()
home = os.getenv("PYTHONPATH")
if not home:
    raise EnvironmentError("PYTHONPATH not set in environment variables.")

from mchorcrux.numpy_core.rl_training.reachable_sets import preload_reachable_sets, select_reachable_set

dt = 0.08


def plot_learning_curve(scores, filename):
    """Plots the total reward per episode and a running average."""
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

def plot_robustness_curve(robustness_history):
    """Plots the STL robustness over time, with a separate subplot for each episode."""
    if len(robustness_history) >= 3:
        print("Plotting only the first 4 episodes for clarity.")
        history_to_plot = robustness_history[:4]
    else: history_to_plot = robustness_history
    num_episodes = len(history_to_plot)
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    axes_flat = axes.flatten()
    
    for i, interval_list in enumerate(history_to_plot):
        # If the interval exists, get .u. Otherwise, insert np.nan
        upper_lims = [interval.u if interval is not None else np.nan for interval in interval_list]
        lower_lims = [interval.l if interval is not None else np.nan for interval in interval_list]
        time_steps = np.arange(len(upper_lims))
        
        ax = axes_flat[i]
        
        ax.plot(time_steps, upper_lims, label="Upper Bound", alpha=0.5)
        ax.plot(time_steps, lower_lims, label="Lower Bound", alpha=0.5)
        
        ax.set_title(f"Episode {i + 1} STL Robustness")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Robustness")
        ax.legend()
        ax.grid(True)
        
    for j in range(num_episodes, 4):
        fig.delaxes(axes_flat[j])
        
    plt.tight_layout()
    plt.savefig("./numpy_core/rl_training/results/robustness_subplots_2x2.png")
    plt.show()


def make_env(dt=0.08, render_on=False, verbose=False, robustness_sampling_rate=10):
    env = ColregsGym(
        dt=dt,
        grid_width=25,
        grid_height=30,
        render_on=render_on,
        final_plot=True,
        verbose=verbose,
        robustness_sampling_rate=robustness_sampling_rate,
    )

    start_pos = (1.5, 7.5, 0.0)
    wave_conditions = (0.05, 1.5, 0)

    env.set_encounter_task(
        start_position=start_pos,
        wave_conditions=wave_conditions,
        encounter_type="crossing",
        separation=30,
        target_speed=0.3,
        goal_ahead_distance=25.0,
        collision_radius=1.0,
        simtime=150.0,
    )
    return env


def decode_action_to_refs(action: np.ndarray, obs: np.ndarray):
    """Map RL action in [-1,1]^2 -> (psi_d, u_d)."""
    a_h, a_u = action
    n, e, psi, u, v, r, g_n, g_e = obs[:8]
    psi_goal = heading_to_goal(n, e, g_n, g_e)
    psi_d = psi_goal + a_h * np.deg2rad(45.0)
    u_d = 0.5 * (a_u + 1.0) * 1.0
    return psi_d, u_d


def train(num_episodes=1, dt=0.08, model_dir="ddpg", verbose=False, robustness_sampling_rate=20):
    crossing_spec = CrossingSpec()
    ellipsoids_Ab_dicts = preload_reachable_sets("voyager")

    env = make_env(dt=dt, render_on=False, verbose=verbose, robustness_sampling_rate=robustness_sampling_rate)
    env.set_spec(crossing_spec)
    env.ellipsoids_Ab_dict = select_reachable_set(ellipsoids_Ab_dicts, env.encounter_speed)

    obs = env.reset()
    obs_dim = obs.shape[0]

    score_history = []

    agent = Agent(
        alpha=0.000025,
        beta=0.00025,
        input_dims=obs_dim,
        tau=0.01,
        env=env,
        n_actions=2,
        chkpt_dir=str(Path(home) / "McHorcrux" / "checkpoints" / model_dir),
    )

    if len(os.listdir(str(Path(home) / "McHorcrux" / "checkpoints" / model_dir))) != 0:
        print("Loading existing models...")
        agent.load_models()

    robustness_history = []

    for episode in range(num_episodes):
        print(f"episode {episode + 1}/{num_episodes}")
        if episode == num_episodes - 1:
            env = make_env(dt=dt, render_on=True, verbose=verbose, robustness_sampling_rate=robustness_sampling_rate)
            env.set_spec(crossing_spec)
            env.ellipsoids_Ab_dict = select_reachable_set(ellipsoids_Ab_dicts, env.encounter_speed)

        if verbose:
            print(f"\n{'='*60}")
            print(f"  EPISODE {episode + 1}/{num_episodes}")
            print(f"{'='*60}")

        obs = env.reset()

        controller = MRACShipController(dt=dt)

        done = False
        score = 0.0
        step = 0
        prev_obs = obs
        robustness = []
        while not done:
            action_rl = agent.choose_action(obs)
            psi_d, u_d = decode_action_to_refs(action_rl, obs)
            tau = controller.compute_action_minimal(env.get_state(), psi_d, u_d)

            obs, reward, done, info = env.step(tau)
            if step % robustness_sampling_rate == 0:
                robustness.append(info.get("current_robustness", None))    

            if done:
                reason = info.get("reason", "")
                if reason == "encounter_collision":
                    reward -= 50.0
                elif reason == "goal_reached":
                    reward += 50.0
                elif reason == "encounter_time_limit":
                    reward -= 10.0

            new_obs = obs
            agent.remember(prev_obs, action_rl, reward, new_obs, int(done))
            if step % 10 == 0:
                agent.learn()
            prev_obs = obs
            score += reward
            step += 1

        robustness_history.append(robustness)
        score_history.append(score)
        running_avg = float(np.mean(score_history[max(0, len(score_history) - 10):]))
        print(f"Episode {episode + 1:4d}/{num_episodes}  score={score:+8.2f}  avg10={running_avg:+8.2f}")
        if verbose:
            stats = env.get_episode_stats()
            reason = info.get("reason", "unknown")
            min_d = f"{stats['min_enc_dist']:.2f} m" if stats["min_enc_dist"] is not None else "N/A"
            mean_d = f"{stats['mean_enc_dist']:.2f} m" if stats["mean_enc_dist"] is not None else "N/A"
            print(f"  End: {reason}  |  Steps: {stats['steps']}  |  SimTime: {stats['sim_time']:.1f}s")
            print(f"  Enc.dist  min={min_d}  mean={mean_d}  |  Collision: {'YES ***' if stats['collision'] else 'no'}")
        agent.save_models()

    env.plot_trajectory()
    plot_learning_curve(score_history, "./numpy_core/rl_training/results/learning_progress.png")
    plot_robustness_curve(robustness_history)
    print(robustness_history)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train COLREGs DDPG agent")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.08)
    parser.add_argument("--model-dir", default="ddpg")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable step-level and episode-level verbose logging")
    args = parser.parse_args()
    train(num_episodes=args.episodes, dt=args.dt, model_dir=args.model_dir, verbose=args.verbose)