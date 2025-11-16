from numpy_core.gym.colregs_gym import ColregsGym
from numpy_core.controllers.adaptive_seakeeping import MRACShipController
from mcsimpy.utils import six2threeDOF
import numpy as np
import time


def main():
    dt = 0.08
    env = ColregsGym(
        dt=dt,
        grid_width=25,   # >= 7.5
        grid_height=30,  # big enough for north up to ~ goal_n
        render_on=True,
        final_plot=True,
    )


    # Own ship start: in the middle, pointing north (heading = 0 deg)
    start_pos = (1.5, 7.5, 0.0)
    wave_conditions = (0.05, 1.5, 0)  # mild waves

    # Configure head-on situation
    env.set_head_on_task(
        start_position=start_pos,
        wave_conditions=wave_conditions,
        separation=18,  # incoming vessel 8 m ahead
        target_speed=0.3,  # incoming speed
        goal_ahead_distance=25.0,  # own goal further ahead on same line
        collision_radius=1.0,
        simtime=150.0,
    )

    controller = MRACShipController(dt=dt)

    simtime = 150.0
    max_steps = int(simtime / dt)
    print("Starting head-on simulation...")
    start_time = time.time()

    for step_count in range(max_steps):
        state = env.get_state()

        goal_n, goal_e, _ = state["goal"]
        action = controller.compute_action(state, (goal_n, goal_e))

        new_state, done, info, reward = env.step(action)

        boat_n, boat_e, yaw = six2threeDOF(new_state["eta"])

        if done:
            print(
                f"Terminated at step {step_count}, reason: {info.get('reason')}"
            )
            break

    total_time = time.time() - start_time
    print(f"Wall-clock time: {total_time:.2f} s")
    print(f"Simulation speed: {(simtime / total_time):.2f}x real-time")
    env.plot_trajectory()



if __name__ == "__main__":
    main()
