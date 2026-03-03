from numpy_core.gym.mc_gym_csad_numpy import McGym
from pacSTL.colregs_spec import ColregsSpec
import numpy as np
import pygame
import matplotlib.pyplot as plt


class ColregsGym(McGym):
    def __init__(self, *args, verbose=False, robustness_sampling_rate=10, **kwargs):
        super().__init__(*args, **kwargs)

        # Encounter scenario
        self.encounter_mode = False
        self._encounter_init = None
        self.encounter_vessel_eta = None     # [n, e, psi]
        self.encounter_speed = 0.0
        self.encounter_radius = 1.0
        self.encounter_max_time = None
        self.encounter_vessel_traj = []

        # STL spec (set via set_spec)
        self.spec = None
        self.ellipsoids_Ab_dict = None
        self.robustness_sampling_rate = robustness_sampling_rate

        # Action bounds (used by DDPG agent)
        self.v_min, self.v_max = -1.0, 1.0
        self.r_min, self.r_max = -1.0, 1.0

        # Dense reward tracking
        self.prev_dist_to_goal = None

        # Verbose logging
        self.verbose = verbose
        self._log_interval = 50
        self._step_count = 0
        self._encounter_dist_history = []
        self._episode_collision = False

    def set_spec(self, spec):
        self.spec = spec

    def reset(self):
        super().reset()
        if self.encounter_mode and self._encounter_init is not None:
            self.encounter_vessel_eta = self._encounter_init.copy()
            self.encounter_vessel_traj = []
        state = self.get_state()
        n, e = state["eta"][:2]
        gn, ge = self.goal[:2]
        self.prev_dist_to_goal = np.hypot(gn - n, ge - e)

        self._step_count = 0
        self._encounter_dist_history = []
        self._episode_collision = False

        return self.get_obs()

    def get_obs(self) -> np.ndarray:
        state = self.get_state()
        eta, nu = state["eta"], state["nu"]
        g_n, g_e = self.goal[:2]
        tgt = self.encounter_vessel_eta[:2] if self.encounter_vessel_eta is not None else np.zeros(2)
        return np.concatenate([eta[:2], [eta[-1]], nu, [g_n, g_e], tgt])
        # Shape: [n, e, psi, u, v, r, g_n, g_e, tgt_n, tgt_e] = 10D

    def compute_reward(self, action, prev_action):
            reward = 0.0
            state = self.get_state()
            eta, nu = state["eta"], state["nu"]
            n, e = eta[0], eta[1]

            # 1. Progress reward
            gn, ge = self.goal[:2]
            curr_dist = np.hypot(gn - n, ge - e)
            reward += (self.prev_dist_to_goal - curr_dist) * 20.0
            self.prev_dist_to_goal = curr_dist

            # 2. STL robustness penalty (Cleanly decoupled!)
            #robustness = self.evaluate_colregs_robustness()
            # reward -= max(robustness, 0.0) * 0.5

            # 3. Living penalty
            reward -= 0.01
            return reward

# -------------------------
# Monitoring 
#-------------------------
    def evaluate_colregs_robustness(self):
        """Calculates and updates the current STL robustness."""
        if (self.spec is None or self.ellipsoids_Ab_dict is None or self.encounter_vessel_eta is None):
            self.current_robustness = 0.0
            return self.current_robustness

        state = self.get_state()
        eta, nu = state["eta"], state["nu"]
        n, e = eta[0], eta[1]
        psi = eta[-1]

        tgt_n, tgt_e, tgt_psi = self.encounter_vessel_eta
        tgt_vn = self.encounter_speed * np.cos(tgt_psi)
        tgt_ve = self.encounter_speed * np.sin(tgt_psi)
        u, v_sway = nu[0], nu[1]
        ego_vn = u * np.cos(psi) - v_sway * np.sin(psi)
        ego_ve = u * np.sin(psi) + v_sway * np.cos(psi)
        
        state_6d = ColregsSpec.transform_state(
            n, e, psi, ego_vn, ego_ve,
            tgt_n, tgt_e, tgt_psi
        )
        trace = self.spec.build_signals(state_6d, self.ellipsoids_Ab_dict)
        rho, _, _ = self.spec.evaluate(trace)
        rho_val = rho[0][1] if isinstance(rho[0], (tuple, list)) else rho[0]
        
        # Update the class attribute
        self.current_robustness = rho_val
        return self.current_robustness

    # -------------------------
    # Scenario configuration
    # -------------------------
    def set_encounter_task(
        self,
        start_position,
        wave_conditions,
        separation=8.0,
        target_speed=0.3,
        goal_ahead_distance=30.0,
        collision_radius=1.0,
        goal_size=1.0,
        goal_heading_deg=None,
        position_tolerance=1.0,
        heading_tolerance_deg=10.0,
        simtime=300.0,
    ):
        """
        Configure a COLREGs encounter scenario.

        Own ship starts at start_position = (north, east, heading_deg) with a goal
        'goal_ahead_distance' metres ahead on its initial course.

        Encounter vessel is placed 'separation' metres ahead on the same line with
        reciprocal heading, moving at constant 'target_speed'.
        """
        own_n, own_e, own_psi_deg = start_position
        own_psi_rad = np.deg2rad(own_psi_deg)

        t_n = own_n + separation * np.cos(own_psi_rad)
        t_e = own_e + separation * np.sin(own_psi_rad)
        t_psi_deg = (own_psi_deg + 180.0) % 360.0

        self.encounter_mode = True
        self.encounter_speed = float(target_speed)
        self.encounter_radius = float(collision_radius)
        self.encounter_max_time = float(simtime)
        self._encounter_init = np.array([t_n, t_e, np.deg2rad(t_psi_deg)])
        self.encounter_vessel_eta = None  # will be set in reset()

        goal_n = own_n + goal_ahead_distance * np.cos(own_psi_rad)
        goal_e = own_e + goal_ahead_distance * np.sin(own_psi_rad)
        goal = (goal_n, goal_e, goal_size)

        if goal_heading_deg is None:
            goal_heading_deg = own_psi_deg

        self.set_task(
            start_position=start_position,
            goal=goal,
            wave_conditions=wave_conditions,
            obstacles=None,
            goal_func=None,
            obstacle_func=None,
            position_tolerance=position_tolerance,
            goal_heading_deg=goal_heading_deg,
            heading_tolerance_deg=heading_tolerance_deg,
            four_corner_test=False,
            simtime=simtime,
        )

    def step(self, action):
        if self.encounter_mode and self.encounter_vessel_eta is not None:
            self._update_encounter_vessel()

        obs, reward, done, info = super().step(action)
        if self._step_count % 10 == 0:
            info["current_robustness"] = self.evaluate_colregs_robustness()

        self._step_count += 1
        if self.encounter_vessel_eta is not None:
            tgt_n, tgt_e = self.encounter_vessel_eta[:2]
            self._encounter_dist_history.append(float(np.hypot(tgt_n - obs[0], tgt_e - obs[1])))
        if done and info.get("reason") == "encounter_collision":
            self._episode_collision = True
        if self.verbose and (self._step_count % self._log_interval == 0):
            self._log_step(obs, reward)

        return obs, reward, done, info

    def _log_step(self, obs, reward):
        n, e, psi, u = obs[0], obs[1], obs[2], obs[3]
        g_n, g_e = obs[6], obs[7]
        dist_goal = float(np.hypot(g_n - n, g_e - e))
        psi_deg = float(np.rad2deg(psi))
        if self.encounter_vessel_eta is not None:
            tgt_n, tgt_e = self.encounter_vessel_eta[:2]
            enc_str = f"{np.hypot(tgt_n - n, tgt_e - e):.2f} m"
        else:
            enc_str = "N/A"
        print(
            f"  [step {self._step_count:5d} | t={self.curr_sim_time:7.2f}s]"
            f"  pos=({n:.2f}N, {e:.2f}E)"
            f"  psi={psi_deg:.1f}deg"
            f"  u={u:.3f} m/s"
            f"  d_goal={dist_goal:.2f} m"
            f"  d_enc={enc_str}"
            f"  r={reward:+.4f}"
        )

    def get_episode_stats(self):
        if self._encounter_dist_history:
            min_d = float(np.min(self._encounter_dist_history))
            mean_d = float(np.mean(self._encounter_dist_history))
        else:
            min_d = mean_d = None
        return {
            "steps": self._step_count,
            "sim_time": float(self.curr_sim_time),
            "collision": self._episode_collision,
            "min_enc_dist": min_d,
            "mean_enc_dist": mean_d,
        }

    # -------------------------
    # State / collision / drawing
    # -------------------------
    def get_state(self):
        state = super().get_state()
        if self.encounter_mode:
            state["target_eta"] = self.encounter_vessel_eta
            state["encounter_mode"] = True
        return state

    def _update_encounter_vessel(self):
        """Constant-speed kinematics for the encounter vessel."""
        n, e, psi = self.encounter_vessel_eta
        n = n + self.encounter_speed * np.cos(psi) * self.dt
        e = e + self.encounter_speed * np.sin(psi) * self.dt
        self.encounter_vessel_eta = np.array([n, e, psi])

        if self.final_plot:
            self.encounter_vessel_traj.append(self.encounter_vessel_eta.copy())

    def _check_termination(self, boat_pos):
        done, info = super()._check_termination(boat_pos)
        if done:
            return done, info

        if not self.encounter_mode:
            return False, {}

        if self.encounter_max_time is not None and self.curr_sim_time > self.encounter_max_time:
            print("Encounter scenario time limit reached.")
            return True, {"reason": "encounter_time_limit"}

        if self.encounter_vessel_eta is not None:
            boat_yaw = self.vessel.get_eta()[5]
            hull_local = self._get_boat_hull_local_pts()
            c, s = np.cos(boat_yaw), np.sin(boat_yaw)
            rot = np.array([[c, s], [-s, c]])

            hull_global = []
            for lx, ly in hull_local:
                dx, dy = rot @ np.array([lx, ly])
                north_pt = boat_pos[0] + dy
                east_pt  = boat_pos[1] + dx
                hull_global.append((north_pt, east_pt))
            hull_global.append(hull_global[0])

            center = np.array(self.encounter_vessel_eta[:2])
            if self._circle_polygon_collision(hull_global, center, self.encounter_radius):
                print("Collision with encounter vessel!")
                return True, {"reason": "encounter_collision"}

        return False, {}

    def _circle_polygon_collision(self, hull_global, center, radius):
        for v_n, v_e in hull_global:
            if np.hypot(v_n - center[0], v_e - center[1]) < radius:
                return True

        for i in range(len(hull_global) - 1):
            p1 = np.array(hull_global[i])
            p2 = np.array(hull_global[i + 1])
            seg = p2 - p1
            seg_len2 = np.dot(seg, seg)
            if seg_len2 < 1e-12:
                continue
            t = np.dot(center - p1, seg) / seg_len2
            t = np.clip(t, 0.0, 1.0)
            closest = p1 + t * seg
            if np.linalg.norm(center - closest) < radius:
                return True

        return False

    def _draw_obstacles(self):
        super()._draw_obstacles()

        if (not self.render_on or self.screen is None or
                not self.encounter_mode or self.encounter_vessel_eta is None):
            return

        n, e, _ = self.encounter_vessel_eta
        px = e * self.x_scale
        py = (self.grid_height - n) * self.y_scale
        radius_px = self.encounter_radius * self.x_scale

        pygame.draw.circle(
            self.screen,
            (255, 100, 100),
            (int(px), int(py)),
            int(radius_px),
        )

    def plot_trajectory(self):
        if not self.final_plot:
            return

        if self.goal_func is not None or self.goal is not None:
            traj = np.array(self.trajectory)
            plt.figure(figsize=(8, 4))
            plt.plot(traj[:, 1], traj[:, 0], 'g-', label="Boat Trajectory")

            if self.encounter_mode and len(self.encounter_vessel_traj) > 0:
                ttraj = np.array(self.encounter_vessel_traj)
                plt.plot(ttraj[:, 1], ttraj[:, 0], 'r--', label="Encounter Vessel")

            g_n, g_e, g_s = self.goal
            plt.scatter(g_e, g_n, c='yellow',
                        s=(g_s * self.x_scale)**2,
                        edgecolor='black', label="Goal")

            for obs_n, obs_e, obs_size in self.obstacles:
                plt.scatter(obs_e, obs_n, c='red',
                            s=(obs_size * self.x_scale)**2,
                            edgecolor='black', label="Obstacle")

            plt.xlim([0, self.grid_width])
            plt.ylim([0, self.grid_height])
            plt.xlabel("East [m]")
            plt.ylabel("North [m]")
            plt.title("Boat Trajectory ({}×{} Domain)".format(self.grid_width, self.grid_height))
            plt.legend(loc='upper right', fontsize='small', scatterpoints=1, markerscale=0.1)
            plt.grid(True)
            plt.show()
