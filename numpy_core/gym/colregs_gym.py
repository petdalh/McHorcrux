from numpy_core.gym.mc_gym_csad_numpy import McGym
from numpy_core.colregs_stl.vessel_state import VesselState, JointState
import numpy as np
import pygame
import matplotlib.pyplot as plt

class ColregsGym(McGym):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # COLREGs scenario parameters
        self.head_on_mode = False
        self.head_on_init = None       
        self.head_on_speed = 0.0       
        self.head_on_radius = 0.0      
        self.head_on_max_time = None   
        self.target_vessel_eta = None  
        self.target_vessel_traj = [] 

        # State histories for STL analysis
        self.ego_state_history: list[VesselState] = []
        self.adversary_state_history: list[VesselState] = []
        self.joint_state_history = JointState([], [], np.array([0.0, 0.0], dtype=float))

        # For dense reward tracking
        self.prev_dist_to_goal = None

    def reset(self):
        state = super().reset()

        # Head-on init first (so get_state includes target)
        if self.head_on_mode and self.head_on_init is not None:
            self.target_vessel_eta = self.head_on_init.copy()
            self.target_vessel_traj = []

        state = self.get_state()

        # progress tracking
        n, e = state["eta"][:2]
        gn, ge = self.goal[:2]
        self.prev_dist_to_goal = np.hypot(gn - n, ge - e)

        # histories
        self.ego_state_history.clear()
        self.adversary_state_history.clear()

        g_n, g_e = state["goal"][:2] if "goal" in state else self.goal[:2]
        self.joint_state_history = JointState([], [], np.array([g_n, g_e], dtype=float))

        self._log_joint_state(state)
        return state


    def _log_joint_state(self, state: dict) -> None:
        eta = state["eta"]
        nu = state["nu"]

        ego_vs = VesselState(
            px=float(eta[0]),
            py=float(eta[1]),
            theta=float(eta[-1]),
            v=float(nu[0]),
            omega=float(nu[2]),
        )
        self.ego_state_history.append(ego_vs)
        self.joint_state_history.ego.append(ego_vs)

        tgt_eta = state.get("target_eta", None)
        if tgt_eta is None:
            adv_vs = VesselState(0.0, 0.0, 0.0, 0.0, 0.0)
        else:
            adv_vs = VesselState(
                px=float(tgt_eta[0]),
                py=float(tgt_eta[1]),
                theta=float(tgt_eta[-1]),
                v=float(state.get("target_speed", self.head_on_speed if self.head_on_mode else 0.0)),
                omega=float(state.get("target_omega", 0.0)),
            )

        self.adversary_state_history.append(adv_vs)
        self.joint_state_history.adversary.append(adv_vs)


    def compute_reward(self, action, prev_action):
        """
        DENSE REWARD FUNCTION
        Overwrites the McGym placeholder to provide feedback every time-step.
        """
        reward = 0.0
        state = self.get_state()
        eta = state["eta"]
        nu = state["nu"] # [u, v, r]
        n, e = eta[0], eta[1]
        
        # 1. Progress Reward (Directional Feedback)
        # Reward the agent for every meter it gets closer to the goal
        gn, ge = self.goal[:2]
        curr_dist = np.hypot(gn - n, ge - e)
        dist_change = self.prev_dist_to_goal - curr_dist
        
        # We multiply by a factor (e.g., 20) so the agent "feels" the movement
        reward += dist_change * 20.0 
        self.prev_dist_to_goal = curr_dist

        # 2. COLREGs Rule 14 "Starboard Bias"
        # In a head-on situation, ships should turn to starboard (right).
        # We penalize turning to port (left) when the target is within a 'reaction' distance.
        if self.head_on_mode and self.target_vessel_eta is not None:
            tn, te = self.target_vessel_eta[:2]
            dist_to_target = np.hypot(tn - n, te - e)
            
            # If target is within 15m, penalize negative yaw rate (turning port)
            # In most marine models, positive 'r' (nu[2]) is a starboard turn.
            yaw_rate = nu[2]
            if dist_to_target < 15.0 and yaw_rate < -0.05:
                reward -= 0.5  # Small penalty for "illegal" turn direction

        # 3. Comfort/Efficiency Penalties
        # Small penalty for high control effort (prevents jerky movements)
        # reward -= 0.005 * np.linalg.norm(action - prev_action)
        
        # Small living penalty to encourage reaching the goal faster
        reward -= 0.01

        return reward

    # -------------------------
    # Scenario configuration
    # -------------------------
    def set_head_on_task(
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
        Configure a simple COLREGs Rule 14 head-on encounter.

        Own ship:
            - starts at start_position = (north, east, heading_deg)
            - has a goal 'goal_ahead_distance' meters ahead on its initial course.

        Incoming vessel:
            - placed 'separation' meters ahead on the same line,
            - with reciprocal course (heading + 180 deg),
            - moves at constant 'target_speed' and is treated as a circle of
              radius 'collision_radius'.
        """
        own_n, own_e, own_psi_deg = start_position
        own_psi_rad = np.deg2rad(own_psi_deg)

        # Place the incoming vessel on the same track, in front of us
        t_n = own_n + separation * np.cos(own_psi_rad)
        t_e = own_e + separation * np.sin(own_psi_rad)
        t_psi_deg = (own_psi_deg + 180.0) % 360.0

        # Store head-on configuration
        self.head_on_mode = True
        self.head_on_speed = float(target_speed)
        self.head_on_radius = float(collision_radius)
        self.head_on_max_time = float(simtime)
        self.head_on_init = np.array([t_n, t_e, np.deg2rad(t_psi_deg)])
        self.target_vessel_eta = None  # will be set in reset()

        # Define a goal further ahead on own initial track
        goal_n = own_n + goal_ahead_distance * np.cos(own_psi_rad)
        goal_e = own_e + goal_ahead_distance * np.sin(own_psi_rad)
        goal = (goal_n, goal_e, goal_size)

        if goal_heading_deg is None:
            goal_heading_deg = own_psi_deg

        # Use the base set_task to configure own ship, goal, waves, etc.
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
        if self.head_on_mode and self.target_vessel_eta is not None:
            self._update_head_on_vessel()

        new_state, done, info, reward = super().step(action)

        # ensure target info is present (depends on your get_state implementation)
        self._log_joint_state(new_state)

        return new_state, done, info, reward

    # -------------------------
    # State / collision / drawing
    # -------------------------
    def get_state(self):
        state = super().get_state()
        if self.head_on_mode:
            state["target_eta"] = self.target_vessel_eta
            state["head_on_mode"] = True
        return state

    def _update_head_on_vessel(self):
        """
        Simple kinematics for the incoming vessel:
        constant speed and heading in the N–E plane.
        """
        n, e, psi = self.target_vessel_eta
        n = n + self.head_on_speed * np.cos(psi) * self.dt
        e = e + self.head_on_speed * np.sin(psi) * self.dt
        self.target_vessel_eta = np.array([n, e, psi])

        if self.final_plot:
            self.target_vessel_traj.append(self.target_vessel_eta.copy())

    def _check_termination(self, boat_pos):
        """
        Extend the base termination logic with head-on collision and time limit.
        """
        # Let the base class handle goal and static obstacles
        done, info = super()._check_termination(boat_pos)
        if done:
            return done, info

        # Head-on logic
        if not self.head_on_mode:
            return False, {}

        # Time horizon
        if self.head_on_max_time is not None and self.curr_sim_time > self.head_on_max_time:
            print("Head-on scenario time limit reached.")
            return True, {"reason": "head_on_time_limit"}

        # Collision with incoming vessel (circle vs. own hull polygon)
        if self.target_vessel_eta is not None:
            boat_yaw = self.vessel.get_eta()[5]  # radians
            hull_local = self._get_boat_hull_local_pts()
            c, s = np.cos(boat_yaw), np.sin(boat_yaw)
            rot = np.array([[c, s], [-s, c]])

            hull_global = []
            for lx, ly in hull_local:
                dx, dy = rot @ np.array([lx, ly])
                north_pt = boat_pos[0] + dy
                east_pt  = boat_pos[1] + dx
                hull_global.append((north_pt, east_pt))
            # close polygon
            hull_global.append(hull_global[0])

            center = np.array(self.target_vessel_eta[:2])
            if self._circle_polygon_collision(hull_global, center, self.head_on_radius):
                print("Collision with incoming vessel (head-on)!")
                return True, {"reason": "head_on_collision"}

        return False, {}

    def _circle_polygon_collision(self, hull_global, center, radius):
        """
        Check collision between a convex polygon (boat hull) and a circle.

        hull_global: list[(n, e), ...] closed polygon.
        center: np.array([n, e])
        radius: float
        """
        # 1) Vertex check
        for v_n, v_e in hull_global:
            if np.hypot(v_n - center[0], v_e - center[1]) < radius:
                return True

        # 2) Edge–segment check
        for i in range(len(hull_global) - 1):
            p1 = np.array(hull_global[i])
            p2 = np.array(hull_global[i + 1])
            seg = p2 - p1
            seg_len2 = np.dot(seg, seg)
            if seg_len2 < 1e-12:  # degenerate edge
                continue

            t = np.dot(center - p1, seg) / seg_len2
            t = np.clip(t, 0.0, 1.0)
            closest = p1 + t * seg
            if np.linalg.norm(center - closest) < radius:
                return True

        return False

    def _draw_obstacles(self):
        """
        Draw base obstacles, then the head-on target vessel as a red-ish circle.
        """
        super()._draw_obstacles()

        if (not self.render_on or self.screen is None or
                not self.head_on_mode or self.target_vessel_eta is None):
            return

        n, e, _ = self.target_vessel_eta
        px = e * self.x_scale
        py = (self.grid_height - n) * self.y_scale
        radius_px = self.head_on_radius * self.x_scale

        pygame.draw.circle(
            self.screen,
            (255, 100, 100),  # distinguish from generic obstacles
            (int(px), int(py)),
            int(radius_px),
        )

            
    def plot_trajectory(self):
        """
        Same as McGym.plot_trajectory, but also plots the incoming vessel
        trajectory when in head-on mode.
        """
        if not self.final_plot:
            return

        # Main trajectory figure 
        if self.goal_func is not None or self.goal is not None:
            traj = np.array(self.trajectory)
            plt.figure(figsize=(8, 4))
            plt.plot(traj[:, 1], traj[:, 0], 'g-', label="Boat Trajectory")

            # Incoming vessel path (new)
            if self.head_on_mode and len(self.target_vessel_traj) > 0:
                ttraj = np.array(self.target_vessel_traj)
                plt.plot(ttraj[:, 1], ttraj[:, 0], 'r--', label="Incoming Vessel")

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
