from numpy_core.gym.mc_gym_csad_numpy import McGym
import numpy as np
import pygame

class ColregsGym(McGym):
    """
    Extension of McGym that adds a simple COLREGs Rule 14 head-on target vessel.

    Own ship dynamics, wave loads, goal, obstacles etc are all handled by McGym.
    This class adds:
      - a kinematic "incoming vessel" on a straight, reciprocal course
      - head-on collision detection (circle vs. own hull)
      - optional time limit for the head-on scenario
      - target_eta in the state dict
      - drawing of the incoming vessel in pygame
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # COLREGs scenario parameters
        self.head_on_mode = False
        self.head_on_init = None       # np.array([n, e, psi])
        self.head_on_speed = 0.0       # [m/s]
        self.head_on_radius = 0.0      # [m]
        self.head_on_max_time = None   # [s]
        self.target_vessel_eta = None  # np.array([n, e, psi])
        self.target_vessel_traj = []   # for plotting

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

    # -------------------------
    # Lifecycle overrides
    # -------------------------
    def reset(self):
        """Reset environment and incoming vessel."""
        super().reset()

        if self.head_on_mode and self.head_on_init is not None:
            self.target_vessel_eta = self.head_on_init.copy()
            self.target_vessel_traj = []

    def step(self, action):
        """Advance both the own ship and the incoming vessel."""
        if self.head_on_mode and self.target_vessel_eta is not None:
            self._update_head_on_vessel()

        return super().step(action)

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
        # 1) Let the base class handle goal and static obstacles
        done, info = super()._check_termination(boat_pos)
        if done:
            return done, info

        # 2) Head-on logic
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

        # --- Main trajectory figure ---
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

        # --- Four-corner plots unchanged ---
        if self.four_corner_test:
            traj = np.array(self.trajectory)
            true_vel = np.array(self.true_vel)

            plt.figure(figsize=(8, 4))
            plt.plot(traj[:, 1], traj[:, 0], 'b-', label="Boat Trajectory")
            plt.plot(self.store_xd[:, 1], self.store_xd[:, 0], 'g-', label="Desired Trajectory")
            plt.xlim([0, self.grid_width])
            plt.ylim([0, self.grid_height])
            plt.xlabel("East [m]")
            plt.ylabel("North [m]")
            plt.title("Desired Trajectory and real trajectory ({}×{} Domain)".format(self.grid_width, self.grid_height))
            plt.legend(loc='upper right', fontsize='small', scatterpoints=1, markerscale=0.1)
            plt.grid(True)
            plt.show()

            plt.figure(figsize=(8, 4))
            plt.plot(self.t, self.store_xd[:, 0], 'r-', label="North")
            plt.plot(self.t, self.store_xd[:, 1], 'g-', label="East")
            plt.plot(self.t, traj[:, 0], 'b-', label="North (actual)")
            plt.plot(self.t, traj[:, 1], 'c-', label="East (actual)")
            plt.xlabel("Time [s]")
            plt.ylabel("Position [m]")
            plt.title("Desired trajectory over time")
            plt.legend(loc='upper right', fontsize='small', scatterpoints=1, markerscale=0.1)
            plt.show()

            plt.figure(figsize=(8, 4))
            plt.plot(self.t, self.store_xd[:, 3], 'r-', label="yaw")
            plt.plot(self.t, traj[:, 2], 'b-', label="yaw (actual)")
            plt.xlabel("Time [s]")
            plt.ylabel("Degrees [rad]")
            plt.title("Desired yaw over time")
            plt.legend(loc='upper right', fontsize='small', scatterpoints=1, markerscale=0.1)
            plt.show()

            plt.figure(figsize=(8, 4))
            plt.plot(self.t, self.store_xd[:, 6], 'r-', label="North")
            plt.plot(self.t, true_vel[:, 0], 'b-', label="North (actual)")
            plt.plot(self.t, self.store_xd[:, 7], 'g-', label="East")
            plt.plot(self.t, true_vel[:, 1], 'c-', label="East (actual)")
            plt.xlabel("Time [s]")
            plt.ylabel("Velocity [m/s]")
            plt.title("Desired velocity over time")
            plt.legend(loc='upper right', fontsize='small', scatterpoints=1, markerscale=0.1)
            plt.show()

            plt.figure(figsize=(8, 4))
            plt.plot(self.t, self.store_xd[:, 8], 'r-', label="yaw")
            plt.plot(self.t, true_vel[:, 2], 'b-', label="yaw (actual)")
            plt.xlabel("Time [s]")
            plt.ylabel("Velocity [rad/s]")
            plt.title("Desired yaw velocity over time")
            plt.legend(loc='upper right', fontsize='small', scatterpoints=1, markerscale=0.1)
            plt.show()

