from collections import deque

import numpy as np
from mcsimpy.utils import six2threeDOF, pipi


class BacksteppingLOSController:
    """
    Backstepping LOS path-following controller from:

        Fossen, Breivik and Skjetne (2003):
        "Line-of-Sight Path Following of Underactuated Marine Craft"

    Tracks:
        u   -> u_d
        psi -> psi_d

    Handles uncontrolled sway through the dynamic controller state alpha_2
    (Theorem 1 in the paper).

    Reference signal expectations
    -----------------------------
    The caller is expected to provide:
        psi_d, u_d                : smooth references (e.g. integrated from
                                    a rate/acceleration action)
        psi_d_dot, u_d_dot        : exact first derivatives of the references
        psi_d_ddot                : exact second derivative of psi_d

    For an action space of (surge acceleration, yaw rate), the natural
    choice is:
        u_d_dot   = commanded surge acceleration  (RL action)
        psi_d_dot = commanded yaw rate            (RL action)
        psi_d_ddot = 0    (the commanded yaw rate is piecewise constant)

    If any derivative is passed as None, it is recovered by backward
    finite differencing of psi_d / u_d. This fallback is kept only for
    convenience; it produces spikes whenever the reference changes
    abruptly and should be avoided in practice.
    """

    def __init__(
        self,
        dt: float,
        M: np.ndarray,
        D: np.ndarray,
        c: float = 0.75,
        k1: float = 25.0,
        k2: float = 10.0,
        k3: float = 2.5,
        debug_logging: bool = True,
        debug_log_length: int = 10000,
    ):
        self._dt = float(dt)

        # Convert 6-DOF matrices to 3-DOF: surge, sway, yaw
        self.M = six2threeDOF(M)
        self.D = six2threeDOF(D)

        self.m11 = self.M[0, 0]
        self.m22 = self.M[1, 1]
        self.m23 = self.M[1, 2]
        self.m32 = self.M[2, 1]
        self.m33 = self.M[2, 2]

        self.d11 = self.D[0, 0]
        self.d22 = self.D[1, 1]
        self.d23 = self.D[1, 2]
        self.d32 = self.D[2, 1]
        self.d33 = self.D[2, 2]

        # Paper notation:
        # c  = heading backstepping gain
        # k1 = surge velocity error gain
        # k2 = sway dynamic-state gain
        # k3 = yaw-rate error gain
        self.c = c
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

        # Dynamic controller state for the uncontrolled sway mode (alpha_2).
        # Integrated forward each step.
        self.alpha_2 = 0.0

        # Reference history for the (fallback) finite-difference path
        self._psi_d_prev = None
        self._psi_d_dot_prev = None
        self._u_d_prev = None

        self._debug_logging = debug_logging
        self._debug_log = deque(maxlen=debug_log_length)
        self._last_debug = None

    # ------------------------------------------------------------------
    # Debug log helpers
    # ------------------------------------------------------------------
    def _copy_debug(self, debug: dict):
        return {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in debug.items()
        }

    def _record_debug(self, debug: dict):
        debug = self._copy_debug(debug)
        self._last_debug = debug
        if self._debug_logging:
            self._debug_log.append(debug)

    # ------------------------------------------------------------------
    # Reference derivative fallback (finite differencing)
    # ------------------------------------------------------------------
    def _reference_derivatives(self, psi_d: float, u_d: float):
        """
        Backward-difference fallback used only when the caller does not
        provide derivatives. Produces spikes when psi_d / u_d are not
        smooth, so prefer to pass the analytic derivatives.
        """
        if self._psi_d_prev is None:
            psi_d_dot = 0.0
            psi_d_ddot = 0.0
            u_d_dot = 0.0
        else:
            psi_d_dot = pipi(psi_d - self._psi_d_prev) / self._dt
            u_d_dot = (u_d - self._u_d_prev) / self._dt

            if self._psi_d_dot_prev is None:
                psi_d_ddot = 0.0
            else:
                psi_d_ddot = (psi_d_dot - self._psi_d_dot_prev) / self._dt

        self._psi_d_prev = psi_d
        self._psi_d_dot_prev = psi_d_dot
        self._u_d_prev = u_d

        return psi_d_dot, psi_d_ddot, u_d_dot

    # ------------------------------------------------------------------
    # Main control law
    # ------------------------------------------------------------------
    def compute_action(
        self,
        state: dict,
        psi_d: float,
        u_d: float,
        psi_d_dot: float | None = None,
        psi_d_ddot: float | None = None,
        u_d_dot: float | None = None,
        calculate_debug: bool = False,
    ):
        """
        Compute 3-DOF generalized force:

            action = [surge force, sway force, yaw moment]

        The vessel is underactuated, so the sway entry is always zero.
        """

        psi = float(state["eta"][2])
        nu = np.asarray(state["nu"][:3], dtype=float)
        u, v, r = nu

        # --------------------------------------------------------------
        # Reference derivatives
        #
        # Use caller-provided values when available (preferred).
        # Fall back to finite differencing only for any that are None.
        # --------------------------------------------------------------
        if psi_d_dot is None or psi_d_ddot is None or u_d_dot is None:
            fd_psi_d_dot, fd_psi_d_ddot, fd_u_d_dot = self._reference_derivatives(
                psi_d=psi_d,
                u_d=u_d,
            )
            if psi_d_dot is None:
                psi_d_dot = fd_psi_d_dot
            if psi_d_ddot is None:
                psi_d_ddot = fd_psi_d_ddot
            if u_d_dot is None:
                u_d_dot = fd_u_d_dot
        else:
            # Keep the finite-difference state in sync so that switching
            # back to the fallback later does not produce a one-step jump.
            self._psi_d_prev = psi_d
            self._psi_d_dot_prev = psi_d_dot
            self._u_d_prev = u_d

        # --------------------------------------------------------------
        # Heading error and stabilizing function alpha_3 (Eq. 26-27)
        # --------------------------------------------------------------
        # z1 = psi - psi_d   (paper notation)
        z1 = pipi(psi - psi_d)

        # alpha_3 = -c*z1 + r_d
        alpha_3 = -self.c * z1 + psi_d_dot

        # alpha_3_dot = -c*(r - r_d) + r_d_dot
        alpha_3_dot = -self.c * (r - psi_d_dot) + psi_d_ddot

        # --------------------------------------------------------------
        # Velocity errors  z2 = [u-u_d, v-alpha_2, r-alpha_3]
        # --------------------------------------------------------------
        z_u = u - u_d
        z_v = v - self.alpha_2
        z_r = r - alpha_3

        # --------------------------------------------------------------
        # Dynamic sway stabilizing function alpha_2 (Theorem 1)
        #
        # m22 * alpha_2_dot =
        #     -d22 * alpha_2
        #     + (k2 - d22) * (v - alpha_2)
        #     - m23 * alpha_3_dot
        #     - d23 * r
        #
        # alpha_2_dot is evaluated at the *pre-update* state and used
        # immediately in the yaw law, then alpha_2 is integrated forward
        # for the next call.
        # --------------------------------------------------------------
        alpha_2_dot = (
            -self.d22 * self.alpha_2
            + (self.k2 - self.d22) * z_v
            - self.m23 * alpha_3_dot
            - self.d23 * r
        ) / self.m22

        # --------------------------------------------------------------
        # Control laws (Theorem 1)
        #
        # Surge:
        #   tau_1 = m11*u_d_dot + d11*u - k1*(u - u_d)
        #
        # Yaw:
        #   tau_3 = m32*alpha_2_dot + m33*alpha_3_dot
        #         + d32*v + d33*r
        #         - k3*(r - alpha_3) - z1
        # --------------------------------------------------------------
        F_x = self.m11 * u_d_dot + self.d11 * u - self.k1 * z_u

        M_z = (
            self.m32 * alpha_2_dot
            + self.m33 * alpha_3_dot
            + self.d32 * v
            + self.d33 * r
            - self.k3 * z_r
            - z1
        )

        # Integrate alpha_2 forward for the next call
        self.alpha_2 += self._dt * alpha_2_dot

        action = np.array([F_x, 0.0, M_z], dtype=float)

        debug = {
            "psi": psi,
            "psi_d": psi_d,
            "psi_d_dot": psi_d_dot,
            "psi_d_ddot": psi_d_ddot,
            "u_d": u_d,
            "u_d_dot": u_d_dot,
            "u": u,
            "v": v,
            "r": r,
            "z1_heading": z1,
            "z_u": z_u,
            "z_v": z_v,
            "z_r": z_r,
            "alpha_2": self.alpha_2,
            "alpha_2_dot": alpha_2_dot,
            "alpha_3": alpha_3,
            "alpha_3_dot": alpha_3_dot,
            "F_x": F_x,
            "M_z": M_z,
            "action": action,
        }
        self._record_debug(debug)

        if calculate_debug:
            return action, self._copy_debug(debug)
        return action

    # ------------------------------------------------------------------
    # Lifecycle / configuration
    # ------------------------------------------------------------------
    def reset(self):
        self.alpha_2 = 0.0
        self._psi_d_prev = None
        self._psi_d_dot_prev = None
        self._u_d_prev = None
        self.clear_debug_log()

    def get_last_debug(self):
        if self._last_debug is None:
            return None
        return self._copy_debug(self._last_debug)

    def get_debug_log(self):
        return [self._copy_debug(sample) for sample in self._debug_log]

    def clear_debug_log(self):
        self._debug_log.clear()
        self._last_debug = None

    def set_debug_logging(self, enabled: bool):
        self._debug_logging = enabled

    def set_gains(
        self,
        c: float | None = None,
        k1: float | None = None,
        k2: float | None = None,
        k3: float | None = None,
    ):
        if c is not None:
            self.c = c
        if k1 is not None:
            self.k1 = k1
        if k2 is not None:
            self.k2 = k2
        if k3 is not None:
            self.k3 = k3
