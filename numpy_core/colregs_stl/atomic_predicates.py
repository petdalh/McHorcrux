from numpy_core.colregs_stl.vessel_state import VesselState, JointState
import numpy as np

class AtomicPredicate:
    def __init__(self, name, max_val_state, joint_state: JointState = None):
        self.name = name
        self.max_val_state = max_val_state
        self.joint_state = joint_state

    def evaluate_robust(self, state, k=1):
        raw_robustness = self._compute_raw(state, k)
        return raw_robustness / self.max_val_state

class PositionHalfplane(AtomicPredicate):
    def __init__(self, beta, v_max):
        super().__init__("pos_halfplane", max_val_state=v_max)
        self.beta = beta

    def _compute_raw(self, joint_state, k):
        x_E = joint_state.ego[k]
        x_A = joint_state.adversary[k]
        
        p_E = np.array([x_E.px, x_E.py])
        p_A = np.array([x_A.px, x_A.py])
        
        angle = x_E.theta + self.beta
        normal_vector = np.array([-np.sin(angle), np.cos(angle)])
        
        return np.dot(normal_vector, (p_A - p_E))

class OrientationHalfplane(AtomicPredicate):
    def __init__(self, gamma, omega_max):
        super().__init__("ori_halfplane", max_val_state=omega_max)
        self.gamma = gamma 

    def _compute_raw(self, joint_state, k):
        x_E = joint_state.ego[k]
        x_A = joint_state.adversary[k]
        
        diff = x_A.theta - (x_E.theta + self.gamma)
        return np.arcsin(np.sin(diff))

class ChangeCourse(AtomicPredicate):
    def __init__(self, delta, alpha_max, theta_ref=0.0):
        super().__init__("change_course", max_val_state=alpha_max)
        self.delta = delta 
        self.theta_ref = theta_ref 

    def _compute_raw(self, joint_state, k):
        x_E = joint_state.ego[k]
        return self.theta_ref + self.delta - x_E.theta

class VelocityHalfplane(AtomicPredicate):
    def __init__(self, epsilon_offset, a_max, d_zone):
        super().__init__("vel_halfplane", max_val_state=a_max)
        self.epsilon_offset = epsilon_offset
        self.d_zone = d_zone

    def _compute_raw(self, joint_state, k):
        x_E = joint_state.ego[k]
        x_A = joint_state.adversary[k]
        
        p_diff = np.array([x_A.px - x_E.px, x_A.py - x_E.py])
        dist = np.linalg.norm(p_diff)
        
        ratio = (2 * self.d_zone) / max(dist, 0.001)
        epsilon = np.arcsin(np.clip(ratio, -1.0, 1.0))
        
        v_E_vec = x_E.v * np.array([np.cos(x_E.theta), np.sin(x_E.theta)])
        v_A_vec = x_A.v * np.array([np.cos(x_A.theta), np.sin(x_A.theta)])
        v_rel = v_E_vec - v_A_vec

        rot_angle = self.epsilon_offset + (np.pi / 2)
        rot_mat = np.array([[np.cos(rot_angle), -np.sin(rot_angle)], 
                           [np.sin(rot_angle), np.cos(rot_angle)]])
        
        return (1.0 / dist) * np.dot(rot_mat @ p_diff, v_rel)

class TimeHorizon(AtomicPredicate):
    def __init__(self, t_h, a_max):
        super().__init__("time_horizon", max_val_state=a_max)
        self.t_h = t_h 

    def _compute_raw(self, joint_state, k):
        x_E = joint_state.ego[k]
        x_A = joint_state.adversary[k]
        
        p_diff = np.array([x_A.px - x_E.px, x_A.py - x_E.py])
        v_E_vec = x_E.v * np.array([np.cos(x_E.theta), np.sin(x_E.theta)])
        v_A_vec = x_A.v * np.array([np.cos(x_A.theta), np.sin(x_A.theta)])
        v_rel_norm = np.linalg.norm(v_E_vec - v_A_vec)
        
        return v_rel_norm - (np.linalg.norm(p_diff) / self.t_h)

class DrivesFaster(AtomicPredicate):
    def __init__(self, a_max):
        super().__init__("drives_faster", max_val_state=a_max)

    def _compute_raw(self, joint_state, k):
        x_E = joint_state.ego[k]
        x_A = joint_state.adversary[k]
        return x_E.v - x_A.v