from numpy_core.colregs_stl.vessel_state import VesselState, JointState
import numpy as np
class AtomicPredicate:
    def __init__(self, name, max_val_state, joint_state: JointState = None):
        self.name = name
        self.max_val_state = max_val_state
        self.joint_state = joint_state

    def evaluate_robust(self, state, k=1.0):
        raw_robustness = self._compute_raw(state, k)
        return raw_robustness / self.max_val_state

class PositionHalfplane(AtomicPredicate):
    def __init__(self, beta, v_max):
        super().__init__("pos_halfplane", max_val_scale=v_max)
        self.beta = beta # The angle defining the half-plane

    def _compute_raw(self, joint_state, k):
        x_E = joint_state.ego[k]
        x_A = joint_state.adversary[k]
        
        # North-East coordinates: p = [px, py]
        p_E = np.array([x_E.px, x_E.py])
        p_A = np.array([x_A.px, x_A.py])
        
        # Vector: [-sin(theta + beta), cos(theta + beta)]
        angle = x_E.theta + self.beta
        normal_vector = np.array([-np.sin(angle), np.cos(angle)])
        
        # Signed distance = normal_vector dot (p_A - p_E)
        return np.dot(normal_vector, (p_A - p_E))

class OrientationHalfplane(AtomicPredicate):
    def __init__(self, gamma, omega_max):
        super().__init__("ori_halfplane", max_val_state=omega_max)
        self.gamma = gamma 

    def _compute_raw(self, joint_state, k):
        x_E = joint_state.ego[k]
        x_A = joint_state.adversary[k]
        
        # Minimal signed angular difference 
        # Uses arcsin(sin(...)) to handle the wrap-around logic
        diff = x_A.theta - (x_E.theta + self.gamma)
        return np.arcsin(np.sin(diff))

class ChangeCourse(AtomicPredicate):
    def __init__(self, delta, alpha_max, theta_ref=0.0):
        super().__init__("change_course", max_val_state=alpha_max)
        self.delta = delta # Desired change in orientation
        self.theta_ref = theta_ref # Reference orientation 

    def _compute_raw(self, joint_state, k):
        x_E = joint_state.ego[k]
        # Angle change relative to reference 
        return self.theta_ref + self.delta - x_E.theta

class VelocityHalfplane(AtomicPredicate):
    def __init__(self, d_zone, a_max):
        super().__init__("vel_halfplane", max_val_state=a_max)
        self.d_zone = d_zone # Radius of protected zone

    def _compute_raw(self, joint_state, k):
        x_E = joint_state.ego[k]
        x_A = joint_state.adversary[k]
        
        p_diff = x_A.to_vec()[:2] - x_E.to_vec()[:2]
        dist = np.linalg.norm(p_diff)
        
        # Tangent line angle epsilon 
        epsilon = np.arcsin((2 * self.d_zone) / dist)
        
        # Relative velocity vector 
        v_E_vec = x_E.v * np.array([np.cos(x_E.theta), np.sin(x_E.theta)])
        v_A_vec = x_A.v * np.array([np.cos(x_A.theta), np.sin(x_A.theta)])
        v_rel = v_E_vec - v_A_vec
        
        # Eq (16): Perpendicular distance in velocity space
        # Rotation matrix R(epsilon + pi/2) 
        rot_angle = epsilon + (np.pi / 2)
        rot_mat = np.array([[np.cos(rot_angle), -np.sin(rot_angle)], 
                           [np.sin(rot_angle), np.cos(rot_angle)]])
        
        norm_factor = 1.0 / dist # Scaling inside the raw compute
        return norm_factor * np.dot(rot_mat @ p_diff, v_rel)

class TimeHorizon(AtomicPredicate):
    def __init__(self, t_h, a_max):
        super().__init__("time_horizon", max_val_state=a_max)
        self.t_h = t_h # Look-ahead time horizon 

    def _compute_raw(self, joint_state, k):
        x_E = joint_state.ego[k]
        x_A = joint_state.adversary[k]
        
        p_diff = x_A.to_vec()[:2] - x_E.to_vec()[:2]
        # Same relative velocity calculation as above
        v_E_vec = x_E.v * np.array([np.cos(x_E.theta), np.sin(x_E.theta)])
        v_A_vec = x_A.v * np.array([np.cos(x_A.theta), np.sin(x_A.theta)])
        v_rel_norm = np.linalg.norm(v_E_vec - v_A_vec)
        
        # Velocity required to travel distance in t_h
        return v_rel_norm - (np.linalg.norm(p_diff) / self.t_h)

class DrivesFaster(AtomicPredicate):
    def __init__(self, a_max):
        super().__init__("drives_faster", max_val_state=a_max)

    def _compute_raw(self, joint_state, k):
        x_E = joint_state.ego[k]
        x_A = joint_state.adversary[k]
        # Velocity difference 
        return x_E.v - x_A.v


