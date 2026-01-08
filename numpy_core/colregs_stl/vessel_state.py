from dataclasses import dataclass
import numpy as np

@dataclass
class VesselState:
    px: float   # North 
    py: float   # East 
    theta: float # Orientation 
    v: float     # Linear velocity 
    omega: float # Angular velocity 

    def to_vec(self):
        return np.array([self.px, self.py, self.theta, self.v, self.omega])

@dataclass
class JointState:
    # Lists of VesselState representing the sequence of states over time
    ego: list[VesselState] 
    adversary: list[VesselState]
    goal_pos: np.ndarray 