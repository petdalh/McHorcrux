from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class VesselState:
    px: float       # North position [m]
    py: float       # East position [m]
    theta: float    # Heading [rad]
    v: float        # Surge speed [m/s]
    omega: float    # Yaw rate [rad/s]


@dataclass
class JointState:
    ego: List[VesselState]
    adversary: List[VesselState]
    goal: np.ndarray  # [goal_n, goal_e]
