class AtomicPredicate:
    def __init__(self, name, max_val_state):
        self.name = name
        self.max_val_state = max_val_state

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
        
        return

