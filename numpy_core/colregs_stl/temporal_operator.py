class TemporalOperator(STLFormula):
    def __init__(self, formula: STLFormula, interval: tuple[int, int]):
        self.formula = formula
        self.a, self.b = interval # Interval [a, b] in time steps

class Always(TemporalOperator):
    def evaluate_robust(self, joint_state, k):
        window_values = [
            self.formula.evaluate_robust(joint_state, i) 
            for i in range(k + self.a, k + self.b + 1)
        ]
        return np.min(window_values)

class Eventually(TemporalOperator):
    def evaluate_robust(self, joint_state, k):
        window_values = [
            self.formula.evaluate_robust(joint_state, i) 
            for i in range(k + self.a, k + self.b + 1)
        ]
        return np.max(window_values)