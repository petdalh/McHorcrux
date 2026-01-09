import numpy as np
from abc import ABC, abstractmethod

class STLFormula(ABC):
    @abstractmethod
    def evaluate_robust(self, joint_state, k: int) -> float:
        """Returns the robustness value rho at time step k[cite: 109]."""
        pass

# --- Boolean Operators ---
class Negation(STLFormula):
    def __init__(self, formula: STLFormula):
        self.formula = formula
    def evaluate_robust(self, joint_state, k):
        return -self.formula.evaluate_robust(joint_state, k) 

class Conjunction(STLFormula):
    def __init__(self, formulas: list[STLFormula]):
        self.formulas = formulas
    def evaluate_robust(self, joint_state, k):
        return np.min([f.evaluate_robust(joint_state, k) for f in self.formulas])

class Disjunction(STLFormula):
    def __init__(self, formulas: list[STLFormula]):
        self.formulas = formulas
    def evaluate_robust(self, joint_state, k):
        return np.max([f.evaluate_robust(joint_state, k) for f in self.formulas])

# --- Temporal Operators ---
class Always(STLFormula):
    def __init__(self, formula: STLFormula, interval: tuple[int, int]):
        self.formula = formula
        self.a, self.b = interval
    def evaluate_robust(self, joint_state, k):
        return np.min([self.formula.evaluate_robust(joint_state, i) 
                      for i in range(k + self.a, k + self.b + 1)])

class Eventually(STLFormula):
    def __init__(self, formula: STLFormula, interval: tuple[int, int]):
        self.formula = formula
        self.a, self.b = interval
    def evaluate_robust(self, joint_state, k):
        return np.max([self.formula.evaluate_robust(joint_state, i) 
                      for i in range(k + self.a, k + self.b + 1)])