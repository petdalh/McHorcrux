from numpy_core.colregs_stl.stl_formula import STLFormula
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