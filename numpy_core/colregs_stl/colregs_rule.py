from numpy_core.colregs_stl.stl_core import Conjunction, Negation, Always, Eventually, IA_Implication
from numpy_core.colregs_stl.atomic_predicates import (
    PositionHalfplane,
    OrientationHalfplane,
    ChangeCourse,
    VelocityHalfplane,
    TimeHorizon,
)

class COLREGsRule:
    def __init__(self, name: str, params: dict):
        self.name = name
        self.params = params
        self.antecedent, self.consequent = self._build_logic()
        self.ia_wrapper = IA_Implication(self.antecedent, self.consequent)

    def _build_logic(self):
        raise NotImplementedError

    def evaluate(self, joint_state, k):
        return self.ia_wrapper.evaluate_robust(joint_state, k)

class CrossingRule(COLREGsRule):
    def _build_logic(self):
        in_pos = Conjunction([
            PositionHalfplane(self.params['beta_low'], self.params['v_max']),
            Negation(PositionHalfplane(self.params['beta_high'], self.params['v_max']))
        ])
        
        in_ori = Conjunction([
            OrientationHalfplane(self.params['gamma_low'], self.params['omega_max']),
            OrientationHalfplane(self.params['gamma_high'], self.params['omega_max'])
        ])

        risk = Conjunction([
            VelocityHalfplane(-self.params['epsilon'], self.params['a_max'], self.params['d_zone']),
            VelocityHalfplane(self.params['epsilon'], self.params['a_max'], self.params['d_zone']),
            TimeHorizon(self.params['t_h'], self.params['a_max'])
        ])

        encounter = Conjunction([in_pos, in_ori, risk])
        antecedent = Always(encounter, (0, self.params['t_p']))

        maneuver = Eventually(
            ChangeCourse(-self.params['delta'], self.params['alpha_max']),
            (self.params['t_p'], self.params['t_p'] + self.params['t_m'])
        )
        consequent = Conjunction([maneuver, Eventually(Negation(risk), (self.params['t_p'], self.params['t_p'] + 2*self.params['t_m']))])

        return antecedent, consequent

class HeadOnRule(COLREGsRule):
    def _build_logic(self):
        in_pos = Conjunction([
            PositionHalfplane(self.params['beta_low'], self.params['v_max']),
            Negation(PositionHalfplane(self.params['beta_high'], self.params['v_max']))
        ])
        
        in_ori = Conjunction([
            OrientationHalfplane(self.params['gamma_low'], self.params['omega_max']),
            OrientationHalfplane(self.params['gamma_high'], self.params['omega_max'])
        ])

        risk = Conjunction([
            VelocityHalfplane(-self.params['epsilon'], self.params['a_max'], self.params['d_zone']),
            VelocityHalfplane(self.params['epsilon'], self.params['a_max'], self.params['d_zone']),
            TimeHorizon(self.params['t_h'], self.params['a_max'])
        ])

        encounter = Conjunction([in_pos, in_ori, risk])
        antecedent = Always(encounter, (0, self.params['t_p']))

        maneuver = Eventually(
            ChangeCourse(-self.params['delta'], self.params['alpha_max']),
            (self.params['t_p'], self.params['t_p'] + self.params['t_m'])
        )
        
        clearance = Eventually(
            Negation(risk),
            (self.params['t_p'], self.params['t_p'] + 2 * self.params['t_m'])
        )

        consequent = Conjunction([maneuver, clearance])

        return antecedent, consequent