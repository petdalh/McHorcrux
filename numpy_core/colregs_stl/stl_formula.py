from abc import ABC, abstractmethod

class STLFormula(ABC):
    @abstractmethod
    def evaluate_robust(self, joint_state, k):
        raise NotImplementedError