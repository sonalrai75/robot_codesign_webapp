from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class RobotModel(ABC):
    """DOF-agnostic robot model contract used by the web/API layer."""
    @property
    @abstractmethod
    def dof(self) -> int: ...

    @abstractmethod
    def forward_kinematics(self, q: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def inverse_kinematics(self, x: np.ndarray, branch: int = 1) -> np.ndarray: ...

    @abstractmethod
    def inertia_matrix(self, q: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def performance_vector(self) -> np.ndarray: ...

    @abstractmethod
    def design_vector(self) -> np.ndarray: ...

    @abstractmethod
    def with_design_vector(self, x: np.ndarray) -> "RobotModel": ...
