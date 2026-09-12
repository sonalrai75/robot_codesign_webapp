from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class PathResult:
    name: str
    u: np.ndarray
    q: np.ndarray          # shape (N,dof)
    q_u: np.ndarray        # shape (N,dof)
    q_uu: np.ndarray       # shape (N,dof)
    metadata: dict

class PathPlanner:
    name="base"
    def plan(self, robot, start_xy, end_xy, branch=1, n=101) -> PathResult:
        raise NotImplementedError
