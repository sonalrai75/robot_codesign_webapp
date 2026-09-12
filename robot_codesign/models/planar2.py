from __future__ import annotations
from dataclasses import dataclass, replace
import numpy as np
from robot_codesign.dynamics.inertia import inertia_matrix as chapter5_inertia, inertia_matrix_distributed
from robot_codesign.geometry.geodesic import endpoint_xy, inverse_kinematics_absolute
from robot_codesign.fem.two_link_variable import flexible_frequencies_variable_hz

@dataclass(frozen=True)
class Planar2DOFRobot:
    l1: float = 0.1419
    l2: float = 0.4581
    rho: float = 2700.0
    E: float = 2.7e10
    joint_mass: float = 0.09966
    tip_mass: float = 0.000122
    width1_out: float = 0.010
    width2_out: float = 0.038
    # in-plane dimensions, one per FE element
    t1: tuple = tuple([0.19] * 10)
    t2: tuple = tuple([0.005263157894736842] * 10)

    @property
    def dof(self): return 2

    def forward_kinematics(self, q):
        return endpoint_xy(np.asarray(q,float), self.l1, self.l2)

    def inverse_kinematics(self, x, branch=1):
        return inverse_kinematics_absolute(np.asarray(x,float), self.l1, self.l2, elbow=branch)

    def _areas(self):
        # Average areas are retained only for legacy/reporting compatibility.
        return self.width1_out*float(np.mean(self.t1)), self.width2_out*float(np.mean(self.t2))

    def inertia_params(self):
        A1,A2=self._areas()
        return dict(l1=self.l1,l2=self.l2,A1=A1,A2=A2,rho=self.rho,M1=self.joint_mass,M2=self.tip_mass)

    def inertia_matrix(self, q):
        # Use every structural element in the rigid-body inertia.  For a uniform
        # section this reduces exactly to the historical Chapter-5 equations.
        areas1=self.width1_out*np.asarray(self.t1,float)
        areas2=self.width2_out*np.asarray(self.t2,float)
        return inertia_matrix_distributed(
            *np.asarray(q,float), self.l1, self.l2, areas1, areas2,
            self.rho, self.joint_mass, self.tip_mass)

    def frequencies_hz(self):
        f=flexible_frequencies_variable_hz(
            self.l1,self.l2,np.asarray(self.t1),np.asarray(self.t2),
            self.width1_out,self.width2_out,self.E,self.rho,
            M_joint2=self.joint_mass,M_tip=self.tip_mass)
        # The assembled model has a near-zero rigid/mechanism mode; flexible
        # performance begins with the first physically meaningful bending mode.
        f=f[f > 1.0]
        if len(f)<2:
            raise RuntimeError("FE model returned fewer than two flexible modes")
        return f[:2]

    def worst_case_inertia_eigs(self):
        return np.linalg.eigvalsh(self.inertia_matrix(np.array([0.0,0.0])))

    def performance_vector(self):
        h=self.worst_case_inertia_eigs(); f=self.frequencies_hz()
        return np.array([h[0],h[-1],f[0],f[1]],float)

    def mass(self):
        A1=self.width1_out*np.mean(self.t1); A2=self.width2_out*np.mean(self.t2)
        return float(self.rho*(self.l1*A1+self.l2*A2)+self.joint_mass+self.tip_mass)

    def design_vector(self):
        return np.log(np.r_[np.asarray(self.t1,float),np.asarray(self.t2,float)])

    def with_design_vector(self, x):
        t=np.exp(np.asarray(x,float)); n=len(self.t1)
        if len(t)!=2*n: raise ValueError(f"Expected {2*n} design variables")
        return replace(self,t1=tuple(t[:n]),t2=tuple(t[n:]))
