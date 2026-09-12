import numpy as np
from robot_codesign.geometry.geodesic import (
    inverse_kinematics_absolute, endpoint_xy, solve_geodesic, geodesic_residual
)

P=dict(l1=0.3,l2=0.3,A1=0.0006,A2=0.0006,rho=2700.0,M1=0.09966,M2=0.000122)

def test_absolute_angle_ik():
    for pt in ([0.4,0.0],[0.4,0.2],[0.6,0.0]):
        q=inverse_kinematics_absolute(pt,P["l1"],P["l2"],elbow=1)
        assert np.allclose(endpoint_xy(q,P["l1"],P["l2"]),pt,atol=1e-10)

def test_geodesic_bvp_residual():
    q0=inverse_kinematics_absolute([0.4,0.0],P["l1"],P["l2"],elbow=1)
    q1=inverse_kinematics_absolute([0.4,0.2],P["l1"],P["l2"],elbow=1)
    sol=solve_geodesic(q0,q1,P,tol=1e-6)
    assert geodesic_residual(sol,P) < 1e-5
