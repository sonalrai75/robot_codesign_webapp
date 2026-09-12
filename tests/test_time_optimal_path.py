import numpy as np
from robot_codesign.geometry.geodesic import inverse_kinematics_absolute, solve_geodesic
from robot_codesign.control.time_optimal_path import rest_to_rest_time, infer_common_torque_limit

P=dict(l1=0.3,l2=0.3,A1=0.0006,A2=0.0006,rho=2700.0,M1=0.09966,M2=0.000122)

def test_time_scales_with_inverse_sqrt_torque():
    q0=inverse_kinematics_absolute([0.4,0.0],P["l1"],P["l2"],1)
    q1=inverse_kinematics_absolute([0.4,0.2],P["l1"],P["l2"],1)
    sol=solve_geodesic(q0,q1,P)
    t1=rest_to_rest_time(sol,P,(1,1),n=2001)["time"]
    t4=rest_to_rest_time(sol,P,(4,4),n=2001)["time"]
    assert np.isclose(t4,t1/2,rtol=1e-6)

def test_infer_common_torque_limit():
    assert np.isclose(infer_common_torque_limit(0.2,0.1),4.0)
