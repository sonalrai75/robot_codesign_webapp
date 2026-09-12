import numpy as np
from robot_codesign.geometry.geodesic import (
    inverse_kinematics_absolute, endpoint_xy, solve_geodesic,
    geodesic_length, geodesic_residual
)

rho=2700.0
M1=0.09966
M2=0.000122

designs = {
 "initial": dict(l1=0.3,l2=0.3,A1=0.0006,A2=0.0006,rho=rho,M1=M1,M2=M2),
 "intermediate": dict(l1=0.1419,l2=0.4581,A1=0.0019,A2=0.0002,rho=rho,M1=M1,M2=M2),
}

tasks = {
 "task_1": (np.array([0.4,0.0]), np.array([0.4,0.2])),
 "task_2": (np.array([0.4,0.0]), np.array([0.6,0.0])),
}

for tname,(p0,p1) in tasks.items():
    print(tname)
    for dname,p in designs.items():
        q0=inverse_kinematics_absolute(p0,p["l1"],p["l2"],elbow=1)
        q1=inverse_kinematics_absolute(p1,p["l1"],p["l2"],elbow=1)
        sol=solve_geodesic(q0,q1,p)
        L=geodesic_length(sol,p)
        r=geodesic_residual(sol,p)
        e0=np.linalg.norm(endpoint_xy(q0,p["l1"],p["l2"])-p0)
        e1=np.linalg.norm(endpoint_xy(q1,p["l1"],p["l2"])-p1)
        print(f"  {dname:12s}:")
        print(f"    q0(deg)={np.degrees(q0)}")
        print(f"    q1(deg)={np.degrees(q1)}")
        print(f"    geodesic length={L:.9f}")
        print(f"    max geodesic residual={r:.3e}")
        print(f"    IK endpoint errors={e0:.3e}, {e1:.3e}")
