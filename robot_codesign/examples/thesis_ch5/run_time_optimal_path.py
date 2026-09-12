import numpy as np
from robot_codesign.geometry.geodesic import inverse_kinematics_absolute, solve_geodesic
from robot_codesign.control.time_optimal_path import rest_to_rest_time, infer_common_torque_limit

rho=2700.0
M1=0.09966
M2=0.000122
designs={
 "initial": dict(l1=0.3,l2=0.3,A1=0.0006,A2=0.0006,rho=rho,M1=M1,M2=M2),
 "intermediate": dict(l1=0.1419,l2=0.4581,A1=0.0019,A2=0.0002,rho=rho,M1=M1,M2=M2),
}
tasks={
 "task_1": (np.array([0.4,0.0]), np.array([0.4,0.2]), 0.0487, 0.0272),
 "task_2": (np.array([0.4,0.0]), np.array([0.6,0.0]), 0.0635, 0.0490),
}

for tname,(p0,p1,t_initial_thesis,t_inter_thesis) in tasks.items():
    print(tname)
    best={}
    for dname,p in designs.items():
        candidates=[]
        for elbow0 in (+1,-1):
            for elbow1 in (+1,-1):
                q0=inverse_kinematics_absolute(p0,p["l1"],p["l2"],elbow0)
                q1=inverse_kinematics_absolute(p1,p["l1"],p["l2"],elbow1)
                sol=solve_geodesic(q0,q1,p)
                r=rest_to_rest_time(sol,p,(1.0,1.0))
                candidates.append((r["time"],r["path_length"],elbow0,elbow1))
        candidates.sort()
        best[dname]=candidates[0]
        print(" ",dname,"best U=1 candidate:",candidates[0])
    U=infer_common_torque_limit(best["initial"][0],t_initial_thesis)
    pred=best["intermediate"][0]/np.sqrt(U)
    print(f" inferred common equal joint torque from initial target: {U:.6f} N-m")
    print(f" predicted intermediate time: {pred:.6f} s")
    print(f" thesis intermediate time:    {t_inter_thesis:.6f} s")
    print()
