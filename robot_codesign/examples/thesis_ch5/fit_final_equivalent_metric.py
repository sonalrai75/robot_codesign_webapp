import numpy as np
from scipy.optimize import minimize_scalar
from robot_codesign.geometry.geodesic import inverse_kinematics_absolute
from robot_codesign.geometry.equivalent_metric import (
    metric_from_eigenvalues_and_angle,
    solve_metric_geodesic,
    rest_to_rest_metric_time,
)

# Published FINAL-DESIGN inertia eigenvalues from Table 5.1.
LAM_MIN=0.0066
LAM_MAX=0.0281

# Chapter 5 final link lengths.
L1=0.1419
L2=0.4581

# 40 N-m is treated here as an inferred working hypothesis, not a stated
# Chapter 5 parameter. It is close to the value that reproduces the original
# design travel times and is also a torque level used experimentally in Ch. 4.
U=(40.0,40.0)

tasks={
 "task_1": (np.array([0.4,0.0]),np.array([0.4,0.2]),0.0272,-1,-1),
 "task_2": (np.array([0.4,0.0]),np.array([0.6,0.0]),0.0490,+1,-1),
}

def evaluate(angle):
    metric=metric_from_eigenvalues_and_angle(LAM_MIN,LAM_MAX,angle)
    rows={}
    for name,(p0,p1,target,e0,e1) in tasks.items():
        q0=inverse_kinematics_absolute(p0,L1,L2,e0)
        q1=inverse_kinematics_absolute(p1,L1,L2,e1)
        sol=solve_metric_geodesic(q0,q1,metric)
        r=rest_to_rest_metric_time(sol,metric,U)
        rows[name]=(r["time"],r["path_length"],target)
    return rows

def objective(angle):
    rows=evaluate(angle)
    return sum(((t-target)/target)**2 for t,_,target in rows.values())

fit=minimize_scalar(objective,bounds=(-np.pi/2,np.pi/2),method="bounded")
rows=evaluate(fit.x)
metric=metric_from_eigenvalues_and_angle(LAM_MIN,LAM_MAX,fit.x)
print(f"best equivalent-metric angle = {np.degrees(fit.x):.4f} deg")
print("H(q1-q2=0) =")
print(metric.matrix([0.0,0.0]))
print()
for name,(t,L,target) in rows.items():
    print(f"{name}: predicted={t:.6f} s, thesis={target:.6f} s, error={(t/target-1)*100:+.2f}%")
