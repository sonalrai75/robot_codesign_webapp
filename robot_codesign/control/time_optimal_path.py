from __future__ import annotations
import numpy as np
from robot_codesign.dynamics.inertia import inertia_matrix

def geodesic_path_data(sol, params, n=8001):
    """Return arc-length coordinate and torque map b(s)=H(q)dq/ds."""
    u=np.linspace(0.0,1.0,n)
    y=sol.sol(u)
    metric_speed=np.empty(n)
    B=np.empty((2,n))
    for k in range(n):
        q=y[:2,k]
        qu=y[2:,k]
        H=inertia_matrix(*q, **params)
        c=np.sqrt(qu @ H @ qu)
        metric_speed[k]=c
        qs=qu/c
        B[:,k]=H @ qs
    s=np.r_[0.0, np.cumsum(
        0.5*(metric_speed[1:]+metric_speed[:-1])*np.diff(u)
    )]
    return s,B,metric_speed

def rest_to_rest_time(sol, params, torque_limits=(1.0,1.0), n=8001):
    """Time-optimal rest-to-rest motion along a fixed geodesic.

    Uses Chapter 5's geodesic reduction T_i = [H dq/ds]_i * d2s/dt2.
    Independent symmetric joint torque limits are assumed.

    This is a reconstruction utility, not yet claimed as an exact reproduction
    of the unpublished numerical implementation used in the 1993 thesis.
    """
    s,B,c=geodesic_path_data(sol,params,n=n)
    limits=np.asarray(torque_limits,dtype=float)[:,None]
    absB=np.maximum(np.abs(B),1e-15)
    amax=np.min(limits/absB,axis=0)

    ds=np.diff(s)
    vf2=np.zeros_like(s)
    vf2[1:]=2*np.cumsum(0.5*(amax[1:]+amax[:-1])*ds)

    vb2=np.zeros_like(s)
    back=2*np.cumsum((0.5*(amax[1:]+amax[:-1])*ds)[::-1])
    vb2[:-1]=back[::-1]

    v2=np.minimum(vf2,vb2)
    vmid=np.sqrt(np.maximum(0.5*(v2[:-1]+v2[1:]),1e-30))
    time=float(np.sum(ds/vmid))
    switch=int(np.argmin(np.abs(vf2-vb2)))
    return {
        "time": time,
        "switch_s": float(s[switch]),
        "path_length": float(s[-1]),
        "s": s,
        "amax": amax,
        "speed": np.sqrt(np.maximum(v2,0.0)),
        "torque_map": B,
    }

def infer_common_torque_limit(unit_torque_time, target_time):
    """For a common scaling of all torque limits, time scales as U^(-1/2)."""
    return float((unit_torque_time/target_time)**2)
