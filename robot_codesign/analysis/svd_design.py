from __future__ import annotations
import numpy as np
from robot_codesign.sensitivity.svd import central_jacobian, svd_analysis

PRIMARY_NAMES=["H_min","H_max","f1_Hz","f2_Hz"]

def normalized_performance(robot):
    return np.log(robot.performance_vector())

def analyze_design_space(robot,rel_step=1e-4):
    x=robot.design_vector()
    def fun(z): return normalized_performance(robot.with_design_vector(z))
    J=central_jacobian(fun,x,rel_step=rel_step)
    a=svd_analysis(J)
    U,s,VT=np.linalg.svd(J,full_matrices=True)
    rank=a["rank"]
    N=VT[rank:].T
    return {
        "jacobian":J,
        "singular_values":s,
        "rank":rank,
        "condition":float(a["condition"]),
        "null_basis":N,
        "active_right_vectors":VT[:rank].T,
        "primary_names":PRIMARY_NAMES,
    }

def secondary_metric(robot,objective="mass"):
    t=np.r_[np.asarray(robot.t1,float),np.asarray(robot.t2,float)]
    if objective=="mass":
        return robot.mass()
    if objective=="smoothness":
        a=np.asarray(robot.t1,float); b=np.asarray(robot.t2,float)
        return float(np.sum(np.diff(a)**2)+np.sum(np.diff(b)**2))
    if objective=="peak_width":
        return float(np.max(t))
    if objective=="rms_width":
        return float(np.sqrt(np.mean(t*t)))
    if objective=="maneuverability":
        # Maximize the smallest active singular value; return negative for minimization.
        return -float(analyze_design_space(robot)["singular_values"][-1])
    raise ValueError(f"Unknown secondary objective: {objective}")

def secondary_direction(robot,objective="mass"):
    info=analyze_design_space(robot); x=robot.design_vector(); N=info["null_basis"]
    def g(z): return secondary_metric(robot.with_design_vector(z),objective)
    grad=np.empty_like(x); h=1e-4
    for j in range(len(x)):
        zp=x.copy(); zm=x.copy(); zp[j]+=h; zm[j]-=h
        grad[j]=(g(zp)-g(zm))/(2*h)
    d=-(N@(N.T@grad)) if N.size else np.zeros_like(x)
    norm=np.linalg.norm(d)
    if norm < 1e-8*max(1.0,np.linalg.norm(grad)):
        return np.zeros_like(x),info
    return d/norm,info

def corrected_null_move(robot,direction,amplitude,target=None,max_iter=12,tol=5e-7):
    """Move along a chosen design-space direction, then correct primary metrics.

    The correction is minimum-norm in log-design coordinates using the local
    normalized-performance Jacobian. This preserves the requested nonlinear
    primary performance vector to numerical tolerance when a nearby solution exists.
    """
    x0=robot.design_vector()
    target=np.asarray(robot.performance_vector() if target is None else target,float)
    d=np.asarray(direction,float); d=d/max(np.linalg.norm(d),1e-15)
    x=x0+float(amplitude)*d
    converged=False
    for _ in range(max_iter):
        r=np.log(robot.with_design_vector(x).performance_vector()/target)
        if np.linalg.norm(r,np.inf)<tol:
            converged=True; break
        cur=robot.with_design_vector(x)
        J=analyze_design_space(cur)["jacobian"]
        dx=-np.linalg.pinv(J,rcond=1e-10)@r
        # trust region in log-width coordinates
        n=np.linalg.norm(dx)
        if n>.25: dx*=.25/n
        x=x+dx
    new=robot.with_design_vector(x)
    residual=np.log(new.performance_vector()/target)
    return new,converged,float(np.linalg.norm(residual,np.inf)),float(np.linalg.norm(x-x0))
