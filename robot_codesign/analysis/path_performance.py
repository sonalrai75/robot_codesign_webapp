from __future__ import annotations
import numpy as np
from scipy.optimize import minimize


def metric_length(robot,path):
    speed=np.array([np.sqrt(max(v@robot.inertia_matrix(q)@v,0.0)) for q,v in zip(path.q,path.q_u)])
    return float(np.trapezoid(speed,path.u))


def euclidean_joint_length(path):
    return float(np.trapezoid(np.linalg.norm(path.q_u,axis=1),path.u))


def _metric_derivatives(robot,q,eps=1e-6):
    d=np.empty((robot.dof,robot.dof,robot.dof))
    for k in range(robot.dof):
        dq=np.zeros(robot.dof); dq[k]=eps
        d[k]=(robot.inertia_matrix(q+dq)-robot.inertia_matrix(q-dq))/(2*eps)
    return d


def _coriolis_quadratic(robot,q,v):
    # covariant first-kind Christoffel quadratic c_i = Gamma_{ijk} v_j v_k
    d=_metric_derivatives(robot,q)
    n=robot.dof; c=np.zeros(n)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                gamma=.5*(d[k,i,j]+d[j,i,k]-d[i,j,k])
                c[i]+=gamma*v[j]*v[k]
    return c


def fixed_path_min_time(robot,path,torque_limits,n_nodes=31):
    """Approximate torque-constrained rest-to-rest time on an arbitrary fixed path.

    The squared path speed x=(du/dt)^2 is optimized at grid nodes.  This is intended
    for interactive comparison; a future direct-collocation module can replace it.
    """
    # resample path onto modest grid via interpolation
    u=np.linspace(0,1,n_nodes)
    q=np.column_stack([np.interp(u,path.u,path.q[:,j]) for j in range(robot.dof)])
    qu=np.column_stack([np.interp(u,path.u,path.q_u[:,j]) for j in range(robot.dof)])
    quu=np.column_stack([np.interp(u,path.u,path.q_uu[:,j]) for j in range(robot.dof)])
    um=.5*(u[:-1]+u[1:]); du=u[1]-u[0]
    qm=.5*(q[:-1]+q[1:]); qum=.5*(qu[:-1]+qu[1:]); quum=.5*(quu[:-1]+quu[1:])
    A=[]; B=[]
    for qi,vi,ai in zip(qm,qum,quum):
        H=robot.inertia_matrix(qi)
        A.append(H@vi)
        B.append(H@ai+_coriolis_quadratic(robot,qi,vi))
    A=np.asarray(A); B=np.asarray(B)
    lim=np.asarray(torque_limits,float)
    shape=u*(1-u); c=.05
    z0=(c*shape)[1:-1]
    def unpack(z): return np.r_[0,np.maximum(z,1e-12),0]
    def torques(z):
        x=unpack(z); acc=(x[1:]-x[:-1])/(2*du); xm=.5*(x[1:]+x[:-1])
        return A*acc[:,None]+B*xm[:,None]
    def objective(z):
        x=unpack(z)
        return float(np.sum(2*du/(np.sqrt(x[:-1])+np.sqrt(x[1:])+1e-15)))
    def constraint(z): return (lim[None,:]**2-torques(z)**2).ravel()
    res=minimize(objective,z0,method="SLSQP",bounds=[(1e-12,None)]*(n_nodes-2),constraints={"type":"ineq","fun":constraint},options={"maxiter":220,"ftol":2e-7,"disp":False})
    x=unpack(res.x)
    tau=torques(res.x)
    acc=(x[1:]-x[:-1])/(2*du)
    xm=.5*(x[1:]+x[:-1])
    return {
        "time_s":objective(res.x),
        "success":bool(res.success),
        "max_torque_ratio":float(np.max(np.abs(tau)/lim[None,:])),
        # Profiles are returned for interactive inspection. Torque and path
        # acceleration live at segment midpoints; squared path speed lives at nodes.
        "u_nodes":u.tolist(),
        "speed_sq_nodes":x.tolist(),
        "u_mid":um.tolist(),
        "path_accel_mid":acc.tolist(),
        "torque_mid":tau.tolist(),
        "speed_sq_mid":xm.tolist(),
    }


def geodesic_min_time_control(robot,path,torque_limits,n_nodes=101):
    """Chapter-5-style time-optimal motion along an inertia-metric geodesic.

    The geodesic removes the velocity-quadratic term when parameterized by
    Riemannian arc length s, so joint torques satisfy

        tau = H(q) q_s * s_ddot.

    The admissible scalar acceleration is therefore set by whichever joint
    torque limit becomes active first.  The time-optimal rest-to-rest policy
    uses the positive acceleration bound before the switching point and the
    negative bound after it.  At least one joint is saturated (to numerical
    precision) on each acceleration/deceleration interval.
    """
    u=np.linspace(0.0,1.0,n_nodes)
    q=np.column_stack([np.interp(u,path.u,path.q[:,j]) for j in range(robot.dof)])
    qu=np.column_stack([np.interp(u,path.u,path.q_u[:,j]) for j in range(robot.dof)])
    lim=np.asarray(torque_limits,float)

    # Metric arc length s(u).
    c=np.empty(n_nodes)
    for i,(qi,vi) in enumerate(zip(q,qu)):
        H=robot.inertia_matrix(qi)
        c[i]=np.sqrt(max(float(vi@H@vi),1e-30))
    du=np.diff(u)
    s=np.r_[0.0,np.cumsum(0.5*(c[:-1]+c[1:])*du)]
    ds=np.diff(s)

    # Midpoint torque map b(s)=H dq/ds and corresponding acceleration bound.
    um=.5*(u[:-1]+u[1:])
    qm=.5*(q[:-1]+q[1:])
    qum=.5*(qu[:-1]+qu[1:])
    B=np.empty((n_nodes-1,robot.dof))
    amax=np.empty(n_nodes-1)
    active=np.empty(n_nodes-1,dtype=int)
    for k,(qi,vi) in enumerate(zip(qm,qum)):
        H=robot.inertia_matrix(qi)
        cm=np.sqrt(max(float(vi@H@vi),1e-30))
        b=H@(vi/cm)
        B[k]=b
        ratios=lim/np.maximum(np.abs(b),1e-15)
        active[k]=int(np.argmin(ratios))
        amax[k]=float(np.min(ratios))

    # Forward/backward reachable squared speed in Riemannian arc length.
    vf2=np.zeros(n_nodes)
    for k in range(n_nodes-1):
        vf2[k+1]=vf2[k]+2.0*amax[k]*ds[k]
    vb2=np.zeros(n_nodes)
    for k in range(n_nodes-2,-1,-1):
        vb2[k]=vb2[k+1]+2.0*amax[k]*ds[k]
    v2=np.minimum(vf2,vb2)
    switch=int(np.argmin(np.abs(vf2-vb2)))

    # Bang-bang sign policy along the joint-space geodesic.
    acc=np.where(np.arange(n_nodes-1)<switch,amax,-amax)
    tau=B*acc[:,None]

    # Time grid from the speed envelope.
    v=np.sqrt(np.maximum(v2,0.0))
    dt=np.empty(n_nodes-1)
    for k in range(n_nodes-1):
        denom=v[k]+v[k+1]
        dt[k]=2.0*ds[k]/max(denom,1e-15)
    t=np.r_[0.0,np.cumsum(dt)]
    tm=.5*(t[:-1]+t[1:])

    ratio=np.abs(tau)/lim[None,:]
    return {
        "time_s":float(t[-1]),
        "success":True,
        "solver_kind":"geodesic_bang_bang",
        "max_torque_ratio":float(np.max(ratio)),
        "u_nodes":u.tolist(),
        "u_mid":um.tolist(),
        "s_nodes":s.tolist(),
        "time_nodes":t.tolist(),
        "time_mid":tm.tolist(),
        "path_speed_nodes":v.tolist(),
        "path_speed_sq_nodes":v2.tolist(),
        "path_accel_mid":acc.tolist(),
        "path_accel_limit_mid":amax.tolist(),
        "torque_mid":tau.tolist(),
        "active_joint_mid":(active+1).tolist(),
        "active_sign_mid":np.sign(acc).astype(int).tolist(),
        "torque_ratio_mid":ratio.tolist(),
        "switch_index":switch,
        "switch_s":float(s[switch]),
        "switch_time_s":float(t[switch]),
        "metric_speed_du":c.tolist(),
    }


def path_min_time_control(robot,path,torque_limits,n_nodes=31):
    """Dispatch to the geodesic bang-bang reduction or generic fixed-path solver."""
    if getattr(path,"name","")=="geodesic":
        return geodesic_min_time_control(robot,path,torque_limits,n_nodes=max(101,n_nodes))
    out=fixed_path_min_time(robot,path,torque_limits,n_nodes=n_nodes)
    out["solver_kind"]="generic_fixed_path_slsqp"
    return out
