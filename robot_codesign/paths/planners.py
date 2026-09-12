from __future__ import annotations
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from robot_codesign.geometry.geodesic import solve_geodesic
from .base import PathPlanner, PathResult


def _unwrap_rows(q):
    q=np.asarray(q,float).copy()
    for k in range(1,len(q)):
        q[k]=q[k-1]+np.angle(np.exp(1j*(q[k]-q[k-1])))
    return q

class JointLinePlanner(PathPlanner):
    name="joint_line"
    def plan(self,robot,start_xy,end_xy,branch=1,n=101):
        q0=robot.inverse_kinematics(start_xy,branch); q1=robot.inverse_kinematics(end_xy,branch)
        u=np.linspace(0,1,n); d=q1-q0
        q=q0[None,:]+u[:,None]*d[None,:]
        qu=np.repeat(d[None,:],n,axis=0); quu=np.zeros_like(qu)
        return PathResult(self.name,u,q,qu,quu,{"branch":branch})

class CartesianLinePlanner(PathPlanner):
    name="cartesian_line"
    def plan(self,robot,start_xy,end_xy,branch=1,n=101):
        ud=np.linspace(0,1,max(301,n*3)); x0=np.asarray(start_xy,float); x1=np.asarray(end_xy,float)
        qd=_unwrap_rows(np.array([robot.inverse_kinematics(x0+t*(x1-x0),branch) for t in ud]))
        spl=[CubicSpline(ud,qd[:,j],bc_type="natural") for j in range(robot.dof)]
        u=np.linspace(0,1,n)
        q=np.column_stack([s(u) for s in spl]); qu=np.column_stack([s(u,1) for s in spl]); quu=np.column_stack([s(u,2) for s in spl])
        return PathResult(self.name,u,q,qu,quu,{"branch":branch})

class GeodesicPlanner(PathPlanner):
    name="geodesic"
    def plan(self,robot,start_xy,end_xy,branch=1,n=101):
        q0=robot.inverse_kinematics(start_xy,branch); q1=robot.inverse_kinematics(end_xy,branch)
        sol=solve_geodesic(q0,q1,n=min(101,n),tol=1e-6,metric=robot.inertia_matrix)
        u=np.linspace(0,1,n); y=sol.sol(u); yp=sol.sol(u,1)
        return PathResult(self.name,u,y[:2].T,y[2:].T,yp[2:].T,{"branch":branch,"bvp_success":bool(sol.success)})

class CubicOptimizedPlanner(PathPlanner):
    """Experimental low-dimensional path optimization.

    Optimizes a common cubic-Bezier offset in 2-DOF joint space.  This deliberately
    keeps V0.1 responsive; later versions can swap in direct collocation without
    changing the PathPlanner interface.
    """
    name="optimized_cubic"
    def _make(self,q0,q1,off,u):
        P0=q0; P3=q1; d=q1-q0; P1=q0+d/3+off; P2=q0+2*d/3+off
        s=1-u
        q=(s**3)[:,None]*P0+3*(s*s*u)[:,None]*P1+3*(s*u*u)[:,None]*P2+(u**3)[:,None]*P3
        qu=3*(s*s)[:,None]*(P1-P0)+6*(s*u)[:,None]*(P2-P1)+3*(u*u)[:,None]*(P3-P2)
        quu=6*s[:,None]*(P2-2*P1+P0)+6*u[:,None]*(P3-2*P2+P1)
        return q,qu,quu
    def plan(self,robot,start_xy,end_xy,branch=1,n=101):
        q0=robot.inverse_kinematics(start_xy,branch); q1=robot.inverse_kinematics(end_xy,branch)
        uc=np.linspace(0,1,31)
        def objective(off):
            q,qu,_=self._make(q0,q1,np.asarray(off),uc)
            v=[]
            for qi,vi in zip(q,qu): v.append(np.sqrt(max(vi@robot.inertia_matrix(qi)@vi,0)))
            return float(np.trapezoid(v,uc))
        opt=minimize(objective,np.zeros(2),method="Nelder-Mead",options={"maxiter":35,"xatol":1e-3,"fatol":1e-6})
        u=np.linspace(0,1,n); q,qu,quu=self._make(q0,q1,opt.x,u)
        return PathResult(self.name,u,q,qu,quu,{"branch":branch,"offset":opt.x.tolist(),"optimization_success":bool(opt.success),"objective":"Riemannian path length (V0.1)"})

PLANNERS={p.name:p for p in [GeodesicPlanner(),JointLinePlanner(),CartesianLinePlanner(),CubicOptimizedPlanner()]}
