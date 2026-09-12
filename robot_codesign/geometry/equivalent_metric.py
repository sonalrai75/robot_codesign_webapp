from __future__ import annotations
import numpy as np
from scipy.integrate import solve_bvp

class CosineCoupledMetric:
    """Two-link metric H(q) with constant diagonals and cosine coupling.

    H(q) = [[h11, c*cos(q1-q2)],
            [c*cos(q1-q2), h22]]

    This matches the Chapter 5 rigid-body metric structure.  It is useful when
    working with an equivalent final metric inferred from published eigenvalues,
    without pretending that the underlying distributed shape has been recovered.
    """
    def __init__(self, h11: float, h22: float, coupling: float):
        self.h11=float(h11)
        self.h22=float(h22)
        self.coupling=float(coupling)

    def matrix(self, q):
        q=np.asarray(q,dtype=float)
        c=self.coupling*np.cos(q[0]-q[1])
        return np.array([[self.h11,c],[c,self.h22]],dtype=float)

    def derivatives(self, q):
        q=np.asarray(q,dtype=float)
        s=np.sin(q[0]-q[1])
        out=np.zeros((2,2,2),dtype=float)
        out[0,0,1]=out[0,1,0]=-self.coupling*s
        out[1,0,1]=out[1,1,0]= self.coupling*s
        return out

    def christoffel(self,q):
        G=self.matrix(q)
        Gi=np.linalg.inv(G)
        dG=self.derivatives(q)
        Gamma=np.zeros((2,2,2),dtype=float)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    Gamma[i,j,k]=0.5*sum(
                        Gi[i,l]*(dG[j,l,k]+dG[k,l,j]-dG[l,j,k])
                        for l in range(2)
                    )
        return Gamma

def solve_metric_geodesic(q_start,q_end,metric,n=101,tol=1e-8):
    q_start=np.asarray(q_start,dtype=float)
    q_end=np.asarray(q_end,dtype=float)
    u=np.linspace(0.0,1.0,n)
    qg=q_start[:,None]+(q_end-q_start)[:,None]*u
    vg=np.repeat((q_end-q_start)[:,None],n,axis=1)
    yg=np.vstack([qg,vg])

    def fun(u,y):
        out=np.zeros_like(y)
        out[:2]=y[2:]
        for k in range(y.shape[1]):
            v=y[2:,k]
            Gamma=metric.christoffel(y[:2,k])
            out[2:,k]=[
                -np.einsum("jk,j,k->",Gamma[i],v,v)
                for i in range(2)
            ]
        return out

    def bc(ya,yb):
        return np.r_[ya[:2]-q_start,yb[:2]-q_end]

    sol=solve_bvp(fun,bc,u,yg,tol=tol,max_nodes=5000)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol

def rest_to_rest_metric_time(sol,metric,torque_limits=(40.0,40.0),n=8001):
    u=np.linspace(0.0,1.0,n)
    y=sol.sol(u)
    c=np.empty(n)
    B=np.empty((2,n))
    for k in range(n):
        q=y[:2,k]
        qu=y[2:,k]
        G=metric.matrix(q)
        c[k]=np.sqrt(qu@G@qu)
        B[:,k]=G@(qu/c[k])

    s=np.r_[0.0,np.cumsum(0.5*(c[1:]+c[:-1])*np.diff(u))]
    limits=np.asarray(torque_limits,dtype=float)[:,None]
    amax=np.min(limits/np.maximum(np.abs(B),1e-15),axis=0)
    ds=np.diff(s)

    vf2=np.zeros(n)
    vf2[1:]=2*np.cumsum(0.5*(amax[1:]+amax[:-1])*ds)

    vb2=np.zeros(n)
    vb2[:-1]=(2*np.cumsum(
        (0.5*(amax[1:]+amax[:-1])*ds)[::-1]
    ))[::-1]

    v2=np.minimum(vf2,vb2)
    vmid=np.sqrt(np.maximum(0.5*(v2[:-1]+v2[1:]),1e-30))
    return {
        "time": float(np.sum(ds/vmid)),
        "path_length": float(s[-1]),
    }

def metric_from_eigenvalues_and_angle(lam_min,lam_max,angle_rad):
    """Build H(q1-q2=0) from published eigenvalues and an eigenvector angle."""
    V=np.array([
        [np.cos(angle_rad),-np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    H0=V@np.diag([lam_min,lam_max])@V.T
    return CosineCoupledMetric(H0[0,0],H0[1,1],H0[0,1])
