from __future__ import annotations
import numpy as np
from scipy.linalg import eigvalsh

def frame_element(E,A,I,rho,L,phi):
    """2-D Euler-Bernoulli frame element: [u,v,theta] at each node."""
    c,s=np.cos(phi),np.sin(phi)
    k=np.array([
      [E*A/L,0,0,-E*A/L,0,0],
      [0,12*E*I/L**3,6*E*I/L**2,0,-12*E*I/L**3,6*E*I/L**2],
      [0,6*E*I/L**2,4*E*I/L,0,-6*E*I/L**2,2*E*I/L],
      [-E*A/L,0,0,E*A/L,0,0],
      [0,-12*E*I/L**3,-6*E*I/L**2,0,12*E*I/L**3,-6*E*I/L**2],
      [0,6*E*I/L**2,2*E*I/L,0,-6*E*I/L**2,4*E*I/L]
    ],float)
    # consistent axial + bending mass
    m=rho*A*L/420*np.array([
      [140,0,0,70,0,0],[0,156,22*L,0,54,-13*L],[0,22*L,4*L**2,0,13*L,-3*L**2],
      [70,0,0,140,0,0],[0,54,13*L,0,156,-22*L],[0,-13*L,-3*L**2,0,-22*L,4*L**2]
    ],float)
    R=np.array([[c,s,0],[-s,c,0],[0,0,1.]])
    T=np.zeros((6,6)); T[:3,:3]=R; T[3:,3:]=R
    return T.T@m@T,T.T@k@T

def polyline_frame_frequencies(nodes, thickness, width, E, rho,
                               point_masses=None, fixed_dofs=(0,1), zero_tol=1e-3):
    nodes=np.asarray(nodes,float); thickness=np.asarray(thickness,float)
    n=len(nodes)-1; nd=3*(n+1)
    M=np.zeros((nd,nd)); K=np.zeros((nd,nd))
    for e in range(n):
        d=nodes[e+1]-nodes[e]; L=np.linalg.norm(d); phi=np.arctan2(d[1],d[0])
        A=width*thickness[e]; I=width*thickness[e]**3/12
        me,ke=frame_element(E,A,I,rho,L,phi)
        ix=list(range(3*e,3*e+6))
        M[np.ix_(ix,ix)]+=me; K[np.ix_(ix,ix)]+=ke
    if point_masses:
        for node,mass in point_masses.items():
            M[3*node,3*node]+=mass; M[3*node+1,3*node+1]+=mass
    keep=[i for i in range(nd) if i not in fixed_dofs]
    lam=eigvalsh(K[np.ix_(keep,keep)],M[np.ix_(keep,keep)])
    lam=lam[lam>zero_tol]
    return np.sqrt(lam)/(2*np.pi)
