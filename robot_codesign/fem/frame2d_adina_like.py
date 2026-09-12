
import numpy as np
from scipy.linalg import eigvalsh

def frame2d_local(E, rho, A, I, L):
    k = np.array([
        [ E*A/L,0,0,-E*A/L,0,0],
        [0,12*E*I/L**3,6*E*I/L**2,0,-12*E*I/L**3,6*E*I/L**2],
        [0,6*E*I/L**2,4*E*I/L,0,-6*E*I/L**2,2*E*I/L],
        [-E*A/L,0,0,E*A/L,0,0],
        [0,-12*E*I/L**3,-6*E*I/L**2,0,12*E*I/L**3,-6*E*I/L**2],
        [0,6*E*I/L**2,2*E*I/L,0,-6*E*I/L**2,4*E*I/L]], float)
    m = rho*A*L/420*np.array([
        [140,0,0,70,0,0],
        [0,156,22*L,0,54,-13*L],
        [0,22*L,4*L**2,0,13*L,-3*L**2],
        [70,0,0,140,0,0],
        [0,54,13*L,0,156,-22*L],
        [0,-13*L,-3*L**2,0,-22*L,4*L**2]], float)
    return m,k

def transform(theta):
    c,s=np.cos(theta),np.sin(theta)
    R=np.array([[c,s,0],[-s,c,0],[0,0,1.]])
    T=np.zeros((6,6)); T[:3,:3]=R; T[3:,3:]=R
    return T
