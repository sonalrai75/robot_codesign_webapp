from __future__ import annotations
import numpy as np
from scipy.integrate import solve_bvp
from robot_codesign.dynamics.inertia import inertia_matrix

def endpoint_xy(q, l1, l2):
    """Forward kinematics for Chapter 5 absolute joint angles."""
    q1, q2 = np.asarray(q)
    return np.array([
        l1*np.cos(q1) + l2*np.cos(q2),
        l1*np.sin(q1) + l2*np.sin(q2),
    ])

def inverse_kinematics_absolute(xy, l1, l2, elbow=1):
    """IK for q1,q2 both measured from the fixed horizontal reference.

    elbow=+1 and -1 return the two mirror branches when both exist.
    """
    x, y = map(float, xy)
    r2 = x*x + y*y
    c_rel = (r2 - l1*l1 - l2*l2)/(2*l1*l2)
    if c_rel < -1-1e-12 or c_rel > 1+1e-12:
        raise ValueError("Target is outside the two-link workspace.")
    c_rel = np.clip(c_rel, -1.0, 1.0)
    theta2_rel = elbow*np.arccos(c_rel)
    q1 = np.arctan2(y, x) - np.arctan2(
        l2*np.sin(theta2_rel), l1 + l2*np.cos(theta2_rel)
    )
    q2 = q1 + theta2_rel
    return np.array([q1, q2])

def _metric_derivatives(q, params=None, eps=1e-6, metric=None):
    q = np.asarray(q, dtype=float)
    deriv = np.empty((2,2,2), dtype=float)  # dG/dq_k
    for k in range(2):
        dq = np.zeros(2); dq[k] = eps
        if metric is None:
            gp = inertia_matrix(*(q+dq), **params)
            gm = inertia_matrix(*(q-dq), **params)
        else:
            gp = metric(q+dq)
            gm = metric(q-dq)
        deriv[k] = (gp-gm)/(2*eps)
    return deriv

def christoffel_second_kind(q, params=None, metric=None):
    """Gamma^i_jk for the inertia metric H(q)."""
    G = inertia_matrix(*q, **params) if metric is None else metric(np.asarray(q,float))
    Ginv = np.linalg.inv(G)
    dG = _metric_derivatives(q, params, metric=metric)
    Gamma = np.zeros((2,2,2), dtype=float)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0.0
                for ell in range(2):
                    s += Ginv[i,ell] * (
                        dG[j,ell,k] + dG[k,ell,j] - dG[ell,j,k]
                    )
                Gamma[i,j,k] = 0.5*s
    return Gamma

def solve_geodesic(q_start, q_end, params=None, n=101, tol=1e-7, metric=None):
    """Solve d2q^i/ds2 + Gamma^i_jk dq^j/ds dq^k/ds = 0 as a BVP.

    The independent variable is an affine path parameter in [0,1].
    """
    q_start = np.asarray(q_start, dtype=float)
    q_end = np.asarray(q_end, dtype=float)
    u = np.linspace(0.0, 1.0, n)
    q_guess = q_start[:,None] + (q_end-q_start)[:,None]*u
    v_guess = np.repeat((q_end-q_start)[:,None], n, axis=1)
    y_guess = np.vstack([q_guess, v_guess])

    def fun(u, y):
        out = np.zeros_like(y)
        out[:2] = y[2:]
        for col in range(y.shape[1]):
            q = y[:2,col]
            v = y[2:,col]
            Gamma = christoffel_second_kind(q, params, metric=metric)
            a = np.zeros(2)
            for i in range(2):
                a[i] = -np.einsum("jk,j,k->", Gamma[i], v, v)
            out[2:,col] = a
        return out

    def bc(ya, yb):
        return np.r_[ya[:2]-q_start, yb[:2]-q_end]

    sol = solve_bvp(fun, bc, u, y_guess, tol=tol, max_nodes=2000)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol

def geodesic_length(sol, params=None, n=1001, metric=None):
    u = np.linspace(0.0,1.0,n)
    y = sol.sol(u)
    speed = np.empty(n)
    for k in range(n):
        q = y[:2,k]
        v = y[2:,k]
        G = inertia_matrix(*q, **params) if metric is None else metric(q)
        speed[k] = np.sqrt(v @ G @ v)
    return np.trapz(speed, u)

def geodesic_residual(sol, params=None, n=101, metric=None):
    """Max absolute geodesic equation residual using the solved state derivative."""
    u = np.linspace(0.0,1.0,n)
    y = sol.sol(u)
    yp = sol.sol(u, 1)
    maxr = 0.0
    for k in range(n):
        q = y[:2,k]; v=y[2:,k]; a=yp[2:,k]
        Gamma = christoffel_second_kind(q, params, metric=metric)
        r = np.array([
            a[i] + np.einsum("jk,j,k->", Gamma[i], v, v)
            for i in range(2)
        ])
        maxr = max(maxr, float(np.max(np.abs(r))))
    return maxr
