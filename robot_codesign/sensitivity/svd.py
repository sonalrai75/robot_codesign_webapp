from __future__ import annotations
import numpy as np

def central_jacobian(fun, x, rel_step=1e-4):
    """Central finite-difference Jacobian of vector fun(x)."""
    x=np.asarray(x,float)
    y=np.asarray(fun(x),float)
    J=np.empty((y.size,x.size))
    for j in range(x.size):
        h=rel_step*max(1.0,abs(x[j]))
        xp=x.copy(); xm=x.copy()
        xp[j]+=h; xm[j]-=h
        J[:,j]=(np.asarray(fun(xp))-np.asarray(fun(xm)))/(2*h)
    return J

def svd_analysis(J, rcond=None):
    """Return SVD, numerical rank, condition estimate, and pseudoinverse."""
    J=np.asarray(J,float)
    U,s,VT=np.linalg.svd(J,full_matrices=False)
    if rcond is None:
        rcond=max(J.shape)*np.finfo(float).eps
    tol=(s[0]*rcond) if s.size else 0.0
    keep=s>tol
    pinv=(VT.T[:,keep]/s[keep])@U.T[keep,:] if np.any(keep) else np.zeros((J.shape[1],J.shape[0]))
    cond=(s[0]/s[keep][-1]) if np.any(keep) else np.inf
    return {"U":U,"sigma":s,"VT":VT,"rank":int(keep.sum()),"condition":cond,"pinv":pinv}
