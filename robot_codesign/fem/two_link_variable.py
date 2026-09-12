from __future__ import annotations
import numpy as np
from scipy.linalg import eigvalsh
from .beam import euler_bernoulli_element_matrices

def assemble_two_link_variable_sections(
    l1, l2, thickness1, thickness2, width1, width2,
    E, rho, M_joint2=0.0, M_tip=0.0
):
    """Chapter-5-style two-link FE model with one rectangular section per element.

    `thickness1` and `thickness2` are arrays, one value per beam element.
    The bending inertia used here is I = width*thickness**3/12, corresponding
    to in-plane bending when `width` is perpendicular to the plane of rotation.
    """
    t1=np.asarray(thickness1,dtype=float)
    t2=np.asarray(thickness2,dtype=float)
    if len(t1) != len(t2):
        raise ValueError("Both links must currently use the same element count.")
    n=len(t1)

    # Generalized DOF count:
    # link 1: 2(n+1)
    # link 2 base shares translation but owns rotation, then n new nodes.
    # For n=10 this is 43 DOF, matching the original implementation.
    ndof = 4*n + 3
    M=np.zeros((ndof,ndof))
    K=np.zeros((ndof,ndof))

    def map1(node):
        return [2*node, 2*node+1]

    link1_tip_translation = 2*n
    link2_base_rotation = 2*n + 2
    first_new_link2 = 2*n + 3

    def map2(node):
        if node == 0:
            return [link1_tip_translation, link2_base_rotation]
        return [
            first_new_link2 + 2*(node-1),
            first_new_link2 + 2*(node-1) + 1
        ]

    for L, ts, width, mapper in (
        (l1, t1, width1, map1),
        (l2, t2, width2, map2),
    ):
        Le=L/n
        for e,t in enumerate(ts):
            A=width*t
            I=width*t**3/12.0
            me,ke=euler_bernoulli_element_matrices(E,rho,A,I,Le)
            dofs=mapper(e)+mapper(e+1)
            M[np.ix_(dofs,dofs)] += me
            K[np.ix_(dofs,dofs)] += ke

    M[link1_tip_translation,link1_tip_translation] += M_joint2
    M[map2(n)[0],map2(n)[0]] += M_tip

    # Base translation fixed; base rotation remains free.
    keep=np.array([i for i in range(ndof) if i != 0],dtype=int)
    return M[np.ix_(keep,keep)], K[np.ix_(keep,keep)]

def flexible_frequencies_variable_hz(*args, zero_tol=1e-3, **kwargs):
    M,K=assemble_two_link_variable_sections(*args,**kwargs)
    lam=eigvalsh(K,M)
    lam=np.real(lam)
    lam=lam[lam > zero_tol]
    return np.sqrt(lam)/(2*np.pi)
