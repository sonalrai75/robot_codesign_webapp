from __future__ import annotations
import numpy as np
from scipy.linalg import eigvalsh
from .beam import euler_bernoulli_element_matrices, rectangular_inertia

def assemble_two_link_hinged(
    l1: float, l2: float,
    A1: float, A2: float,
    width1: float, width2: float,
    E: float, rho: float,
    M_joint2: float = 0.0,
    M_tip: float = 0.0,
    elements_per_link: int = 10,
):
    """Assemble the Chapter-5-style collinear two-link flexible model at q1-q2=0.

    Boundary/compatibility interpretation from the thesis:
      * joint 1 translation fixed, rotation about z free;
      * joint 2 translations are compatible between links;
      * link rotations at joint 2 are independent (relative z rotation is free).

    Euler-Bernoulli DOFs are [transverse displacement, section rotation].
    Concentrated M_joint2 is added to the shared joint-2 translational DOF.
    Concentrated M_tip is added to link-2 tip translation.

    Returns reduced (M, K) after fixing the joint-1 translation.
    """
    n = elements_per_link

    # Link 1 has 11 nodes => 22 DOF.
    # Link 2 node 0 shares translation with link1 tip but has its own rotation.
    # Link 2 nodes 1..10 each add 2 DOF: total 43 DOF.
    ndof = 43
    M = np.zeros((ndof, ndof), dtype=float)
    K = np.zeros((ndof, ndof), dtype=float)

    def map1(node):
        return [2*node, 2*node + 1]

    def map2(node):
        if node == 0:
            return [20, 22]  # shared translation, independent rotation
        return [23 + 2*(node-1), 24 + 2*(node-1)]

    for L, A, width, mapper in (
        (l1, A1, width1, map1),
        (l2, A2, width2, map2),
    ):
        thickness, I = rectangular_inertia(A, width)
        Le = L / n
        me, ke = euler_bernoulli_element_matrices(E, rho, A, I, Le)
        for e in range(n):
            dofs = mapper(e) + mapper(e+1)
            M[np.ix_(dofs, dofs)] += me
            K[np.ix_(dofs, dofs)] += ke

    # Thesis concentrated masses.
    M[20, 20] += M_joint2
    tip_translation = map2(n)[0]
    M[tip_translation, tip_translation] += M_tip

    # Joint 1: transverse displacement fixed, rotation free.
    keep = np.array([i for i in range(ndof) if i != 0], dtype=int)
    return M[np.ix_(keep, keep)], K[np.ix_(keep, keep)]


def flexible_frequencies_hz(*args, zero_tol=1e-3, **kwargs):
    """Return positive flexible frequencies, omitting rigid-body mechanisms."""
    M, K = assemble_two_link_hinged(*args, **kwargs)
    lam = eigvalsh(K, M)
    lam = np.real(lam)
    positive = lam[lam > zero_tol]
    return np.sqrt(positive) / (2*np.pi)
