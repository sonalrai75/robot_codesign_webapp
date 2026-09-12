from __future__ import annotations
import numpy as np


def inertia_matrix(
    q1: float,
    q2: float,
    l1: float,
    l2: float,
    A1: float,
    A2: float,
    rho: float,
    M1: float,
    M2: float,
) -> np.ndarray:
    """Chapter 5 two-link rigid-body inertia matrix for uniform link areas.

    q1 and q2 are absolute joint angles measured from the fixed reference line.
    M1 is the concentrated mass at joint 2; M2 is the tip payload.
    """
    H11 = rho * A1 * l1**3 / 3.0 + l1**2 * (M1 + M2 + rho * A2 * l2)
    H22 = rho * A2 * l2**3 / 3.0 + M2 * l2**2
    H12 = (
        rho * A2 * l1 * l2**2 / 2.0
        + M2 * l1 * l2
    ) * np.cos(q1 - q2)

    return np.array([[H11, H12], [H12, H22]], dtype=float)


def _piecewise_mass_moments(length: float, areas, rho: float):
    """Exact mass, first moment, and second moment for equal-length sections.

    Areas are piecewise constant over equally spaced elements measured from the
    proximal joint.  Returns (int dm, int r dm, int r^2 dm).
    """
    a = np.asarray(areas, dtype=float)
    if a.ndim != 1 or len(a) == 0:
        raise ValueError("areas must be a non-empty 1-D sequence")
    if np.any(a <= 0):
        raise ValueError("all section areas must be positive")
    edges = np.linspace(0.0, float(length), len(a) + 1)
    lo, hi = edges[:-1], edges[1:]
    m0 = rho * np.sum(a * (hi - lo))
    m1 = rho * np.sum(a * (hi**2 - lo**2) / 2.0)
    m2 = rho * np.sum(a * (hi**3 - lo**3) / 3.0)
    return float(m0), float(m1), float(m2)


def inertia_matrix_distributed(
    q1: float,
    q2: float,
    l1: float,
    l2: float,
    areas1,
    areas2,
    rho: float,
    M1: float,
    M2: float,
) -> np.ndarray:
    """Rigid-body inertia using the full piecewise mass distribution.

    This is the exact counterpart of the Chapter-5 uniform-area equations for
    piecewise-constant equal-length sections.  Redistribution of material within
    either link therefore changes the rigid inertia even when total link mass is
    unchanged.
    """
    m10, _m11, m12 = _piecewise_mass_moments(l1, areas1, rho)
    m20, m21, m22 = _piecewise_mass_moments(l2, areas2, rho)

    c = np.cos(q1 - q2)
    H11 = m12 + l1**2 * (M1 + M2 + m20)
    H22 = m22 + M2 * l2**2
    H12 = l1 * (m21 + M2 * l2) * c
    return np.array([[H11, H12], [H12, H22]], dtype=float)


def inertia_eigenvalues(*args, **kwargs) -> np.ndarray:
    """Return eigenvalues in ascending order for the legacy uniform model."""
    return np.linalg.eigvalsh(inertia_matrix(*args, **kwargs))
