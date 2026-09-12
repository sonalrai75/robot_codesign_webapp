from __future__ import annotations
import numpy as np

def euler_bernoulli_element_matrices(E: float, rho: float, A: float, I: float, L: float):
    """Consistent 2D Euler-Bernoulli bending element matrices.

    DOFs: [w1, theta1, w2, theta2].

    This is the initial transparent FE building block. Chapter 5 states 10 beam
    elements per link with rectangular sections. The exact nominal width/thickness
    pair is not explicit in the thesis, so section identification is kept separate.
    """
    k = (E * I / L**3) * np.array([
        [12, 6*L, -12, 6*L],
        [6*L, 4*L**2, -6*L, 2*L**2],
        [-12, -6*L, 12, -6*L],
        [6*L, 2*L**2, -6*L, 4*L**2],
    ], dtype=float)

    m = (rho * A * L / 420.0) * np.array([
        [156, 22*L, 54, -13*L],
        [22*L, 4*L**2, 13*L, -3*L**2],
        [54, 13*L, 156, -22*L],
        [-13*L, -3*L**2, -22*L, 4*L**2],
    ], dtype=float)
    return m, k

def rectangular_inertia(area: float, width: float) -> tuple[float, float]:
    """Given rectangular area and width, return (thickness, I=width*thickness^3/12)."""
    thickness = area / width
    I = width * thickness**3 / 12.0
    return thickness, I
