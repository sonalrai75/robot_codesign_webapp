import numpy as np
from robot_codesign.fem.two_link import flexible_frequencies_hz

def test_concentrated_masses_change_frequencies():
    common = dict(
        l1=0.3, l2=0.3, A1=0.0006, A2=0.0006,
        width1=0.09438683, width2=0.06335978,
        E=2.7e10, rho=2700.0, elements_per_link=10
    )
    f_no = flexible_frequencies_hz(**common, M_joint2=0.0, M_tip=0.0)[:2]
    f_mass = flexible_frequencies_hz(**common, M_joint2=0.09966, M_tip=0.000122)[:2]
    assert np.max(np.abs(f_no-f_mass)) > 0.1

def test_nominal_identified_sections_reproduce_targets():
    f = flexible_frequencies_hz(
        l1=0.3, l2=0.3, A1=0.0006, A2=0.0006,
        width1=0.09438683, width2=0.06335978,
        E=2.7e10, rho=2700.0,
        M_joint2=0.09966, M_tip=0.000122,
        elements_per_link=10
    )[:2]
    assert np.allclose(f, [116.3, 268.2], atol=0.02)
