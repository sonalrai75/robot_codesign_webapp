import numpy as np
from robot_codesign.fem.two_link_variable import assemble_two_link_variable_sections

def test_variable_section_assembly_is_symmetric():
    t1=np.linspace(.01,.02,10)
    t2=np.linspace(.02,.01,10)
    M,K=assemble_two_link_variable_sections(
        .1419,.4581,t1,t2,.01,.038,2.7e10,2700.0,
        M_joint2=.1,M_tip=0.0
    )
    assert np.allclose(M,M.T)
    assert np.allclose(K,K.T)
    assert M.shape == K.shape == (42,42)
