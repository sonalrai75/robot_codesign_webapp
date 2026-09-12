import numpy as np
from robot_codesign.sensitivity.svd import central_jacobian, svd_analysis

def test_central_jacobian_linear():
    A=np.array([[1.,2.],[-3.,4.]])
    J=central_jacobian(lambda x:A@x,np.array([.3,-.4]))
    assert np.allclose(J,A,rtol=1e-8,atol=1e-8)

def test_svd_pseudoinverse_reconstruction():
    J=np.array([[1.,2.,0.],[0.,1.,1.]])
    a=svd_analysis(J)
    assert a["rank"]==2
    assert np.allclose(J@a["pinv"]@J,J)
