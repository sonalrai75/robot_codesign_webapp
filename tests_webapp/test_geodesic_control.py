import numpy as np
from fastapi.testclient import TestClient
from webapp.app import app


def test_geodesic_control_saturates_one_joint_each_interval():
    d=TestClient(app).post('/api/v1/analyze',json={}).json()
    T=d['path']['timing']
    assert T['solver_kind']=='geodesic_bang_bang'
    ratios=np.asarray(T['torque_ratio_mid'],float)
    assert np.allclose(np.max(ratios,axis=1),1.0,rtol=1e-8,atol=1e-8)
    signs=np.asarray(T['active_sign_mid'])
    assert np.any(signs>0) and np.any(signs<0)
    assert 0.0 < T['switch_time_s'] < T['time_s']
