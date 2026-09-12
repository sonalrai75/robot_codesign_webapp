import numpy as np
from robot_codesign.dynamics.inertia import inertia_matrix, inertia_matrix_distributed
from robot_codesign.models.planar2 import Planar2DOFRobot
from robot_codesign.paths.planners import GeodesicPlanner
from robot_codesign.analysis.path_performance import path_min_time_control


def test_distributed_uniform_matches_chapter5_exactly():
    l1,l2=0.3,0.4; A1,A2=6e-4,3e-4; rho=2700.; M1=.1; M2=.02
    q1,q2=.4,-.2
    areas1=np.full(10,A1); areas2=np.full(10,A2)
    H0=inertia_matrix(q1,q2,l1,l2,A1,A2,rho,M1,M2)
    Hd=inertia_matrix_distributed(q1,q2,l1,l2,areas1,areas2,rho,M1,M2)
    assert np.allclose(Hd,H0,rtol=1e-13,atol=1e-14)


def test_same_mean_shape_redistribution_changes_inertia_and_travel_time():
    r0=Planar2DOFRobot()
    # Keep each link mean width exactly unchanged, but move material proximally
    # on link 1 and distally on link 2.
    t1=np.array(r0.t1,float); t2=np.array(r0.t2,float)
    ramp=np.linspace(1.25,0.75,10); ramp/=ramp.mean()
    ramp2=np.linspace(0.75,1.25,10); ramp2/=ramp2.mean()
    r1=Planar2DOFRobot(t1=tuple(t1*ramp),t2=tuple(t2*ramp2))
    assert np.isclose(np.mean(r1.t1),np.mean(r0.t1))
    assert np.isclose(np.mean(r1.t2),np.mean(r0.t2))
    q=np.array([.2,.7])
    assert not np.allclose(r1.inertia_matrix(q),r0.inertia_matrix(q),rtol=1e-5,atol=1e-10)

    start=np.array([.4,0.0]); end=np.array([.4,.2])
    p0=GeodesicPlanner().plan(r0,start,end,n=101)
    p1=GeodesicPlanner().plan(r1,start,end,n=101)
    c0=path_min_time_control(r0,p0,[40.,40.])
    c1=path_min_time_control(r1,p1,[40.,40.])
    assert abs(c1['time_s']-c0['time_s']) > 1e-6
