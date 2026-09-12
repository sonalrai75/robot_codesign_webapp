import numpy as np
from robot_codesign.fem.frame2d import polyline_frame_frequencies

# center offsets from Figure 5-6 digitization at element centers
c1=np.array([-.001585,-.001530,-.001415,-.001360,-.001140,-.001160,-.000810,-.000700,-.000640,-.000695])
c2=np.array([-.000640,-.000475,-.000585,.000050,-.000200,-.000035,.000300,.000300,.000525,.000245])
h1=np.array([.02173,.02472,.02783,.03170,.03570,.04032,.04434,.04922,.05310,.05609])
h2=np.array([.05132,.04745,.04235,.03970,.03558,.02971,.02484,.02140,.01885,.01585])

def midpoint_to_nodes(c,L):
    # stable approximate inversion of midpoint values to 11 nodal values,
    # with zero-mean regularization; used only as a curvature diagnostic.
    n=len(c)
    A=np.zeros((n,n+1))
    for i in range(n): A[i,i:i+2]=.5
    D=np.zeros((n-1,n+1))
    for i in range(n-1): D[i,i:i+3]=[1,-2,1]
    AA=np.vstack([A,0.05*D])
    bb=np.r_[c,np.zeros(n-1)]
    return np.linalg.lstsq(AA,bb,rcond=None)[0]

for name,L,c,h,w in [("link1",.1419,c1,h1,.010),("link2",.4581,c2,h2,.038)]:
    y=midpoint_to_nodes(c,L); x=np.linspace(0,L,11)
    curved=np.c_[x,y]; straight=np.c_[x,np.zeros_like(x)]
    fc=polyline_frame_frequencies(curved,h,w,2.7e10,2700.)
    fs=polyline_frame_frequencies(straight,h,w,2.7e10,2700.)
    seg=np.diff(curved,axis=0)
    ang=np.degrees(np.arctan2(seg[:,1],seg[:,0]))
    print(name)
    print("  segment-angle range (deg):",ang.min(),ang.max())
    print("  first two standalone frame frequencies, straight:",fs[:2])
    print("  first two standalone frame frequencies, curved: ",fc[:2])
    print("  ratio curved/straight:",fc[:2]/fs[:2])

print("\nThis is a geometry-sensitivity diagnostic only. It shows whether the")
print("measured centerline curvature is large enough to matter in a 2-D frame")
print("formulation; it is not yet the Chapter-5 two-link joint model.")
