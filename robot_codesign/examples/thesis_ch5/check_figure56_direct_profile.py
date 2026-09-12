import numpy as np
from robot_codesign.fem.two_link_variable import flexible_frequencies_variable_hz

# Approximate direct digitization of the upper/lower outlines in Figure 5-6,
# read at the 10 element midpoints of each link.
#
# IMPORTANT: this is intentionally a hypothesis test.  The thesis does not state
# that the plotted vertical ordinate is literally the rectangular-section thickness.
TOP1=np.array([.00928,.01083,.01250,.01449,.01671,.01900,.02136,.02391,.02591,.02735])
BOT1=np.array([-.01245,-.01389,-.01533,-.01721,-.01899,-.02132,-.02298,-.02531,-.02719,-.02874])
TOP2=np.array([.02502,.02325,.02059,.01990,.01759,.01482,.01272,.01100,.00995,.00817])
BOT2=np.array([-.02630,-.02420,-.02176,-.01980,-.01799,-.01489,-.01212,-.01040,-.00890,-.00768])

T1=TOP1-BOT1
T2=TOP2-BOT2

l1=0.1419
l2=0.4581
w1=0.010
w2=0.038
rho=2700.0
E=2.7e10
M1=0.09966
M2=0.000122

dz1=l1/10
dz2=l2/10
z1=(np.arange(10)+0.5)*dz1
z2=(np.arange(10)+0.5)*dz2
A1=w1*T1
A2=w2*T2

m_link1=rho*np.sum(A1*dz1)
m_link2=rho*np.sum(A2*dz2)

I1=rho*np.sum(A1*z1**2*dz1)
I2=rho*np.sum(A2*z2**2*dz2)
first_moment2=rho*np.sum(A2*z2*dz2)

H11=I1+l1**2*(M1+M2+m_link2)
H22=I2+M2*l2**2
H12=l1*first_moment2+M2*l1*l2
H=np.array([[H11,H12],[H12,H22]])
evals=np.linalg.eigvalsh(H)

freq=flexible_frequencies_variable_hz(
    l1,l2,T1,T2,w1,w2,E,rho,
    M_joint2=M1,M_tip=M2
)

print("Direct Figure 5-6 ordinate-as-thickness hypothesis")
print("------------------------------------------------")
print("Link 1 digitized thicknesses (m):", np.round(T1,6))
print("Link 2 digitized thicknesses (m):", np.round(T2,6))
print()
print(f"Predicted link masses: {m_link1:.6f} kg, {m_link2:.6f} kg")
print("Predicted H(q1-q2=0):")
print(H)
print("Predicted inertia eigenvalues:", evals)
print("Thesis final inertia eigenvalues: [0.0066, 0.0281]")
print()
print("Predicted first two flexible frequencies (Hz):", freq[:2])
print("Thesis final flexible frequencies (Hz): [72.0, 279.3]")
print()
print("CONCLUSION: the plotted ordinate cannot safely be treated as literal")
print("rectangular-section thickness under the stated 1 cm / 3.8 cm widths.")
print("The thesis does not provide enough information to infer the missing")
print("plot-to-section mapping directly from Figure 5-6 alone.")
