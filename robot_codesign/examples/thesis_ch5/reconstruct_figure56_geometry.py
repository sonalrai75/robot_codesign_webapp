import numpy as np

# Approximate digitization at the ten element midpoints, retained from the
# Figure 5-6 check. Coordinates are read from the plotted planar outline.
TOP1=np.array([.00928,.01083,.01250,.01449,.01671,.01900,.02136,.02391,.02591,.02735])
BOT1=np.array([-.01245,-.01389,-.01533,-.01721,-.01899,-.02132,-.02298,-.02531,-.02719,-.02874])
TOP2=np.array([.02502,.02325,.02059,.01990,.01759,.01482,.01272,.01100,.00995,.00817])
BOT2=np.array([-.02630,-.02420,-.02176,-.01980,-.01799,-.01489,-.01212,-.01040,-.00890,-.00768])

def summarize(name,top,bot):
    h=top-bot
    yc=0.5*(top+bot)
    print(name)
    print("  visible height:", np.round(h,6))
    print("  centerline y: ", np.round(yc,6))
    print(f"  height range: {h.min():.6f} .. {h.max():.6f} m")
    print(f"  centerline range: {yc.min():.6f} .. {yc.max():.6f} m")
    print(f"  max |centerline offset|: {np.max(np.abs(yc)):.6f} m")
    return h,yc

h1,c1=summarize("Link 1",TOP1,BOT1)
h2,c2=summarize("Link 2",TOP2,BOT2)

print()
print("Interpretation:")
print("The scan contains two independent planar boundary coordinates per station.")
print("Their midpoint is not identically zero, so reducing the plot immediately")
print("to a single straight-centerline thickness variable discards geometry.")
print("This script does NOT yet claim the plotted ordinate is metrically equal")
print("to the FE section thickness; that mapping remains to be established.")
