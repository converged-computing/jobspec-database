import numpy as np
import re
import sys
import pylab
from mpl_toolkits.mplot3d import Axes3D
import os


streamFileName = sys.argv[1]
markerSize = 0.5 #2
if len(sys.argv) >= 3:
    markerSize = float(sys.argv[2])

f = open(streamFileName, 'r')
stream = f.read()
f.close()

output_path = os.path.dirname(os.path.abspath(streamFileName))  + '/plots_res'
if not os.path.exists(output_path):
    print(20)
    os.mkdir(output_path)

print(output_path)
colors = ["r", "g", "b"]
xStarNames = ["astar","bstar","cstar"]
for i in np.arange(3):
    p = re.compile(xStarNames[i] + " = ([\+\-\d\.]* [\+\-\d\.]* [\+\-\d\.]*)")
    xStarStrings = p.findall(stream)

    xStars = np.zeros((3, len(xStarStrings)), float)

    for j in np.arange(len(xStarStrings)):
        xStars[:,j] = np.array([float(s) for s in xStarStrings[j].split(' ')])
    pylab.clf()

    fig = pylab.figure()
    ax = Axes3D(fig)
    ax.scatter(xStars[0,:],xStars[1,:],xStars[2,:], marker=".", color=colors[i], s=markerSize)
    pylab.title(xStarNames[i] + "s")

    out = os.path.join(output_path, os.path.basename(streamFileName).split('.')[0]+ "_" + xStarNames[i])+'.png'
    pylab.savefig(out)
    print(out)


pylab.close()



