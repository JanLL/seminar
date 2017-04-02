import numpy as np
import matplotlib.pyplot as plt
import copy
import time

import opengm

#para = opengm.InfParam(numberOfThreads=4)	# will er nicht benutzen, keine Ahnung warum...


imgL = plt.imread("imgL.png")[300:500, 400:600]
imgR = plt.imread("imgR.png")[300:500, 400:600]
disparity = np.load("disparity.npy")[300:500, 400:600]



dimx = disparity.shape[1]
dimy = disparity.shape[0]
numLabels = disparity.shape[2]
numVar = dimx*dimy

numberOfStates = np.ones(numVar, dtype=opengm.index_type)*numLabels
gm = opengm.graphicalModel(numberOfStates, operator='adder')


varIndex = lambda x, y: x*dimy + y

#################################### unary factors ###############################

for y in range(dimy):
    for x in range(dimx):
        fid=gm.addFunction(disparity[y,x,:]*100)
        gm.addFactor(fid, varIndex(x,y))

print ('GM mit unaries fertig aufgebaut. Beginne zu rechnen...')

solver = opengm.inference.AlphaExpansion(gm)
solver.infer()
argmin=solver.arg()
unaryOnly = argmin.reshape(dimx,dimy).transpose()
        
#################################### Smoothing Matrix ##################################
# 0 auf Diagonale, C1 auf Hauptnebendiagonale, Rest C2
C1 = 0.5
C2 = 2.0
f = np.ones([numLabels, numLabels], dtype=np.float32) * C2
for l in range(numLabels):
    f[l,l] = 0
    if (l-1 >= 0):
        f[l, l-1] = C1
    if (l+1 < numLabels):
        f[l, l+1] = C1


fid = gm.addFunction(f)

for y in range(dimy):
    for x in range(dimx):
        if (x+1 < dimx):
            gm.addFactor(fid, [varIndex(x,y), varIndex(x+1,y)])
        if (y+1 < dimy):
            gm.addFactor(fid, [varIndex(x,y), varIndex(x,y+1)])
            
print ('GM mit smoothness factors fertig aufgebaut. Beginne zu rechnen...')

t_start = time.time()

solver = opengm.inference.AlphaExpansion(gm)
solver.infer()
argmin=solver.arg()
print "Solving time: %f s" % (time.time()-t_start)
res = argmin.reshape(dimx,dimy).transpose()

f, ax = plt.subplots(2,2,figsize=(14,14))
ax[0,0].imshow(imgL)
ax[0,1].imshow(imgR)
ax[1,0].imshow(unaryOnly, vmin = 0, vmax = numLabels)
ax[1,1].imshow(res, vmin = 0, vmax = numLabels)

plt.savefig('figure.png')

plt.show()

