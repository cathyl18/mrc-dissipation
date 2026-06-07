import numpy as np
import sys
import pickle
from rc_fun import *
import os

p_name = sys.argv[1]
probMsm = float(p_name)

# Which jump operator: 0 - S_z; 1 - S_-
jumpOpList = ['S_Z', 'S_minus']
name = jumpOpList[int(sys.argv[2])]
jumpOp_global = globals()[name]

# The dephasing/relaxation rate: The range of old values: rateList = np.logspace(-5,1,7)
r_name=sys.argv[3]
r = float(r_name)
rate = r
dt = .5

# System size:
Nqubit = int(sys.argv[4]) # NList = [12, 14, 16, 18]

# The number of trajectories; needs tweeking if want to save files..
ens_N = int(sys.argv[5])

####################################################################################
# initialize states:
state0 = (qb[0] + qb[1])/np.sqrt(2)
state0rho = np.outer(state0, state0)

tracemalloc.start()

# make jump operators:
K1 = mkSingleSiteJumpKraus(rate, dt, S_minus)

print('jumpOp =', name, ',Nqubit=', Nqubit, ', rate = ', r, ', p =', p_name )

# path0 = 'data/RC_eigs/' + name
# os.makedirs(path0,exist_ok=True)

# path1 = 'data/RC_eigs/' + name + '/N=' + str(Nqubit)
# os.makedirs(path1,exist_ok=True)

# homepath = 'data/RC_eigs/' + name + '/N=' + str(Nqubit)+ '/r=' + r_name +  'p=' + p_name
# os.makedirs(homepath,exist_ok=True)

path0 = 'data/RC_TN_dyn/' + name
os.makedirs(path0,exist_ok=True)

path1 = 'data/RC_TN_dyn/' + name + '/N=' + str(Nqubit)
os.makedirs(path1,exist_ok=True)

homepath = 'data/RC_TN_dyn/' + name + '/N=' + str(Nqubit)+ '/r=' + r_name +  'p=' + p_name
os.makedirs(homepath,exist_ok=True)


for trajInd in range(ens_N):

    rho = state0rho
    for n in range(Nqubit-1):
        rho = np.tensordot(rho,state0rho, axes=0)

    startTime0 = time.time()

    print('memory costs in unitary', str(tracemalloc.get_traced_memory()[0]/1024**3),flush=True)

    tracemalloc.start()

    t = 200
    purity = np.zeros(t)
    entropy = np.zeros(t)
    minEntropy = np.zeros(t)
    maxEntropy = np.zeros(t)
    mutual_info = np.zeros(t)
    neg = np.zeros(t)



    for i in range(t):
        startTime1 = time.time()
        
        ##S1, Smin, Smax, purity, Iab
        entropy[i], minEntropy[i], maxEntropy[i], purity[i], mutual_info[i] = calcS(rho, Nqubit, [n for n in range(Nqubit//2)])

        if Nqubit>10:
            print('entropies take', time.time()-startTime1,flush=True)

        print('memory costs in unitary', str(tracemalloc.get_traced_memory()[0]/1024**3),flush=True)

        startTime1 = time.time()
        neg[i] = calcNeg(rho, Nqubit, [n for n in range(Nqubit//2)])
        if Nqubit >10:
            print('negativities take', time.time()-startTime1, flush=True)

            
        rho = evolveRandU(rho, Nqubit,i)
        if rate>0:
            rho = jump(rho, Nqubit, K1)

        if probMsm>0:
            rho = mkMsm(rho, Nqubit, probMsm)

    fileInd = int(np.floor(len(os.listdir(homepath))/6))
    if Nqubit>12:
        np.savetxt(homepath + '/ss'+ str(fileInd)+ '.out', rho.reshape([2**Nqubit,2**Nqubit]))

    # Just save the trajectory data immediately after making it
    np.savetxt(homepath + '/entropy_' + str(fileInd) + '.out', entropy)
    np.savetxt(homepath + '/I_' + str(fileInd) + '.out', mutual_info)
    np.savetxt(homepath + '/purity_' + str(fileInd) + '.out', purity)
    np.savetxt(homepath + '/negativity_' + str(fileInd) + '.out', neg)
    np.savetxt(homepath + '/minEntropy_' + str(fileInd) + '.out', minEntropy)
    np.savetxt(homepath + '/maxEntropy_' + str(fileInd) + '.out', maxEntropy)


#     with open(homepath + '/eig1prof_' + str(fileInd) + '.out','wb') as file:
#         pickle.dump(eig1prof, file)
#     with open(homepath + '/eig2prof_' + str(fileInd) + '.out','wb') as file:
#         pickle.dump(eig2prof, file)
#     with open(homepath + '/eigtotprof_' + str(fileInd) + '.out','wb') as file:
#         pickle.dump(eigtotprof, file)
#     with open(homepath + '/eigpartialT_' + str(fileInd) + '.out','wb') as file:
#         pickle.dump(eigpartialT, file)

    if Nqubit >10:
        # Save the state
        
        print('run time for trajectory ', str(trajInd), ' =', time.time()-startTime0, flush=True)


#  190 s for N=10 wo msm in 1 ensemble
