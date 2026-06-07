import numpy as np
import sys
import pickle
from rcFun import *
import

######################################################
### the variables that need to be put in by hand ####

# The probability of measurement
p_name = sys.argv[1]
p = float(p_name)
# Recommended values: probList = np.linspace(0,1,21)
#[0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

# Which jump operator: 0 - S_z; 1 - S_-
jumpOpList = ['S_Z', 'S_minus']
name = jumpOpList[int(sys.argv[2])]
jumpOp_global = globals()[name]

# The dephasing/relaxation rate: The range of old values: rateList = np.logspace(-5,1,7)
r_name=sys.argv[3]
r = float(r_name)

# System size:
N = int(sys.argv[4])
# Old values: NList = [4, 6, 8, 10]; can try 12+ but must be even

# The number of repetition (trajectories?); needs tweeking if want to save files..
ens_N = int(sys.argv[5])

######################################################
# Other params:
#t_total = max(1.2/r, 200)
# Tried to run at least 1.2 times the relaxation time but for small Gamma this goes on forever
t_total = 400
dt=5
steps_N = int(t_total/dt)


qb={}
qb[0]=np.array([1,0],dtype='complex128')
qb[1]=np.array([0,1],dtype='complex128')

# Pure state start:
state0 = (qb[0] + qb[1])/np.sqrt(2)
state0rho = np.outer(state0, state0)

# Mixed state start:
state1rho = (np.outer(qb[0], qb[0]) + np.outer(qb[1], qb[1]))/2

######################################################
############ Start of the code #######################
# Let's make independent files for different rates
# Save the individual dynamics for large relaxation rates

S1List = np.zeros([ens_N,steps_N])
S2List = np.zeros([ens_N,steps_N])
StotList = np.zeros([ens_N,steps_N])
mutualInfoList = np.zeros([ens_N,steps_N])
purityList = np.zeros([ens_N,steps_N])
negativityList = np.zeros([ens_N,steps_N])

print('jumpOp =', name, ',Nqubit=', N,  ', p =', p,', rate = ', r, ', total steps=', steps_N )
for j in range(ens_N):
    startTime = time.time()
    S1List[j,:], S2List[j,:], StotList[j,:], mutualInfoList[j,:], purityList[j,:], negativityList[j,:] = combinedRoutine(initRho = state0rho,
                                Nqubit = N,
                                rate = r,
                                probMsm = p,
                                dt = dt ,
                                jumpOp = jumpOp_global,
                                steps = steps_N,
                                debug = False)
    print('run time for trajectory ', str(j), ' =', time.time()-startTime, flush=True)

homepath='RC_raw/'

np.savetxt(homepath + 'S1_' + name + '_p=' + p_name + '_r=' + r_name + '_N=' + str(N) + '.out', S1List)

np.savetxt(homepath + 'S2_' + name + '_p=' + p_name + '_r=' + r_name + '_N=' + str(N)  + '.out', S2List)

np.savetxt(homepath + 'Stot_' + name + '_p=' + p_name + '_r=' + r_name + '_N=' + str(N)  + '.out', StotList)

np.savetxt(homepath + 'I_' + name + '_p=' + p_name + '_r=' + r_name + '_N=' + str(N) + '.out', mutualInfoList)

np.savetxt(homepath + 'purity_' + name + '_p=' + p_name + '_r=' + r_name + '_N=' + str(N) + '.out', purityList)

np.savetxt(homepath + 'negativity' + name + '_p=' + p_name + '_r=' + r_name + '_N=' + str(N) + '.out', negativity)

