import numpy as np
import sys
import pickle
from rc_fun import *
import os
import time

#pList = np.linspace(0,0.4,21).round(3)
#gammaList = np.linspace(0,5e-4,21).round(6)
p_name = sys.argv[1]
p = float(p_name)
r_name=sys.argv[2]
r = float(r_name)

# Which jump operator: 0 - S_z; 1 - S_- ALWAYS THE LATTER
jumpOpList = ['S_Z', 'S_minus']
name = jumpOpList[1]
jumpOp_global = globals()[name]

# The dephasing/relaxation rate: The range of old values: rateList = np.logspace(-5,1,7)

dt = 10

# System size:
Nqubit = 16

################################################################################################################################################################
t=500

# Take data every 10 steps 
qab_X = np.zeros([t,2])
qab_Z = np.zeros([t,2])
entropy= np.zeros(t)
minEntropy = np.zeros(t)
maxEntropy = np.zeros(t)
purity = np.zeros(t)
mutual_info = np.zeros(t)
secRenyiS = np.zeros(t)
secRenyiMI = np.zeros(t)

## 
path0 = '/home1/yl244/data/RC_state_follow/' + name
os.makedirs(path0,exist_ok=True)

path1 = '/home1/yl244/data/RC_state_follow/' + name + '/N=' + str(Nqubit)
os.makedirs(path1,exist_ok=True)

homepath = '/home1/yl244/data/RC_state_follow/' + name + '/N=' + str(Nqubit)+ '/r=' + r_name +  'p=' + p_name
os.makedirs(homepath,exist_ok=True)

statepath = homepath+'/states'

initialIndex = np.random.randint(0,9)

if os.path.isfile(homepath+'/state/ss'+str(initialIndex)+'.out'):
    # Record data; time average is ensemble average
    with open(statepath + '/ss' + str(initialIndex) + '.out', 'rb') as f:
        rho = f
        
    K1 = mkSingleSiteJumpKraus(r, dt, S_minus)
    
    for i in range(t):        
        startTime = time.time()

        qab_X[i,:] = np.real(np.array(calcGlassX(rho,Nqubit)))
        qab_Z[i,:] = np.real(np.array(calcGlassZ(rho,Nqubit)))
        entropy[i], secRenyiS[i], minEntropy[i], maxEntropy[i], purity[i], mutual_info[i], secRenyiMI[i] = calcS(rho, Nqubit, [n for n in range(Nqubit//2)])   

        rho = evolveRandU(rho, Nqubit,i)
        if r>0:
            rho = jump(rho, Nqubit, K1)
        if probMsm>0:    
            rho = mkMsm(rho, Nqubit, p)
                            
        print('evo takes', time.time()-startTime, flush=True)
                
    data = np.loadtxt(homepath + '/entropy_' + str(fileInd) + '.out')
    entropy = np.concatenate([data, entropy])
    np.savetxt(homepath + '/entropy_' + str(fileInd) + '.out', entropy)

    data = np.loadtxt(homepath + '/I_' + str(fileInd) + '.out')
    mutual_info = np.concatenate([data, mutual_info])    
    np.savetxt(homepath + '/I_' + str(fileInd) + '.out', mutual_info)
    
    data = np.loadtxt(homepath + '/purity_' + str(fileInd) + '.out')
    purity = np.concatenate([data, purity])   
    np.savetxt(homepath + '/purity_' + str(fileInd) + '.out', purity)

    data = np.loadtxt(homepath + '/minEntropy_' + str(fileInd) + '.out')   
    minEntropy = np.concatenate([data, minEntropy])   
    np.savetxt(homepath + '/minEntropy_' + str(fileInd) + '.out', minEntropy)
    
    data = np.loadtxt(homepath + '/maxEntropy_' + str(fileInd) + '.out')   
    maxEntropy = np.concatenate([data, maxEntropy])       
    np.savetxt(homepath + '/maxEntropy_' + str(fileInd) + '.out', maxEntropy)

    data = np.loadtxt(homepath + '/qab_X' + str(fileInd) + '.out')   
    qab_X = np.concatenate([data, qab_X]) 
    np.savetxt(homepath + '/qab_X_' + str(fileInd) + '.out', qab_X)

    data = np.loadtxt(homepath + '/qab_Z' + str(fileInd) + '.out')   
    qab_Z = np.concatenate([data, qab_Z]) 
    np.savetxt(homepath + '/qab_Z_' + str(fileInd) + '.out', qab_Z)
  
else:
    
    # reinitialize the state
    state0 = (qb[0] + qb[1])/np.sqrt(2)
    state0rho = np.outer(state0, state0)
    print('start fresh fixed state')

    rho = state0rho
    for n in range(Nqubit-1):
        rho = np.tensordot(rho,state0rho, axes=0)
        

    # when one needs to evolve the state, one needs to 
    # 1. start evovling up to either 100 steps or 1/r to reach the stready state
    # 3. keep evolving and record 100(?) data points to average the steady state value
    # 2. save the states
    # 4. save all the data
    # 5. fucking done with this project

    # make jump operators:
    K1 = mkSingleSiteJumpKraus(r, dt, S_minus)
    
    print( (r>0))
    if (p<0.1) & (r>0):
        t_eq = int(1/(r*dt))    
    else: 
        t_eq = 150
        
    startTimeEq = time.time()
    for i in range(t_eq):
        rho = evolveRandU(rho, Nqubit,i)

        if r>0:
            rho = jump(rho, Nqubit, K1)

        if p>0:
            rho = mkMsm(rho, Nqubit, p)
            
    print('time taken to equilibriate r=', str(r), 'p=', str(p), time.time() - startTimeEq)

    # The state has equilibriated; now record 100 data points as before

    path0 = '/home1/yl244/data/RC_state_follow/' + name
    os.makedirs(path0,exist_ok=True)

    path1 = '/home1/yl244/data/RC_state_follow/' + name + '/N=' + str(Nqubit)
    os.makedirs(path1,exist_ok=True)

    homepath = '/home1/yl244/data/RC_state_follow/' + name + '/N=' + str(Nqubit)+ '/r=' + r_name +  'p=' + p_name
    os.makedirs(homepath,exist_ok=True)

    
    startTimeCd = time.time()
    for i in range(t):
            
        glassStartTime = time.time()
        qab_X[i,:] = np.real(np.array(calcGlassX(rho,Nqubit)))
        qab_Z[i,:] = np.real(np.array(calcGlassZ(rho,Nqubit)))
        print('glass time taken', time.time() - glassStartTime)

        entropy[i], secRenyiS[i], minEntropy[i], maxEntropy[i], purity[i], mutual_info[i], secRenyiMI[i] = calcS(rho, Nqubit, [n for n in range(Nqubit//2)])  
        
        rho = evolveRandU(rho, Nqubit,i)
        if r>0:
            rho = jump(rho, Nqubit, K1)
        if p>0:    
            rho = mkMsm(rho, Nqubit, p)

        # save the last 1 state:
        if i>t-1:
            statepath = homepath+'/states'
            os.makedirs(statepath,exist_ok=True)
            fileInd = len(os.listdir(statepath))
            with open(statepath + '/ss' + str(fileInd) + '.out', 'wb') as f:
                np.save(f, rho)
                
    fileInd = len(os.listdir(homepath))//9
    print(fileInd, entropy)
    # Just save the trajectory data immediately after making it
    np.savetxt(homepath + '/entropy_' + str(fileInd) + '.out', entropy)
    np.savetxt(homepath + '/I_' + str(fileInd) + '.out', mutual_info)
    np.savetxt(homepath + '/purity_' + str(fileInd) + '.out', purity)
    np.savetxt(homepath + '/minEntropy_' + str(fileInd) + '.out', minEntropy)
    np.savetxt(homepath + '/maxEntropy_' + str(fileInd) + '.out', maxEntropy)
    
    np.savetxt(homepath + '/SRenyi2_' + str(fileInd) + '.out', secRenyiS)
    np.savetxt(homepath + '/MIRenyi2_' + str(fileInd) + '.out', secRenyiMI)
    
    np.savetxt(homepath + '/qab_X_' + str(fileInd) + '.out', qab_X)
    np.savetxt(homepath + '/qab_Z_' + str(fileInd) + '.out', qab_Z)

    print('time taken to collect data ', time.time() - startTimeCd)

    