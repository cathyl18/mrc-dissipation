import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from scipy.sparse import spdiags,linalg,eye
##  BLOCK OF FUNCTIONS USED IN THE MAIN CODE
#----------------------------------------------------------------------

def initialstate(N):   
    ''' 
    Generates a random spin configuration for initial condition
    '''
    state = 2*np.random.randint(2, size=(N,N))-1
    return state


def mcmove(config, Jd, Jh, J123):
    '''
    Monte Carlo move using Metropolis algorithm 
    '''
    N = len(config)
    for i in range(N):
        for j in range(N):
                # try to flip the spin at random site (a,b)
                a = np.random.randint(0, N)
                b = np.random.randint(0, N) 
                s =  config[a, b]
                
                # cost function is different :
                # bulk interaction (6 interactions)

                pairnb = (Jd * (config[(a+1)%N,(b-1)%N] + config[(a-1)%N,b] + config[(a+1)%N,b] +config[(a-1)%N,(b+1)%N]) + \
                              + Jh * (config[a,(b-1)%N] + config[a,(b+1)%N]) )
                trinb = J123 * (config[a,(b-1)%N] * config[(a-1)%N,b%N] + config[(a-1)%N,(b+1)%N] * config[a%N,(b+1)%N] + config[(a+1)%N,b] * config[(a+1)%N,(b-1)%N])


                cost = 2*s*(pairnb + trinb)
                
                # move if cost<0 and if cost>0 and with probability < Boltzmann weight
                if cost < 0:
                    s *= -1
                elif rand() < np.exp(-cost):
                    s *= -1
                config[a, b] = s
    return config



def calcEnergy(config, Jd, Jh, J123):
    '''
    Energy of a given configuration
    '''
    energy = 0 
    
    for a in range(len(config)):
        for b in range(len(config)):
            S = config[a,b]
            
            pairnb = (Jd * (config[a%N,(b-1)%N] + config[(a+1)%N,(b-1)%N] + config[a%N,(b+1)%N] +config[(a-1)%N,(b+1)%N]) + \
                          + Jh * (config[(a-1)%N,b] + config[(a+1)%N,b]) )
            trinb = J123 * (config[a,(b-1)%N] * config[(a-1)%N,b%N] + config[(a-1)%N,(b+1)%N] * config[a%N,(b+1)%N] + config[(a+1)%N,b] * config[(a+1)%N,(b-1)%N])
                 
            energy += -S * (pairnb + trinb)
    return energy/2.  # to compensate for over-counting



def calcMag(config):
    '''
    Magnetization of a given configuration
    '''
    mag = np.sum(config)
    return mag

# The set of parameter space from p and Gamma
def calcJs(p,g,q):
    Jh = (np.log(-((q**2 - 2*p*q**2 + 2*p**2*((q*(1 + q))/2. + g**2))**2/(q**2*(-1 + q**4))) + \
          (q - 2*p*q + p**2*(2*q - 2*(-1 + q)*(g - g**2)))**2/(-1 + q**4))/8. + \
       np.log((q**2 - 2*p*q**2 + 2*p**2*((q*(1 + q))/2. + g**2))**2/(-1 + q**4) - \
          (q - 2*p*q + p**2*(2*q - 2*(-1 + q)*(g - g**2)))**2/(q**2*(-1 + q**4)))/8. - \
       np.log(-(((1 - 2*p + 2*p**2)*(q + 2*g**2)*(q**2 - 2*p*q**2 + 2*p**2*((q*(1 + q))/2. + g**2)))/ \
             (q**2*(-1 + q**4))) + ((q - 2*p*q + p**2*(2*q - 2*(-1 + q)*(g - g**2)))* \
             (p**2*(1 + (-1 + q)*((1 - g)**2 + g**2)) +  \
               (1 - p)**2*(1 + 2*(-1 + q)*(1 - g) + (-2 + q)*(-1 + q)*(1 - g)**2 +  \
                  (-1 + q)*((1 - g)**2 + g**2))))/(-1 + q**4))/4. -  \
       np.log(((1 - 2*p + 2*p**2)*(q + 2*g**2)*(q**2 - 2*p*q**2 + 2*p**2*((q*(1 + q))/2. + g**2)))/ \
           (-1 + q**4) - ((q - 2*p*q + p**2*(2*q - 2*(-1 + q)*(g- g**2)))* \
             (p**2*(1 + (-1 + q)*((1 - g)**2 +g**2)) +  \
               (1 - p)**2*(1 + 2*(-1 + q)*(1 - g) + (-2 + q)*(-1 + q)*(1 - g)**2 +  \
                  (-1 + q)*((1 - g)**2 + g**2))))/(q**2*(-1 + q**4)))/4. +  \
       np.log(-(((1 - 2*p + 2*p**2)**2*(q + 2*g**2)**2)/(q**2*(-1 + q**4))) +  \
          (p**2*(1 + (-1 + q)*((1 - g)**2 + g**2)) +  \
              (1 - p)**2*(1 + 2*(-1 + q)*(1 - g) + (-2 + q)*(-1 + q)*(1 - g)**2 + \
                 (-1 + q)*((1 - g)**2 + g**2)))**2/(-1 + q**4))/8. + \
       np.log(((1 - 2*p + 2*p**2)**2*(q + 2*g**2)**2)/(-1 + q**4) -  \
          (p**2*(1 + (-1 + q)*((1 - g)**2 + g**2)) + \
              (1 - p)**2*(1 + 2*(-1 + q)*(1 - g) + (-2 + q)*(-1 + q)*(1 - g)**2 +  \
                 (-1 + q)*((1 - g)**2 + g**2)))**2/(q**2*(-1 + q**4)))/8.) 
    Jd = (-np.log(-((q**2 - 2*p*q**2 + 2*p**2*((q*(1 + q))/2. + g**2))**2/(q**2*(-1 + q**4))) + \
           (q - 2*p*q + p**2*(2*q - 2*(-1 + q)*(g - g**2)))**2/(-1 + q**4))/8. + \
       np.log((q**2 - 2*p*q**2 + 2*p**2*((q*(1 + q))/2. + g**2))**2/(-1 + q**4) - \
          (q - 2*p*q + p**2*(2*q - 2*(-1 + q)*(g - g**2)))**2/(q**2*(-1 + q**4)))/8. + \
       np.log(-(((1 - 2*p + 2*p**2)**2*(q + 2*g**2)**2)/(q**2*(-1 + q**4))) + \
          (p**2*(1 + (-1 + q)*((1 - g)**2 + g**2)) + \
              (1 - p)**2*(1 + 2*(-1 + q)*(1 - g) + (-2 + q)*(-1 + q)*(1 - g)**2 + (-1 + q)*((1 - g)**2 + g**2)))**2/(-1 + q**4)\
         )/8. -  np.log(((1 - 2*p + 2*p**2)**2*(q + 2*g**2)**2)/(-1 + q**4) - \
          (p**2*(1 + (-1 + q)*((1 - g)**2 + g**2)) + \
              (1 - p)**2*(1 + 2*(-1 + q)*(1 - g) + (-2 + q)*(-1 + q)*(1 - g)**2 + (-1 + q)*((1 - g)**2 + g**2)))**2/ \
           (q**2*(-1 + q**4)))/8.)
    J123 = (-np.log(-((q**2 - 2*p*q**2 + 2*p**2*((q*(1 + q))/2. + g**2))**2/(q**2*(-1 + q**4))) + \
           (q - 2*p*q + p**2*(2*q - 2*(-1 + q)*(g - g**2)))**2/(-1 + q**4))/8. + \
       np.log((q**2 - 2*p*q**2 + 2*p**2*((q*(1 + q))/2. + g**2))**2/(-1 + q**4) - \
          (q - 2*p*q + p**2*(2*q - 2*(-1 + q)*(g - g**2)))**2/(q**2*(-1 + q**4)))/8. + \
       np.log(-(((1 - 2*p + 2*p**2)*(q + 2*g**2)*(q**2 - 2*p*q**2 + 2*p**2*((q*(1 + q))/2. + g**2)))/(q**2*(-1 + q**4))) + \
          ((q - 2*p*q + p**2*(2*q - 2*(-1 + q)*(g - g**2)))*\
             (p**2*(1 + (-1 + q)*((1 - g)**2 + g**2)) + \
               (1 - p)**2*(1 + 2*(-1 + q)*(1 - g) + (-2 + q)*(-1 + q)*(1 - g)**2 + (-1 + q)*((1 - g)**2 + g**2))))/(-1 + q**4))\
         /4. - np.log(((1 - 2*p + 2*p**2)*(q + 2*g**2)*(q**2 - 2*p*q**2 + 2*p**2*((q*(1 + q))/2. + g**2)))/(-1 + q**4) - \
          ((q - 2*p*q + p**2*(2*q - 2*(-1 + q)*(g - g**2)))*\
            (p**2*(1 + (-1 + q)*((1 - g)**2 +g**2)) + \
               (1 - p)**2*(1 + 2*(-1 + q)*(1 - g) + (-2 + q)*(-1 + q)*(1 - g)**2 + (-1 + q)*((1 - g)**2 + g**2))))/\
           (q**2*(-1 + q**4)))/4. - np.log(-(((1 - 2*p + 2*p**2)**2*(q + 2*g**2)**2)/(q**2*(-1 + q**4))) + \
          (p**2*(1 + (-1 + q)*((1 - g)**2 + g**2)) + \
              (1 - p)**2*(1 + 2*(-1 + q)*(1 - g) + (-2 + q)*(-1 + q)*(1 - g)**2 + (-1 + q)*((1 - g)**2 + g**2)))**2/(-1 + q**4)\
          )/8. + np.log(((1 - 2*p + 2*p**2)**2*(q + 2*g**2)**2)/(-1 + q**4) - \
          (p**2*(1 + (-1 + q)*((1 - g)**2 + g**2)) + \
              (1 - p)**2*(1 + 2*(-1 + q)*(1 - g) + (-2 + q)*(-1 + q)*(1 - g)**2 + (-1 + q)*((1 - g)**2 + g**2)))**2/\
           (q**2*(-1 + q**4)))/8.)
    return Jh, Jd, J123

n_p = 39# int(sys.argv[1]) #39
pList = np.linspace(0.02,0.4,n_p)
n_g = 19 # int(sys.argv[2]) #19
gammaList = np.logspace(-5,-2,n_g)

N       =   64     #  size of the lattice, N x N
eqSteps = 2**8       #  number of MC sweeps for equilibration
mcSteps = 2**9       #  number of MC sweeps for calculation


E,M,C,X = np.zeros([n_g,n_p]), np.zeros([n_g,n_p]), np.zeros([n_g,n_p]), np.zeros([n_g,n_p])
MList = np.zeros(mcSteps)
n1, n2  = 1.0/(mcSteps*N*N), 1.0/(mcSteps*mcSteps*N*N) 


#  MAIN PART OF THE CODE
#----------------------------------------------------------------------
import time
## Fix pair wise interaction
for (ind_p,p) in enumerate(pList):
    for (ind_g,g) in enumerate(gammaList):
        # start Monte Carlo steps
        Jh, Jd, J123 = calcJs(p,g,2)
        MList = np.zeros(mcSteps)
        n1, n2  = 1.0/(mcSteps*N*N), 1.0/(mcSteps*mcSteps*N*N) 
        
        startTime = time.time()

        config = initialstate(N)         # initialise
        
        # Fixed boudary condition
        config[0, 0:N//2] = -1 * np.ones(N//2)
        config[0, N//2:N] = np.ones(N//2)
        
        E1 = M1 = E2 = M2 = 0
        iT = 1.0/1; iT2=iT*iT;


        for i in range(eqSteps):         # equilibrate
            mcmove(config, Jd, Jh, J123)           # Monte Carlo moves
            config[0, 0:N//2] = -1 * np.ones(N//2)
            config[0, N//2:N] = np.ones(N//2)

        
        for i in range(mcSteps):
            mcmove(config, Jd, Jh, J123)  
            config[0, 0:N//2] = -1 * np.ones(N//2)
            config[0, N//2:N] = np.ones(N//2)

        
            Ene = calcEnergy(config, Jd, Jh, J123)     # calculate the energy
            Mag = calcMag(config)        # calculate the magnetisation

            E1 = E1 + Ene
            M1 = M1 + Mag
            M2 = M2 + Mag*Mag 
            E2 = E2 + Ene*Ene
            
        np.savetxt('data/Ising/p='+str(p.round(2))+'g='+str(g.round(6))+'.out', config)

        # divide by number of sites and iteractions to obtain intensive values    
        E[ind_g,ind_p] = n1*E1
        M[ind_g,ind_p] = n1*M1
        C[ind_g,ind_p] = (n1*E2 - n2*E1*E1)*iT2
        X[ind_g,ind_p] = (n1*M2 - n2*M1*M1)*iT
        print('Jh=', Jh,', Jd=', Jd, ', J123=', J123, time.time()-startTime)

np.savetxt('data/Ising/E.out',E)
np.savetxt('data/Ising/M.out',M)
np.savetxt('data/Ising/C.out',C)
np.savetxt('data/Ising/X.out',X)

