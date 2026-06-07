import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from scipy.sparse import spdiags,linalg,eye
import sys
import os
import copy


##  BLOCK OF FUNCTIONS USED IN THE MAIN CODE
#----------------------------------------------------------------------

def initialstate(N):
    '''
    Generates a random spin configuration for initial condition
    '''
    state = 2*np.random.randint(2, size=(N,N))-1
    return state

def mcmove(config, Jd, Jh, J123, h):
    '''
    Monte Carlo move using Metropolis algorithm
    '''
    N = len(config)
    for i in range(N):
        for j in range(N):
                # try to flip the spin at random site (a,b)
                a = np.random.randint(0, N) # columns
                b = np.random.randint(0, N) # rows
                s =  config[a, b]

                pairnb = (Jd * (config[(a+1)%N,(b-1)%N] + config[(a-1)%N,b] + config[(a+1)%N,b] +config[(a-1)%N,(b+1)%N]) + \
                          + Jh * (config[a,(b-1)%N] + config[a,(b+1)%N]) )
                trinb = J123 * (config[a,(b-1)%N] * config[(a-1)%N,b%N] + config[(a-1)%N,(b+1)%N] * config[a%N,(b+1)%N] + config[(a+1)%N,b] * config[(a+1)%N,(b-1)%N])

                cost = 2*s*(pairnb + trinb)+ 2*s*h

                # move if cost<0 and if cost>0 and with probability < Boltzmann weight
                if cost < 0:
                    s *= -1
                elif rand() < np.exp(-cost):
                    s *= -1
                config[a, b] = s

    return config



def calcEnergy(config, Jd, Jh, J123, h):
    '''
    Energy of a given configuration
    '''
    energy = 0

    for a in range(len(config)):
        for b in range(len(config)):
            S = config[a,b]

            pairnb = (Jd * (config[(a+1)%N,(b-1)%N] + config[(a-1)%N,b] + config[(a+1)%N,b] +config[(a-1)%N,(b+1)%N])  + Jh * (config[a,(b-1)%N] + config[a,(b+1)%N]) )
            trinb = J123 * (config[a,(b-1)%N] * config[(a-1)%N,b%N] + config[(a-1)%N,(b+1)%N] * config[a%N,(b+1)%N] + config[(a+1)%N,b] * config[(a+1)%N,(b-1)%N])

            energy -= S * (pairnb/2 + trinb/3) + S*h/3
    return energy/2.  # to compensate for over-counting



def calcMag(config):
    '''
    Magnetization of a given configuration
    '''
    mag = np.sum(config)
    return mag

# The set of parameter space from p and Gamma
def calcJs(p,g,q):
    Jh =   -np.log(-((p**2*(1 + ((1 - g)**2 + g**2)*(-1 + q)) + \
              (1 - p)**2*(1 + 2*(1 - g)*(-1 + q) + ((1 - g)**2 + g**2)*(-1 + q) + \
                   (1 - g)**2*(-2 + q)*(-1 + q)))**2/(q**2*(-1 + q**4))) + \
          ((1 - 2*p + 2*p**2)**2*(2*g**2 + q)**2)/(-1 + q**4))/8. - \
    np.log((p**2*(1 + ((1 - g)**2 + g**2)*(-1 + q)) + \
           (1 - p)**2*(1 + 2*(1 - g)*(-1 + q) + ((1 - g)**2 + g**2)*(-1 + q) + \
              (1 - g)**2*(-2 + q)*(-1 + q)))**2/(-1 + q**4) - \
       ((1 - 2*p + 2*p**2)**2*(2*g**2 + q)**2)/(q**2*(-1 + q**4)))/8. + \
    np.log(-(((p**2*(1 + ((1 - g)**2 + g**2)*(-1 + q)) + \
               (1 - p)**2*(1 + 2*(1 - g)*(-1 + q) + ((1 - g)**2 + g**2)*(-1 + q) + \
                   (1 - g)**2*(-2 + q)*(-1 + q)))*(q - 2*p*q + p**2*(-2*(g - g**2)*(-1 + q) + 2*q)))/\
           (q**2*(-1 + q**4))) + ((1 - 2*p + 2*p**2)*(2*g**2 + q)*\
           (q**2 - 2*p*q**2 + 2*p**2*(g**2 + (q*(1 + q))/2.)))/(-1 + q**4))/4. + \
     np.log(((p**2*(1 + ((1 - g)**2 + g**2)*(-1 + q)) + \
              (1 - p)**2*(1 + 2*(1 - g)*(-1 + q) + ((1 - g)**2 + g**2)*(-1 + q) + \
                 (1 - g)**2*(-2 + q)*(-1 + q)))*(q - 2*p*q + p**2*(-2*(g - g**2)*(-1 + q) + 2*q)))/\
          (-1 + q**4) - ((1 - 2*p + 2*p**2)*(2*g**2 + q)*\
            (q**2 - 2*p*q**2 + 2*p**2*(g**2 + (q*(1 + q))/2.)))/(q**2*(-1 + q**4)))/4. - \
      np.log(-((q - 2*p*q + p**2*(-2*(g - g**2)*(-1 + q) + 2*q))**2/(q**2*(-1 + q**4))) + \
         (q**2 - 2*p*q**2 + 2*p**2*(g**2 + (q*(1 + q))/2.))**2/(-1 + q**4))/8. - \
      np.log((q - 2*p*q + p**2*(-2*(g - g**2)*(-1 + q) + 2*q))**2/(-1 + q**4) - \
         (q**2 - 2*p*q**2 + 2*p**2*(g**2 + (q*(1 + q))/2.))**2/(q**2*(-1 + q**4)))/8.\
    
    
    Jd =    np.log(-((p**2*(1 + ((1 - g)**2 + g**2)*(-1 + q)) + \
             (1 - p)**2*(1 + 2*(1 - g)*(-1 + q) + ((1 - g)**2 + g**2)*(-1 + q) + \
                (1 - g)**2*(-2 + q)*(-1 + q)))**2/(q**2*(-1 + q**4))) + \
         ((1 - 2*p + 2*p**2)**2*(2*g**2 + q)**2)/(-1 + q**4))/8. - \
      np.log((p**2*(1 + ((1 - g)**2 + g**2)*(-1 + q)) + \
             (1 - p)**2*(1 + 2*(1 - g)*(-1 + q) + ((1 - g)**2 + g**2)*(-1 + q) + \
                (1 - g)**2*(-2 + q)*(-1 + q)))**2/(-1 + q**4) - \
         ((1 - 2*p + 2*p**2)**2*(2*g**2 + q)**2)/(q**2*(-1 + q**4)))/8. - \
     np.log(-((q - 2*p*q + p**2*(-2*(g - g**2)*(-1 + q) + 2*q))**2/(q**2*(-1 + q**4))) + \
        (q**2 - 2*p*q**2 + 2*p**2*(g**2 + (q*(1 + q))/2.))**2/(-1 + q**4))/8. + \
      np.log((q - 2*p*q + p**2*(-2*(g - g**2)*(-1 + q) + 2*q))**2/(-1 + q**4) - \
         (q**2 - 2*p*q**2 + 2*p**2*(g**2 + (q*(1 + q))/2.))**2/(q**2*(-1 + q**4)))/8.\
    
    
    J123 =   -np.log(-((p**2*(1 + ((1 - g)**2 + g**2)*(-1 + q)) + \
               (1 - p)**2*(1 + 2*(1 - g)*(-1 + q) + ((1 - g)**2 + g**2)*(-1 + q) + \
                 (1 - g)**2*(-2 + q)*(-1 + q)))**2/(q**2*(-1 + q**4))) + \
         ((1 - 2*p + 2*p**2)**2*(2*g**2 + q)**2)/(-1 + q**4))/8. + \
    np.log((p**2*(1 + ((1 - g)**2 + g**2)*(-1 + q)) + \
           (1 - p)**2*(1 + 2*(1 - g)*(-1 + q) + ((1 - g)**2 + g**2)*(-1 + q) + \
                (1 - g)**2*(-2 + q)*(-1 + q)))**2/(-1 + q**4) - \
        ((1 - 2*p + 2*p**2)**2*(2*g**2 + q)**2)/(q**2*(-1 + q**4)))/8. + \
      np.log(-(((p**2*(1 + ((1 - g)**2 + g**2)*(-1 + q)) + \
             (1 - p)**2*(1 + 2*(1 - g)*(-1 + q) + ((1 - g)**2 + g**2)*(-1 + q) + \
                  (1 - g)**2*(-2 + q)*(-1 + q)))*(q - 2*p*q + p**2*(-2*(g - g**2)*(-1 + q) + 2*q)))/\
           (q**2*(-1 + q**4))) + ((1 - 2*p + 2*p**2)*(2*g**2 + q)*\
           (q**2 - 2*p*q**2 + 2*p**2*(g**2 + (q*(1 + q))/2.)))/(-1 + q**4))/4. - \
     np.log(((p**2*(1 + ((1 - g)**2 + g**2)*(-1 + q)) + \
            (1 - p)**2*(1 + 2*(1 - g)*(-1 + q) + ((1 - g)**2 + g**2)*(-1 + q) + \
                (1 - g)**2*(-2 + q)*(-1 + q)))*(q - 2*p*q + p**2*(-2*(g - g**2)*(-1 + q) + 2*q)))/\
         (-1 + q**4) - ((1 - 2*p + 2*p**2)*(2*g**2 + q)*\
            (q**2 - 2*p*q**2 + 2*p**2*(g**2 + (q*(1 + q))/2.)))/(q**2*(-1 + q**4)))/4. - \
      np.log(-((q - 2*p*q + p**2*(-2*(g - g**2)*(-1 + q) + 2*q))**2/(q**2*(-1 + q**4))) + \
         (q**2 - 2*p*q**2 + 2*p**2*(g**2 + (q*(1 + q))/2.))**2/(-1 + q**4))/8. + \
      np.log((q - 2*p*q + p**2*(-2*(g - g**2)*(-1 + q) + 2*q))**2/(-1 + q**4) - \
         (q**2 - 2*p*q**2 + 2*p**2*(g**2 + (q*(1 + q))/2.))**2/(q**2*(-1 + q**4)))/8.
    
    return -Jh, -Jd, -J123

def calcH(p,g,q):
    h =  np.log(-((p**2*(1 + ((1 - g)**2 + g**2)*(-1 + q)) + \
          (1 - p)**2*(1 + 2*(1 - g)*(-1 + q) + ((1 - g)**2 + g**2)*(-1 + q) + \
              (1 - g)**2*(-2 + q)*(-1 + q)))**2/(q**2*(-1 + q**4))) + \
     ((1 - 2*p + 2*p**2)**2*(2*g**2 + q)**2)/(-1 + q**4))/8. + \
    (3*np.log((p**2*(1 + ((1 - g)**2 + g**2)*(-1 + q)) + \
           (1 - p)**2*(1 + 2*(1 - g)*(-1 + q) + ((1 - g)**2 + g**2)*(-1 + q) + \
             (1 - g)**2*(-2 + q)*(-1 + q)))**2/(-1 + q**4) - \
      ((1 - 2*p + 2*p**2)**2*(2*g**2 + q)**2)/(q**2*(-1 + q**4))))/8. - \
    np.log(-(((p**2*(1 + ((1 - g)**2 + g**2)*(-1 + q)) + \
           (1 - p)**2*(1 + 2*(1 - g)*(-1 + q) + ((1 - g)**2 + g**2)*(-1 + q) + \
              (1 - g)**2*(-2 + q)*(-1 + q)))*(q - 2*p*q + p**2*(-2*(g - g**2)*(-1 + q) + 2*q)))/\
       (q**2*(-1 + q**4))) + ((1 - 2*p + 2*p**2)*(2*g**2 + q)*\
       (q**2 - 2*p*q**2 + 2*p**2*(g**2 + (q*(1 + q))/2.)))/(-1 + q**4))/4. + \
    np.log(((p**2*(1 + ((1 - g)**2 + g**2)*(-1 + q)) + \
          (1 - p)**2*(1 + 2*(1 - g)*(-1 + q) + ((1 - g)**2 + g**2)*(-1 + q) + \
            (1 - g)**2*(-2 + q)*(-1 + q)))*(q - 2*p*q + p**2*(-2*(g - g**2)*(-1 + q) + 2*q)))/\
     (-1 + q**4) - ((1 - 2*p + 2*p**2)*(2*g**2 + q)*\
       (q**2 - 2*p*q**2 + 2*p**2*(g**2 + (q*(1 + q))/2.)))/(q**2*(-1 + q**4)))/4. - \
    (3*np.log(-((q - 2*p*q + p**2*(-2*(g - g**2)*(-1 + q) + 2*q))**2/(q**2*(-1 + q**4))) + \
       (q**2 - 2*p*q**2 + 2*p**2*(g**2 + (q*(1 + q))/2.))**2/(-1 + q**4)))/8. - \
     np.log((q - 2*p*q + p**2*(-2*(g - g**2)*(-1 + q) + 2*q))**2/(-1 + q**4) - \
     (q**2 - 2*p*q**2 + 2*p**2*(g**2 + (q*(1 + q))/2.))**2/(q**2*(-1 + q**4)))/8.\
    
    return -h

###------------------------------------------------------------------------------------------------------------------------

p = float(sys.argv[1])
ind_g = int(sys.argv[2])

N  =  32   #  size of the lattice, N x N

n_g = 21
gammaList = np.linspace(0,1e-3,n_g).round(6)*10
g=gammaList[ind_g]
print('p=',p,'g=',g)

#  MAIN PART OF THE CODE
#----------------------------------------------------------------------
import time


eqSteps = 2**19     #  number of MC sweeps for equilibration
mcSteps = 2**19    #  number of MC sweeps for calculation


# start Monte Carlo steps
Jh, Jd, J123 = calcJs(p,g,2)
h = calcH(p,g,2)

n1, n2  = 1.0/(mcSteps*N*N), 1.0/(mcSteps*mcSteps*N*N)

startTime = time.time()

config = initialstate(N)   

E1 = M1 = E2 = M2 = 0
iT = 1.0/1; iT2=iT*iT;

for i in range(eqSteps):         # equilibrate
    mcmove(config, Jd, Jh, J123,h)           # Monte Carlo moves

print('equilibriated after', time.time()-startTime, flush=True)

for i in range(mcSteps):
    mcmove(config, Jd, Jh, J123,h)

    Ene = calcEnergy(config, Jd, Jh, J123, h)     # calculate the energy
    Mag = calcMag(config)        # calculate the magnetisation

    E1 = E1 + Ene
    M1 = M1 + Mag
    M2 = M2 + Mag*Mag
    E2 = E2 + Ene*Ene


E= n1*E1
M= n1*M1
C= (n1*E2 - n2*E1*E1)*iT2
X= (n1*M2 - n2*M1*M1)*iT

print('Jh=', Jh,', Jd=', Jd, ', J123=', J123, 'h=', h, time.time()-startTime, flush=True)
print('E=', E, 'C=', C, 'M=', M, 'X=',X)


np.savetxt('/home1/yl244/data/Ising_pbc/E_p='+str(p)+'g='+str(g.round(6))+'.out', np.array([E]) )
np.savetxt('/home1/yl244/data/Ising_pbc/M_p='+str(p)+'g='+str(g.round(6))+'.out', np.array([M]) )
np.savetxt('/home1/yl244/data/Ising_pbc/C_p='+str(p)+'g='+str(g.round(6))+'.out', np.array([C]) )
np.savetxt('/home1/yl244/data/Ising_pbc/X_p='+str(p)+'g='+str(g.round(6))+'.out', np.array([X]) )


    ## Deprecated procedure for computing the autocorrelation function

#         # create boxes every (2**13) steps
#         if (i == boxCount* frameCount_max):

#             frames = np.zeros([N,N,14])
#             boxCount +=1
#             frameCount = 0

#         # cut the 2**20 data points into chuncks 2**6 and 14 data points across 2**14 points
#         # every folder should have 6 box.outs, and each of them is a NxNx14 matrix
#         if (i == ((boxCount -1)* frameCount_max+ 2**frameCount -1) ):
#             frames[:,:,frameCount] = config>0

#             # Try to refresh every frame:
#             with open('data/Ising_phase/p='+str(p)+'g='+str(g.round(6))+'boxes/b='+str(boxCount)+'.out', 'wb') as f:
#                 np.save(f, frames)
#             print('data/Ising_phase/p='+str(p)+'g='+str(g.round(6))+'boxes/b='+str(boxCount)+'.out')
#             print('frame saved', flush=True)
#             frameCount +=1
