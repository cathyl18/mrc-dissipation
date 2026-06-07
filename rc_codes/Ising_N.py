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

                # bottom edge interaction (4 interaction)
                if a == 0:
                    pairnb = Jd * (config[(a+1)%N,(b-1)%N] + config[(a+1)%N,b]) #+ Jh * (config[a,(b-1)%N] + config[a,(b+1)%N]) 
                    trinb = J123 * config[(a+1)%N,b] * config[(a+1)%N,(b-1)%N]
                elif a==N-1:
                    
#                     pairnb = (Jd * (config[(a-1)%N,b] +config[(a-1)%N,(b+1)%N]) + \
#                           + Jh * (config[a,(b-1)%N] + config[a,(b+1)%N]) )
#                     trinb = J123 * (config[a,(b-1)%N] * config[(a-1)%N,b%N] + config[(a-1)%N,(b+1)%N] * config[a%N,(b+1)%N] )
                    #make sure that the cost is high enough so no flip on boundary
                    pairnb=s*100
                    trinb=s*100
                # bulk interaction (6 interactions)
                else:
                    pairnb = (Jd * (config[(a+1)%N,(b-1)%N] + config[(a-1)%N,b] + config[(a+1)%N,b] +config[(a-1)%N,(b+1)%N])  \
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

            if a == 0:
                pairnb = Jd * (config[(a+1)%N,(b-1)%N] + config[(a+1)%N,b]) #+ Jh * (config[a,(b-1)%N] + config[a,(b+1)%N]) 

                trinb = J123 * config[(a+1)%N,b] * config[(a+1)%N,(b-1)%N]
            elif a==N-1:
                pairnb = (Jd * (config[(a-1)%N,b] +config[(a-1)%N,(b+1)%N]) + \
                          + Jh * (config[a,(b-1)%N] + config[a,(b+1)%N]) )
                trinb = J123 * (config[a,(b-1)%N] * config[(a-1)%N,b%N] + config[(a-1)%N,(b+1)%N] * config[a%N,(b+1)%N] )

                # bulk interaction (6 interactions)
            else:
                pairnb = (Jd * (config[(a+1)%N,(b-1)%N] + config[(a-1)%N,b] + config[(a+1)%N,b] +config[(a-1)%N,(b+1)%N])  + Jh * (config[a,(b-1)%N] + config[a,(b+1)%N]) )
                trinb = J123 * (config[a,(b-1)%N] * config[(a-1)%N,b%N] + config[(a-1)%N,(b+1)%N] * config[a%N,(b+1)%N] + config[(a+1)%N,b] * config[(a+1)%N,(b-1)%N])

            energy -= S * (pairnb/2 + trinb/3) + S*h/3
    return energy  # to compensate for over-counting



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


#-----------------------------------------------------------------------------------------------------------

N  = int(sys.argv[1])   #  size of the lattice, N x N
p = float(sys.argv[2])
print(N)

n_g = 41
gammaList = np.linspace(0,1e-3,11).round(6)*10

E,M,C,X = np.zeros(len(gammaList)), np.zeros(len(gammaList)), np.zeros(len(gammaList)),np.zeros(len(gammaList))


#  MAIN PART OF THE CODE
#----------------------------------------------------------------------
import time


## Fix pair wise interaction

print('N=', N)

for (ind_g,g) in enumerate(gammaList):
    g= g.round(5)
    eqSteps = int(2**19*(N/32)**2 )      #  number of MC sweeps for equilibration
    mcSteps = int(2**19*(N/32)**2 )    #  number of MC sweeps for calculation

    print('g=',g, ' mc steps = ', mcSteps)

    # start Monte Carlo steps
    Jh, Jd, J123 = calcJs(p,g,2)
    h = calcH(p,g,2)
    MList = np.zeros(mcSteps)
    n1, n2  = 1.0/(mcSteps*N*N), 1.0/(mcSteps*mcSteps*N*N)


    startTime = time.time()

    config = initialstate(N)         # initialise
    config[N-1, 0:N//2] = -1 * np.ones(N//2)
    config[N-1, N//2:N] = np.ones(N//2)


    E1 = M1 = E2 = M2 = 0
    iT = 1.0/1; iT2=iT*iT;


    for i in range(eqSteps):         # equilibrate
        mcmove(config, Jd, Jh, J123,h)           # Monte Carlo moves
        config[N-1, 0:N//2] = -1 * np.ones(N//2)
        config[N-1, N//2:N] = np.ones(N//2)

    print('equilibriated after', time.time()-startTime, flush=True)

    for i in range(mcSteps):
        mcmove(config, Jd, Jh, J123,h)
        config[N-1, 0:N//2] = -1 * np.ones(N//2)
        config[N-1, N//2:N] = np.ones(N//2)


        Ene = calcEnergy(config, Jd, Jh, J123, h)     # calculate the energy
        Mag = calcMag(config)        # calculate the magnetisation

        E1 = E1 + Ene
        M1 = M1 + Mag
        M2 = M2 + Mag*Mag
        E2 = E2 + Ene*Ene

    E[ind_g] = n1*E1
    M[ind_g] = n1*M1
    C[ind_g] = (n1*E2 - n2*E1*E1)*iT2
    X[ind_g] = (n1*M2 - n2*M1*M1)*iT

    np.savetxt('/home1/yl244/data/Ising_N/E_N'+str(N)+'p'+str(p)+'.out',E)
    np.savetxt('/home1/yl244/data/Ising_N/M_N'+str(N)+'p'+str(p)+'.out',M)
    np.savetxt('/home1/yl244/data/Ising_N/C_N'+str(N)+'p'+str(p)+'.out',C)
    np.savetxt('/home1/yl244/data/Ising_N/X_N'+str(N)+'p'+str(p)+'.out',X)

    print('data saved, time cost = ', time.time()-startTime)
