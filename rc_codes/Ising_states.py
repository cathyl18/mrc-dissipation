import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from scipy.sparse import spdiags,linalg,eye
import sys
import os

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
                    pairnb = Jd * (config[(a+1)%N,(b-1)%N] + config[(a+1)%N,b])

                    trinb = J123 * config[(a+1)%N,b] * config[(a+1)%N,(b-1)%N]
                elif a==N-1:
                    pairnb = (Jd * (config[(a-1)%N,b] +config[(a-1)%N,(b+1)%N]) + \
                          + Jh * (config[a,(b-1)%N] + config[a,(b+1)%N]) )
                    trinb = J123 * (config[a,(b-1)%N] * config[(a-1)%N,b%N] + config[(a-1)%N,(b+1)%N] * config[a%N,(b+1)%N] )

                # bulk interaction (6 interactions)
                else:
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

            if a == 0:
                pairnb = Jd * (config[(a+1)%N,(b-1)%N] + config[(a+1)%N,b])

                trinb = J123 * config[(a+1)%N,b] * config[(a+1)%N,(b-1)%N]
            elif a==N-1:
                pairnb = (Jd * (config[(a-1)%N,b] +config[(a-1)%N,(b+1)%N]) + \
                          + Jh * (config[a,(b-1)%N] + config[a,(b+1)%N]) )
                trinb = J123 * (config[a,(b-1)%N] * config[(a-1)%N,b%N] + config[(a-1)%N,(b+1)%N] * config[a%N,(b+1)%N] )

                # bulk interaction (6 interactions)
            else:
                pairnb = (Jd * (config[(a+1)%N,(b-1)%N] + config[(a-1)%N,b] + config[(a+1)%N,b] +config[(a-1)%N,(b+1)%N]) + \
                              + Jh * (config[a,(b-1)%N] + config[a,(b+1)%N]) )
                trinb = J123 * (config[a,(b-1)%N] * config[(a-1)%N,b%N] + config[(a-1)%N,(b+1)%N] * config[a%N,(b+1)%N] + config[(a+1)%N,b] * config[(a+1)%N,(b-1)%N])

            energy += -S * (pairnb + trinb) -S*h
    return energy/2.  # to compensate for over-counting



def calcMag(config):
    '''
    Magnetization of a given configuration
    '''
    mag = np.sum(config)
    return mag

# The set of parameter space from p and Gamma
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


N  =   32     #  size of the lattice, N x N

paraList = np.array([[0.25,0.0],[0.25,0.002],[0.25,0.004],[0.25,0.007],[0.25,0.01]])




#  MAIN PART OF THE CODE
#----------------------------------------------------------------------
import time

## Fix pair wise interaction

for (ind, params) in enumerate(paraList):

    eqSteps = 2**19    #  number of MC sweeps for equilibration
    mcSteps = 2**19    #  number of MC sweeps for calculation

    n1, n2  = 1.0/(mcSteps*N*N), 1.0/(mcSteps*mcSteps*N*N)
 
    [p,g] = params
    print('params set p=', p, 'g=', g)

    # start Monte Carlo steps
    Jh, Jd, J123 = calcJs(p,g,2)
    h = calcH(p,g,2)
    MList = np.zeros(mcSteps)
    n1, n2  = 1.0/(mcSteps*N*N), 1.0/(mcSteps*mcSteps*N*N)

    # save 100 steps, every
    dataSize = 2**7
    EList = np.zeros(dataSize)
    MList = np.zeros(dataSize)
    CList = np.zeros(dataSize)
    XList = np.zeros(dataSize)

    startTime = time.time()

    config = initialstate(N)         # initialise
    # Fixed boudary condition (upper half +/-1 and lower open)
    config[N-1, 0:N//2] = -1 * np.ones(N//2)
    config[N-1, N//2:N] = np.ones(N//2)
    #config[0, :] = np.ones(N)


    E1 = M1 = E2 = M2 = 0
    iT = 1.0/1; iT2=iT*iT;


    for i in range(eqSteps):         # equilibrate
        mcmove(config, Jd, Jh, J123,h)           # Monte Carlo moves
        config[N-1, 0:N//2] = -1 * np.ones(N//2)
        config[N-1, N//2:N] = np.ones(N//2)

    print('equilibriated after', time.time()-startTime, flush=True)

    count = 0  # for observable recording
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

        dataInteval = 2**5
        if (i > (mcSteps - dataInteval*dataSize -1) ) & (i% dataInteval ==0):

            EList[count] = E1/((i+1)*N*N)
            MList[count] = M1/((i+1)*N*N)
            CList[count] = (E2/((i+1)*N*N) - E1*E1/((i+1)*(i+1)*N*N))*iT2
            XList[count] = (M2/((i+1)*N*N) - M1*M1/((i+1)*(i+1)*N*N))*iT

            with open('/home1/yl244/data/Ising_states/p='+str(p)+'g='+str(g.round(6))+'_state'+str(count)+'.out', 'wb') as f:
                np.save(f, config)
            count+=1


    np.savetxt('/home1/yl244/data/Ising_states/E_trace_p='+str(p)+'g='+str(g.round(6))+'.out', EList)
    np.savetxt('/home1/yl244/data/Ising_states/M_trace_p='+str(p)+'g='+str(g.round(6))+'.out', MList)
    np.savetxt('/home1/yl244/data/Ising_states/C_trace_p='+str(p)+'g='+str(g.round(6))+'.out', CList)
    np.savetxt('/home1/yl244/data/Ising_states/X_trace_p='+str(p)+'g='+str(g.round(6))+'.out', XList)

    print('Jh=', Jh,', Jd=', Jd, ', J123=', J123, 'h=', h, time.time()-startTime, flush=True)


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
