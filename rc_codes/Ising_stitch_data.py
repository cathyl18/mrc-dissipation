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

def calcH(p,g,q):
    h = (np.log((p**2*(p**2 + 2*q - 4*p*q + 3*p**2*q))/(1 + q + q**2 + q**3)) + \
         3*np.log(((1 + q - 2*p*(1 + q) + p**2*(2 + q))*(1 + q**2 - 2*p*(1 + q**2) + p**2*(2 + q + q**2)))/ \
            ((1 + q)*(1 + q**2))) + 2*np.log(((1 + 2*(-1 + p)*p)* \
              (q + (p + (-1 + p)*q)**2 + 2*g + 2*(-2 + p)*p*g - (-1 + p)**2*g**2))/((1 + q)*(1 + q**2))) - \
         2*np.log(((1 + 2*(-1 + p)*p)*(q + (p + (-1 + p)*q)**2 - 2*(-1 + p)**2*q**2*g + (-1 + p)**2*q**2*g**2))/ \
            ((1 + q)*(1 + q**2))) - np.log(((q + 2*(-1 + p)*p*q)**2 - (p**2 + (-1 + p)**2*(q*(-1 + g)**2 - (-2 +g)*g))**2)/ \
           (-1 + q**4)) - 3*np.log((-(1 + 2*(-1 + p)*p)**2 + q**2*(p**2 + (-1 + p)**2*(q*(-1 + g)**2 - (-2 + g)*g))**2)/ \
            (-1 + q**4)))/8

    return h

###------------------------------------------------------------------------------------------------------------------------


p = float(sys.argv[1])
N  =   32     #  size of the lattice, N x N

gammaList = np.linspace(2.5e-05,9.75e-04,20).round(6)*10
E,M,C,X = np.zeros(len(gammaList)), np.zeros(len(gammaList)), np.zeros(len(gammaList)),np.zeros(len(gammaList))

#  MAIN PART OF THE CODE
#----------------------------------------------------------------------
import time



## Fix pair wise interaction

for (ind_g,g) in enumerate(gammaList):

    eqSteps = 2**18     #  number of MC sweeps for equilibration
    mcSteps = 2**18     #  number of MC sweeps for calculation

    n1, n2  = 1.0/(mcSteps*N*N), 1.0/(mcSteps*mcSteps*N*N)

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
            count+=1


    np.savetxt('/home1/yl244/data/Ising_ext_complete/E_trace_p='+str(p)+'g='+str(g.round(6))+'.out', EList)
    np.savetxt('/home1/yl244/data/Ising_ext_complete/M_trace_p='+str(p)+'g='+str(g.round(6))+'.out', MList)
    np.savetxt('/home1/yl244/data/Ising_ext_complete/C_trace_p='+str(p)+'g='+str(g.round(6))+'.out', CList)
    np.savetxt('/home1/yl244/data/Ising_ext_complete/X_trace_p='+str(p)+'g='+str(g.round(6))+'.out', XList)



    # divide by number of sites and iteractions to obtain intensive values
    E[ind_g] = n1*E1
    M[ind_g] = n1*M1
    C[ind_g] = (n1*E2 - n2*E1*E1)*iT2
    X[ind_g] = (n1*M2 - n2*M1*M1)*iT
    print('Jh=', Jh,', Jd=', Jd, ', J123=', J123, 'h=', h, 'total time so far=', time.time()-startTime, flush=True)
    print(E[ind_g],M[ind_g],C[ind_g],X[ind_g])

    
    # Stitch the new data with the old data with the same p
    
    oldE = np.loadtxt('/home1/yl244/data/Ising_ext/E_p'+str(p)+'.out')
    oldM = np.loadtxt('/home1/yl244/data/Ising_ext/M_p'+str(p)+'.out')
    oldC = np.loadtxt('/home1/yl244/data/Ising_ext/C_p'+str(p)+'.out')
    oldX = np.loadtxt('/home1/yl244/data/Ising_ext/X_p'+str(p)+'.out')

    newGammaList = np.concatenate([np.linspace(0,1e-3,21).round(6)*10,gammaList])
    newIndex = np.argsort(newGammaList)
    
    newE = np.concatenate((oldE,E), axis=0)
    newE = newE[newIndex]
    
    newM = np.concatenate((oldM,M), axis=0)
    newM = newM[newIndex]
    
    newC = np.concatenate((oldC,C), axis=0)
    newC = newC[newIndex]
    
    newX = np.concatenate((oldX,X), axis=0)
    newX = newX[newIndex]
    
    np.savetxt('/home1/yl244/data/Ising_ext_complete/E_p'+str(p)+'.out',newE)
    np.savetxt('/home1/yl244/data/Ising_ext_complete/M_p'+str(p)+'.out',newM)
    np.savetxt('/home1/yl244/data/Ising_ext_complete/C_p'+str(p)+'.out',newC)
    np.savetxt('/home1/yl244/data/Ising_ext_complete/X_p'+str(p)+'.out',newX)

