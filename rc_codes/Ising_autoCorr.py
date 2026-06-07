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


N  =   32     #  size of the lattice, N x N

paraList = np.array([[0.351,0.008],[0.167,0.002],[0.351,0.002]])

t_wList = np.array([2**3,2**5,2**7,2**9,2**11,2**13,2**15,2**17])-1

segments=np.logspace(1,17,5,base=2).round()
deltaT = np.array([1])
for (indT,T) in enumerate(segments[:-1]):
    deltaT = np.concatenate([deltaT, np.logspace( np.log2(segments[indT]+1),np.log2(segments[indT+1]),10+3*indT, base=2).round().astype(int)] )

deltaT=deltaT.flatten()
print(len(deltaT))

traj = sys.argv[1]



#  MAIN PART OF THE CODE
#----------------------------------------------------------------------
import time

## Fix pair wise interaction

for (ind, params) in enumerate(paraList):

    eqSteps = 2**18   #  number of MC sweeps for equilibration
    mcSteps = 2**18    #  number of MC sweeps for calculation

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

    count0 = 0  # for observable recording
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0  # for observable recording
    count5 = 0
    count6 = 0
    count7 = 0
    
    print('alltime=', t_wList[1] + deltaT)
    count = 0
    for i in range(mcSteps):
        mcmove(config, Jd, Jh, J123,h)
        config[N-1, 0:N//2] = -1 * np.ones(N//2)
        config[N-1, N//2:N] = np.ones(N//2)

        # Initialize waiting time 0
        if i == t_wList[0]:
            state0 = copy.deepcopy(config)
            mag0 = calcMag(state0)/(N*N)
            allC0 = np.zeros_like(deltaT).astype(float)
        
        # start data recording
        if count0<=len(deltaT)-1:
            if (i == int(t_wList[0] + deltaT[count0])):
                allC0[count0] = np.sum(config*state0)/(N*N) - calcMag(config)/(N*N) * mag0
                #print(np.sum(config*state0)/(N**2), calcMag(config)/(N*N), mag0, allC0[count0])
                count0 +=1

        # Initialize waiting time 1
        if i == t_wList[1]:
            state1 = copy.deepcopy(config)
            mag1 = calcMag(state1)/(N*N)
            allC1 = np.zeros_like(deltaT).astype(float)
            
        if count1<=len(deltaT)-1:
            if i == int(t_wList[1] + deltaT[count1]):
                allC1[count1] = np.sum(config*state1)/(N*N) - calcMag(config)/(N*N) * mag1
                #print('count1=',count1,'time=',t_wList[1] + deltaT[count1])
                #print(allC1,np.sum(config*state1)/(N*N) - calcMag(config)/(N*N) * mag1 )
                #print('next time=', int(t_wList[1] + deltaT[count1+1]))
                count1 +=1

        # Initialize waiting time 2
        if i == t_wList[2]:
            state2 = copy.deepcopy(config)
            mag2 = calcMag(state2)/(N*N)
            allC2 = np.zeros_like(deltaT).astype(float)
        
        if count2<=len(deltaT)-1:
            if i == int(t_wList[2] + deltaT[count2]):
                allC2[count2] = np.sum(config*state2)/(N*N) - calcMag(config)/(N*N) * mag2
                count2 +=1

        # Initialize waiting time 3
        if i == t_wList[3]:
            state3 = copy.deepcopy(config)
            mag3 = calcMag(state3)/(N*N)
            allC3 = np.zeros_like(deltaT).astype(float)
        
        if count3<=len(deltaT)-1:
            # start data recording
            if i == int(t_wList[3] + deltaT[count3]):
                allC3[count3] = np.sum(config*state3)/(N*N) - calcMag(config)/(N*N) * mag3
                count3 +=1
                
        # Initialize waiting time 4
        if i == t_wList[4]:
            state4 = copy.deepcopy(config)
            mag4 = calcMag(state4)/(N*N)
            allC4 = np.zeros_like(deltaT).astype(float)
        
        if count4<=len(deltaT)-1:
            # start data recording
            if i == int(t_wList[4] + deltaT[count4]):
                allC4[count4] = np.sum(config*state4)/(N*N) - calcMag(config)/(N*N) * mag4
                print(allC4[count4])
                count4 +=1
                
        # Initialize waiting time 5
        if i == t_wList[5]:
            state5 = copy.deepcopy(config)
            mag5 = calcMag(state5)/(N*N)
            allC5 = np.zeros_like(deltaT).astype(float)
        
        if count5<=len(deltaT)-1:
            # start data recording
            if i == int(t_wList[5] + deltaT[count5]):
                allC5[count5] = np.sum(config*state5)/(N*N) - calcMag(config)/(N*N) * mag5
                print(allC5[count5])
                count5 +=1

        # Initialize waiting time 6
        if i == t_wList[6]:
            state6 = copy.deepcopy(config)
            mag6 = calcMag(state6)/(N*N)
            allC6 = np.zeros_like(deltaT).astype(float)
        
        if count6<=len(deltaT)-1:
            # start data recording
            if i == int(t_wList[6] + deltaT[count6]):
                allC6[count6] = np.sum(config*state6)/(N*N) - calcMag(config)/(N*N) * mag6
                print(allC6[count6])

                count6 +=1
                
        # Initialize waiting time 7
        if i == t_wList[7]:
            state7 = copy.deepcopy(config)
            mag7 = calcMag(state7)/(N*N)
            allC7 = np.zeros_like(deltaT).astype(float)
        
        if count7<=len(deltaT)-1:
            # start data recording
            if i == int(t_wList[7] + deltaT[count7]):
                allC7[count7] = np.sum(config*state7)/(N*N) - calcMag(config)/(N*N) * mag7
                print(allC7[count7])

                count7 +=1   
  
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

            #with open('/home1/yl244/data/Ising_states/p='+str(p)+'g='+str(g.round(6))+'_state'+str(count)+'.out', 'wb') as f:
            #    np.save(f, config)

            count+=1

    np.savetxt('/home1/yl244/data/Ising_autocorr/p='+str(p)+'g='+str(g.round(6))+ 'autoC' + str(t_wList[0]) + 'ind' + str(traj)+ '.out', allC0)
    np.savetxt('/home1/yl244/data/Ising_autocorr/p='+str(p)+'g='+str(g.round(6))+ 'autoC' + str(t_wList[1]) + 'ind' + str(traj)+ '.out', allC1)
    np.savetxt('/home1/yl244/data/Ising_autocorr/p='+str(p)+'g='+str(g.round(6))+ 'autoC' + str(t_wList[2]) + 'ind' + str(traj)+ '.out', allC2)
    np.savetxt('/home1/yl244/data/Ising_autocorr/p='+str(p)+'g='+str(g.round(6))+ 'autoC' + str(t_wList[3]) + 'ind' + str(traj)+ '.out', allC3)
    np.savetxt('/home1/yl244/data/Ising_autocorr/p='+str(p)+'g='+str(g.round(6))+ 'autoC' + str(t_wList[4]) + 'ind' + str(traj)+ '.out', allC4)
    np.savetxt('/home1/yl244/data/Ising_autocorr/p='+str(p)+'g='+str(g.round(6))+ 'autoC' + str(t_wList[5]) + 'ind' + str(traj)+ '.out', allC5)
    np.savetxt('/home1/yl244/data/Ising_autocorr/p='+str(p)+'g='+str(g.round(6))+ 'autoC' + str(t_wList[6]) + 'ind' + str(traj)+ '.out', allC6)
    np.savetxt('/home1/yl244/data/Ising_autocorr/p='+str(p)+'g='+str(g.round(6))+ 'autoC' + str(t_wList[7]) + 'ind' + str(traj)+ '.out', allC7)



    np.savetxt('/home1/yl244/data/Ising_autocorr/E_trace_p='+str(p)+'g='+str(g.round(6))+'ind' + str(traj)+ '.out', EList)
    np.savetxt('/home1/yl244/data/Ising_autocorr/M_trace_p='+str(p)+'g='+str(g.round(6))+'ind' + str(traj)+ '.out', MList)
    np.savetxt('/home1/yl244/data/Ising_autocorr/C_trace_p='+str(p)+'g='+str(g.round(6))+'ind' + str(traj)+ '.out', CList)
    np.savetxt('/home1/yl244/data/Ising_autocorr/X_trace_p='+str(p)+'g='+str(g.round(6))+'ind' + str(traj)+ '.out', XList)

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
