import numpy as np
import random
from functools import reduce
from itertools import chain, combinations, product, repeat
from scipy import linalg, sparse
from scipy.stats import unitary_group
import time
import sys
import tracemalloc
import matplotlib.pyplot as plt
import pickle as pkl
import os


# Gates:

Id2 = np.array([[1,0],[0,1]], dtype = 'complex128')
S_X = np.array([[0,1],[1,0]], dtype = 'complex128')
S_Y = np.array([[0,-1j],[1j,0]], dtype = 'complex128')
S_Z = np.array([[1,0],[0,-1]], dtype = 'complex128')

S_plus = (S_X+1j*S_Y)/2
S_minus = (S_X-1j*S_Y)/2

# Single qubit 
qb={}
qb[0]=np.array([1,0],dtype='complex128')
qb[1]=np.array([0,1],dtype='complex128')


# 2 qubit gates
def mkU(n):
    Un = (np.random.random([n,n]) + 1j* np.random.random([n,n]))/np.sqrt(2)
    q,r = linalg.qr(Un)
    return q

def mkSingleSiteJumpKraus(rate, dt, ldbldOp):
    L = np.sqrt(rate) *ldbldOp    
    G = np.kron(L.conj(), L) - 1/2 * np.kron(np.eye(len(L)), L.T.conj()@L) - 1/2 * np.kron(L.T@L.conj(), np.eye(len(L)))
    U = linalg.expm(G * dt)

    X = np.reshape(np.transpose(np.reshape(U, [int(np.log2(len(U)))]*4),(0,2,1,3)),list(U.shape))
    [eigVals, eigVecs] = linalg.eigh(X)
    krausOps=[]
    for i in range(len(U)):
        if abs(eigVals[i]) > 1e-10:
            krausOps = krausOps + [ np.sqrt(eigVals[i]) * np.reshape(eigVecs[:,i],[2,2])]       
    return krausOps

def jump(rho_deep, N, KList):
    # transpose the first N indices
    rho_deep = np.transpose(rho_deep, list(reversed(np.arange(0,N)))+list(np.arange(N,2*N)))
    # fast Kraus sum
    for n in range(N):
        rho = rho_deep 
        newRho=np.zeros_like(rho_deep)
        # sum over all Kraus ops on one site
        for K in KList:
            # act by conjugation 
            # each index is automatically updated
            tempRho = np.tensordot(np.tensordot(K, rho, axes=[1, N-1]), K.T.conj(),  axes=[N, 0])
            newRho += tempRho
            
        # update the rho after summing over all Kraus ops on one site:
        rho_deep = newRho
        
    # transpose the first N indices back
    return np.transpose(rho_deep, list(reversed(np.arange(0,N)))+list(np.arange(N,2*N)))


def evolveRandU(rho_deep, N, layer=2):
    # Odd layers one shifts every qubit by one
    if layer%2 != 0:
        rho_deep = np.transpose(rho_deep,[N-1]+list(np.arange(N-1)) + [2*N-1] +list(np.arange(N,2*N-1)))
            
    # Transpose the first N indices in pairs; just like in the kraus case, but two indices at a time
    newIndex=[]
    for n in reversed(np.arange(0,N,2)):
        newIndex += [n,n+1]
    rho_deep = np.transpose(rho_deep, newIndex+list(np.arange(N,2*N)))

    for i in range(N//2):
        Ms = mkU(4)
        U = Ms.reshape([2]*4)
        UT = Ms.T.conj().reshape([2]*4) 
        # Acting on the first pair of qubits 
        tensorRho = np.tensordot(np.tensordot(U, rho_deep, axes = [[2,3], [N-2,N-1]]), UT, axes = [[N,N+1],[0,1]])
        rho_deep = tensorRho
       
    if layer%2 != 0:       
        # Now the indices are {i_N-3, i_N-2, ..., i_N-1, i_0}, {j_N-1, j_0, j_1, ..., j_N-2}
        rho_deep = np.transpose(rho_deep, newIndex+list(np.arange(N,2*N)))
        # Now the indices are {i_N-1, i_0, ..., i_N-3, i_N-2}, {j_N-1, j_0, j_1, ..., j_N-2}
        rho_deep = np.transpose(rho_deep, list(np.arange(1,N)) + [0] +list(np.arange(N+1,2*N)) +[N])

    else:
        # transpose even layers:
        rho_deep = np.transpose(rho_deep, newIndex+list(np.arange(N,2*N)))

    return rho_deep


def mkMsm(rho_deep, N, probMsm):
    sites = [n for n in range(N)]
    randomNumbers = np.random.uniform(0,1,Nqubit)
    msmSites = np.where(randomNumbers < probMsm)[0]

    Projs = [
            (np.eye(2)+S_Z)/2,
            (np.eye(2)+(-1)*S_Z)/2
        ]
    
    if list(msmSites) !=[]: 
        for ms in msmSites:
            # project the ms-th qubit
            newRho = np.tensordot( np.tensordot(Projs[0],rho_deep, axes=[1,ms]), Projs[0].T.conj(), axes=[N+ms, 0]) 
            # transpose back to the original 
            newRho = np.transpose(newRho,list(np.arange(1,ms+1))+[0]+list(np.arange(ms+1,N+ms))+ [2*N-1] + list(np.arange(N+ms, 2*N-1)) )
            prob_0 = np.real(np.trace(newRho.reshape([2**N,2**N])))
            # flip if the probability is greater than the Born prob in down spin
            if np.random.random(1) < prob_0:
                newRho = newRho/prob_0
            else: 
                newRho = np.tensordot( np.tensordot(Projs[1],rho_deep, axes=[1,ms]), Projs[1].T.conj(), axes=[N+ms, 0]) / (1-prob_0)
                newRho = np.transpose( newRho, list(np.arange(1,ms+1))+[0]+list(np.arange(ms+1,N+ms))+ [2*N-1] + list(np.arange(N+ms, 2*N-1)) )

            # update the density matrix after measurement
            rho_deep = newRho
            
    return rho_deep


def partial_trace(rho_deep, N, ele):
    # need to keep track of the new indices 
    ele = np.sort(ele)
    for (ind, e) in enumerate(ele):
        e1 = e - ind
        e2 = N + e1 - ind
        rho_deep = np.trace(rho_deep, axis1=e1, axis2=e2)
    return rho_deep

def calcS(rho_deep, N, ele):
    # remember to reshape the tensor to obtain eigenvalues
    reduced_Rho = np.reshape(partial_trace(rho_deep, N, ele), [2**(Nqubit - len(ele)),2**(Nqubit - len(ele))])
    eVals_1 = np.linalg.eigvalsh(reduced_Rho)
    eVals_1 = eVals_1[np.where(eVals_1>1e-12)]
    S1 = -sum(eVals_1*np.log2(eVals_1)) 
    
    otherEle = list(set([n for n in range(N)])-set(ele) )
    reduced_Rho_2 = np.reshape(partial_trace(rho_deep, N, otherEle), [2**(Nqubit - len(ele)),2**(Nqubit - len(ele))])
    eVals_2 = np.linalg.eigvalsh(reduced_Rho_2)
    eVals_2 = eVals_2[np.where(eVals_2>1e-12)]    
    S2 = -sum(eVals_2*np.log2(eVals_2)) 
    
    eVals_total = np.linalg.eigvalsh(np.reshape(rho_deep, [2**Nqubit,2**Nqubit]))
    eVals_total = eVals_total[np.where(eVals_total>1e-12)]
    Smin = -np.max(eVals_total) * np.log2(np.max(eVals_total))
    
    purity = sum(eVals_total**2)
    
    Iab = S1 + S2 + sum(eVals_total*np.log2(eVals_total)) 
    
    return S1, Smin, purity, Iab

def calcS_simple(rho_deep, N, ele):
    # remember to reshape the tensor to obtain eigenvalues
    reduced_Rho = np.reshape(partial_trace(rho_deep, N, ele), [2**(Nqubit - len(ele)),2**(Nqubit - len(ele))])
    eVals = np.linalg.eigvalsh(reduced_Rho)
    eVals = eVals[np.where(eVals>1e-12)]
    S = -sum(eVals*np.log2(eVals)) 

    return S

def calcGlassX(rho_deep, N):
    # Transform the tensor into matrix 
    rhoX = np.zeros_like(rho_deep)
    rhoXX= np.zeros_like(rho_deep)
    
    X1 = 0
    X2 = 0
    singleX = 0
    allprod = [ele for ele in product(np.arange(N),np.arange(N))]

    counter = 0
    for prod in allprod:
        # act by conjugation 
        #print(list(np.arange(1,n2+1))+[0]+list(np.arange(n2+1,2*N)), np.shape(rho_deep))
        tempMat = np.transpose(np.tensordot(S_X, rho_deep, axes=[1, prod[0]]),list(np.arange(1,prod[0]+1))+[0]+list(np.arange(prod[0]+1,2*N)))
        
        tempMat1 = np.tensordot(tempMat, tempMat, [np.arange(N),np.arange(N,2*N)])
        if counter < N:
            singleX = np.trace(np.reshape(tempMat1,[2**N, 2**N]))
        #X1 += np.trace(np.reshape(tempMat1,[2**N, 2**N]))
        
        tempMat2 = np.transpose(np.tensordot(S_X, rho_deep, axes=[1, prod[1]]),list(np.arange(1,prod[1]+1))+[0]+list(np.arange(prod[1]+1,2*N)))
        tempMat2 = np.tensordot(tempMat2, tempMat2, [np.arange(N),np.arange(N,2*N)])
        X1 += np.trace(np.reshape(tempMat1,[2**N, 2**N]))*np.trace(np.reshape(tempMat2,[2**N, 2**N]))
            
        tempMat = np.transpose(np.tensordot(S_X, tempMat, axes=[1, prod[1]]),list(np.arange(1,prod[1]+1))+[0]+list(np.arange(prod[1]+1,2*N)))
        tempMat = np.tensordot(tempMat, tempMat, [np.arange(N),np.arange(N,2*N)])
        # This should create (rho X_i X_j)^2
        X2 += np.trace(np.reshape(tempMat,[2**N, 2**N]))
        counter +=1

    return (X2-X1)/N, singleX/N
        
def calcMI_2pt(rho_deep, N):
    # compute the correlation of of the two antipodal points of length 1, e.g. the first and the N/2 site
    # by tracing out the rest of the system and subtracting the total entropy of the two sites
    
    S1 = calcS_simple(rho_deep, N, [1])
    S2 = calcS_simple(rho_deep, N, [N//2])
    S12 =  S2 = calcS_simple(rho_deep, N, [1, N//2])
    #print(S1,S2,S12)

    return S1+S2-S12
    
def calcTriMI(rho_deep, N):
    subSize = N//3
    S1 = calcS_simple(rho_deep, N, [n for n in range(subSize)] ) 
    S2 = calcS_simple(rho_deep, N, [n for n in range(subSize, 2*subSize)])
    S3 = calcS_simple(rho_deep, N, [n for n in range(2*subSize,N)])
    S12 = calcS_simple(rho_deep, N, [n for n in range(2*subSize)])
    S23 = calcS_simple(rho_deep, N, [n for n in range(subSize,N)] )
    S13 = calcS_simple(rho_deep, N, [n for n in range(subSize)]+ [n for n in range(2*subSize,N)])
    S123 = calcS_simple(rho_deep, N,[n for n in range(N)] )
    #print(S1,S2,S3,S12,S23,S13,S123)
    return S1+S2+S3-S12-S23-S13+S123


def calcGlassZ(rho_deep, N):
    # Transform the tensor into matrix 

    Z1 = 0
    Z2 = 0
    singleZ = 0
    allprod = [ele for ele in product(np.arange(N),np.arange(N))]

    counter = 0
    for prod in allprod:
        # act by conjugation 
        #print(list(np.arange(1,n2+1))+[0]+list(np.arange(n2+1,2*N)), np.shape(rho_deep))
        tempMat = np.transpose(np.tensordot(S_Z, rho_deep, axes=[1, prod[0]]),list(np.arange(1,prod[0]+1))+[0]+list(np.arange(prod[0]+1,2*N)))
        
        tempMat1 = np.tensordot(tempMat, tempMat, [np.arange(N),np.arange(N,2*N)])
        if counter < N:
            singleZ = np.trace(np.reshape(tempMat1,[2**N, 2**N]))
        #X1 += np.trace(np.reshape(tempMat1,[2**N, 2**N]))
        
        tempMat2 = np.transpose(np.tensordot(S_Z, rho_deep, axes=[1, prod[1]]),list(np.arange(1,prod[1]+1))+[0]+list(np.arange(prod[1]+1,2*N)))
        tempMat2 = np.tensordot(tempMat2, tempMat2, [np.arange(N),np.arange(N,2*N)])
        Z1 += np.trace(np.reshape(tempMat1,[2**N, 2**N]))*np.trace(np.reshape(tempMat2,[2**N, 2**N]))
            
        tempMat = np.transpose(np.tensordot(S_Z, tempMat, axes=[1, prod[1]]),list(np.arange(1,prod[1]+1))+[0]+list(np.arange(prod[1]+1,2*N)))
        tempMat = np.tensordot(tempMat, tempMat, [np.arange(N),np.arange(N,2*N)])
        # This should create (rho X_i X_j)^2
        Z2 += np.trace(np.reshape(tempMat,[2**N, 2**N]))
        counter +=1

    return (Z2-Z1)/N, singleZ/N
        
      
def calcNeg(rho, N, transpose_ele):
    transpose_ele = np.array(transpose_ele)
    if N < max(transpose_ele):
        print('error')
    else:
        newInd = list(N + transpose_ele) + list(np.arange(transpose_ele[-1]+1, N)) + list(transpose_ele) + list(np.arange(transpose_ele[-1]+N+1, 2*N))
        transRho = np.reshape(np.transpose(rho, newInd), [2**N, 2**N])
        neg = (np.trace(linalg.sqrtm(transRho.T.conj()@transRho))-1)/2
    if np.abs(np.imag(neg))<1e-9:
        neg = np.real(neg)         
    return neg


homepath = '/home1/yl244/data/RC_glass/'
os.makedirs(homepath,exist_ok=True)

# New method -

# initialize states:
state0 = (qb[0] + qb[1])/np.sqrt(2)
state0rho = np.outer(state0, state0)

Nqubit = 6
rho2_deep = state0rho
for n in range(Nqubit-1):
    rho2_deep = np.tensordot(rho2_deep,state0rho, axes=0)

    

dt=10
rateList = np.concatenate([np.array([0]),np.logspace(-5,-1,5)])
pList = np.concatenate([np.linspace(0,0.15,4),np.concatenate([np.linspace(0.17,0.33,6),np.linspace(.4,.7,4)])]).round(2)

ensN = 30
qab_X = {}
qab_Z = {}
MI = {}
triMI = {}

for p in pList:
    for r in rateList:
        if r==0:
            t = 100
        else:
            t = int(1/(r*dt))

        qab_X[p,r] = np.zeros([t,2],dtype='complex128')
        qab_Z[p,r] = np.zeros([t,2],dtype='complex128')
        MI[p,r] = np.zeros([t,1],dtype='complex128')
        triMI[p,r] = np.zeros([t,1],dtype='complex128')
        
        K1 = mkSingleSiteJumpKraus(r, dt, S_minus)

        startTime = time.time()
        for n in range(ensN):
            rho = rho2_deep

            for i in range(t):
                qab_X[p,r][i,:] += np.real(np.array(calcGlassX(rho,Nqubit))/ensN)
                qab_Z[p,r][i,:] += np.real(np.array(calcGlassZ(rho,Nqubit))/ensN)
                MI[p,r][i] += np.real(calcMI_2pt(rho,Nqubit)/ensN)
                triMI[p,r][i] += np.real(calcTriMI(rho,Nqubit)/ensN)

                rho = evolveRandU(rho, Nqubit,i)
                rho = jump(rho, Nqubit, K1)
                rho = mkMsm(rho, Nqubit, p)

        print(time.time()-startTime, flush=True)

with open(homepath + 'qab_X.pkl', 'wb') as f:
    pkl.dump(qab_X, f)
    
with open(homepath + 'qab_Z.pkl', 'wb') as f:
    pkl.dump(qab_Z, f)
    
with open(homepath + 'MI.pkl', 'wb') as f:
    pkl.dump(MI, f)
    
with open(homepath + 'triMI.pkl', 'wb') as f:
    pkl.dump(triMI, f)