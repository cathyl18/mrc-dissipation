import numpy as np
import random
from functools import reduce
from itertools import chain, combinations, product, repeat
from scipy import linalg, sparse
from scipy.stats import unitary_group
import time


# Gates:

Id2 = np.array([[1,0],[0,1]], dtype = 'complex128')
S_X = np.array([[0,1],[1,0]], dtype = 'complex128')
S_Y = np.array([[0,-1j],[1j,0]], dtype = 'complex128')
S_Z = np.array([[1,0],[0,-1]], dtype = 'complex128')
paulis = [Id2, S_X, S_Y, S_Z]/np.sqrt(2)
basis4by4=np.kron(paulis,paulis)

S_plus = (S_X+1j*S_Y)/2
S_minus = (S_X-1j*S_Y)/2

qb={}
qb[0]=np.array([1,0],dtype='complex128')
qb[1]=np.array([0,1],dtype='complex128')



def inner(X,Y):
    if type(X)!=np.ndarray:
        X=X.toarray()
        Y=Y.toarray()
    prod=np.trace(X.T.conj()@Y)
    return prod

# uniform-U(4) matrices:
# checked that the eigenvalue distribution is uniform
def mkU(n):
    Un = (np.random.random([n,n]) + 1j* np.random.random([n,n]))/np.sqrt(2)
    q,r = linalg.qr(Un)
    #d = np.diag(r)
    #ph = d/abs(d)
    #q = q * ph * q
    return q

def mkLiouvillian(rate, dt, ldbldOp):

    L1 = np.kron(np.sqrt(rate/2) *ldbldOp, np.eye(2))
    L2 = np.kron(np.eye(2), np.sqrt(rate/2) *ldbldOp)

    h = -1j * linalg.logm(unitary_group.rvs(4)) #- (np.kron(ldbldOp, np.eye(2)) + np.kron(np.eye(2), ldbldOp))

    ## if unitary_group does not work:
    #h = -1j * linalg.logm(mkU(4))

    H =  (np.kron(h.conj(), np.eye(4)) - np.kron(np.eye(4), h) )

    #print(np.allclose(H, H.T.conj()))
    lindblads = [L1,L2]

    G = np.zeros([len(L1)**2,len(L1)**2], dtype = 'complex128')
    for L in lindblads:
        G += np.kron(L.conj(), L) - 1/2 * np.kron(np.eye(len(L)), L.T.conj()@L) - 1/2 * np.kron(L.T@L.conj(), np.eye(len(L)))

    # print(np.allclose(G, G.T.conj()) )
    denseU=linalg.expm( (-1j*H + G) * dt)

    return denseU # to act on the vectorized density matrix of 2 qubits

def mkKrausOps(U):
    X = np.reshape(np.transpose(np.reshape(U,[4]*int(np.log2(len(U)))),[0,2,1,3]),list(U.shape))
    [eigVals, eigVecs] = linalg.eigh(X)

    krausOps=[]

    for i in range(len(U)):
        if abs(eigVals[i])>1e-10:
            krausOps = krausOps + [ sparse.csr_matrix(np.sqrt(eigVals[i]) * np.reshape(eigVecs[:,i],[4,4]))]

    return krausOps


def decompKrausOps(krausOps):
    # for each kraus operator, there is a set of k_abs that
    # stores the inner product of each on-site kraus ops with the basis in a row

    k_abs=np.array([[] for i in range(16)], dtype='complex128').T
    for (k,op) in enumerate(krausOps):
        k_temp = np.zeros(len(basis4by4), dtype='complex128')
        for (i,b) in enumerate(basis4by4):
            k_temp[i] = inner(b, op)
        k_abs = np.vstack([k_abs,k_temp])

    return k_abs

def mkGlobalKrausOps(layer, Nqubit, rate, dt, ldbldOp, debug=False):

    allKs = {}

    if np.mod(layer,2) == 0:
        # position records the site with the non-trivial unitary

        positions = [i for i in range(0, Nqubit-1, 2)]

        for pos in positions:

            # one needs to make N/2 different random unitaries
            krausOps = mkKrausOps(mkLiouvillian(rate, dt, ldbldOp))
            allKs[pos] = []

            for i in range(len(krausOps)):
                U_chain = chain(repeat(np.eye(4), pos//2), [krausOps[i]], repeat(np.eye(4), Nqubit//2 - pos//2 - 1))
                allKs[pos] += [reduce(sparse.kron, U_chain)]

    else:
        positions = list(range(1,Nqubit,2))

        for pos in positions:
            krausOps = mkKrausOps(mkLiouvillian(rate, dt, ldbldOp))
            allKs[pos] = []

            if pos < Nqubit-1:

                for i in range(len(krausOps)):
                    U_chain = chain([np.eye(2)], repeat(np.eye(4), pos//2), [krausOps[i]], repeat(np.eye(4), Nqubit//2 - pos//2 - 2), [np.eye(2)])
                    allKs[pos] += [reduce(sparse.kron, U_chain)]

            else:
                # decomposition of Kraus is required with PBC

                k_abs = decompKrausOps(krausOps)

                for row in k_abs:

                    K_jk = sparse.csr_matrix(np.zeros([2**Nqubit,2**Nqubit], dtype='complex128') )

                    for (basisInd, whichPauli) in enumerate(row):
                        U_chain = np.array([np.eye(2)] * Nqubit,dtype='complex128')
                        U_chain[pos] = paulis[basisInd//4]
                        U_chain[np.mod(pos+1,Nqubit)] = paulis[np.mod(basisInd,4)]

                        K_jk +=  row[basisInd] * reduce(sparse.kron, chain(U_chain))

                    allKs[pos] = allKs[pos] + [K_jk]

    return allKs

# employ the faster Kraus sum
def evolveKrausU(allKs, rho, debug = False):

    Nqubit = int(np.log2(len(rho.toarray())))

    newRho = rho
    for key in allKs.keys():
        rho = newRho
        newRho = sparse.csr_matrix(np.zeros([2**Nqubit,2**Nqubit]) )

        for mat in allKs[key]:
            newRho += mat @ rho @ mat.T.conj()

            #if debug == True:
            #    print(np.trace(newRho.toarray()))

    return newRho



def mkS_zMsm(probMsm, rho , debug = False):

    Nqubit = int(np.log2(len(rho.toarray())))
    sites = [n for n in range(Nqubit)]
    randomNumbers = np.random.uniform(0,1,Nqubit)
    msmSites = np.where(randomNumbers < probMsm)[0]

    if list(msmSites) !=[]:

        Projs = [
            (np.eye(2)+S_Z)/2,
            (np.eye(2)+(-1)*S_Z)/2
        ]

        numMsSites=len(msmSites)

        for (j,ms) in enumerate(msmSites):
            proj_chain = [np.eye(2)] * Nqubit
            proj_chain[ms] = Projs[0]
            proj0p = reduce(sparse.kron, proj_chain)

            if debug == True: print(proj_chain)

            proj_chain = [np.eye(2)] * Nqubit
            proj_chain[ms] = Projs[1]
            proj1p = reduce(sparse.kron, proj_chain)

            if debug == True: print("projector", proj0p.shape, " rho", rho.shape)

            # probability of having one of the outcomes
            prob = np.trace((proj0p @ rho @ proj0p.T.conj()).toarray())

            if np.random.random(1) < prob:
                testRho = (proj0p @ rho @ proj0p.T.conj())/prob

            else:
                testRho = (proj1p @ rho @ proj1p.T.conj())/(1-prob)

            rho = testRho

    else:
        testRho = rho

    return testRho


def ptrace_rho(rho, elements):
    operator = rho
    halflength = int(np.log2(operator.shape[0]))
    reorder = np.reshape(operator, (2,)*(2*halflength))
    temp = np.transpose(reorder, [n//2+halflength*(n%2) for n in range(0,halflength*2)])

    elems = set(elements)
    counter = 0
    indexarr = []
    for i in range(len(temp.shape)//2):
        if i in elems:
            indexarr += [counter,counter]
            counter += 1
        else:
            indexarr += [counter,counter+1]
            counter += 2
    operator = np.einsum(temp, indexarr)

    length = len(operator.shape)
    halflength = length//2

    if elements != []:
        operator = np.transpose(operator,[length-2,length-1]+list(range(0,length-2))) #cyclic permutation for toqito

    reorder = np.transpose(operator,[2*n for n in range(0,halflength)]+[2*n+1 for n in range(0,halflength)])
    return np.reshape(reorder, (np.prod(operator.shape[::2]), np.prod(operator.shape[1::2])))


def calcS(densityM, partiesToTraceOut):
    Ntot=int(np.log2(densityM.shape[0]))
    redRho = ptrace_rho(densityM, partiesToTraceOut)
    eVals = np.linalg.eigvalsh(redRho)
    eVals = eVals[np.where(eVals>1e-12)]
    S = -sum(eVals*np.log2(eVals))
    return S


def calcPurity(densityM):
    purity = np.trace(densityM@densityM)
    return purity

def flatten_tensor_op(operator):
    length = len(operator.shape)
    halflength = length//2
    reorder = np.transpose(operator,[2*n for n in range(0,halflength)]+[2*n+1 for n in range(0,halflength)])
    return np.reshape(reorder, (np.prod(operator.shape[::2]), np.prod(operator.shape[1::2])))

#only works for tensor products of 2-state systems
def unflatten_tensor_op(operator):
    halflength = int(np.log2(operator.shape[0]))
    reorder = np.reshape(operator, (2,)*(2*halflength))
    return np.transpose(reorder, [n//2+halflength*(n%2) for n in range(0,halflength*2)])

def calcNegativity(densitymatx, transpose_indices):
    unflatrho = unflatten_tensor_op(densitymatx)
    transposer = list(range(len(unflatrho.shape)))
    for i in transpose_indices:
        transposer[2*i+1],transposer[2*i] = transposer[2*i],transposer[2*i+1]
    transposed = flatten_tensor_op(np.transpose(unflatrho,tuple(transposer)))
    return (np.trace(linalg.sqrtm(transposed.T.conj()@transposed))-1)/2


def combinedRoutine(initRho, Nqubit, rate, probMsm, dt, jumpOp, steps, debug = False):
    rho = sparse.csr_matrix(reduce(sparse.kron, [initRho]*Nqubit))

    startTime = time.time()

    entropy1 = np.zeros(steps)
    entropy2 = np.zeros(steps)
    entropy_tot = np.zeros(steps)
    mutual_info = np.zeros(steps)
    purity = np.zeros(steps)
    negativity = np.zeros(steps)

    for s in range(steps):
        allKrausOps = mkGlobalKrausOps(s, Nqubit, rate, dt, jumpOp, debug = False)
        rho = evolveKrausU(allKrausOps, rho)
        rho = mkS_zMsm(probMsm, rho)

        entropy1[s] = calcS(rho.toarray(), partiesToTraceOut = list(range(0, Nqubit//2-1)))
        entropy2[s] = calcS(rho.toarray(), partiesToTraceOut = list(range(Nqubit//2, Nqubit)))
        entropy_tot[s] = calcS(rho.toarray(), partiesToTraceOut = [])
        mutual_info[s] = entropy1[s] + entropy2[s] - entropy_tot[s]
        purity[s] = np.real(calcPurity(rho.toarray()))
        negativity[s] = calcNegativity(rho.toarray(), [n for n in range(Nqubit//2)])

    return entropy1, entropy2, entropy_tot, mutual_info, purity, negativity

