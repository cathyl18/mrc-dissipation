-*- coding: utf-8 -*-
import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
cl.PYOPENCL_COMPILER_OUTPUT=1


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

class Ising: #initializes offset lattices A,B for d dimensional cubic ising problem (size N^d)

   def __init__(self,N,J,dimension=2, oclString = "Ising.cl"):
       self.N, self.J = N,J*2
       self.D = dimension
       self.rng = np.random.default_rng()
       self.context = cl.create_some_context()
       self.queu = cl.CommandQueue(self.context)

       #Load .cl file and create compute program
       file = open(oclString, 'r')
       kernelString = "".join(file.readlines())
       self.program = cl.Program(self.context,kernelString).build()

       self.A = np.random.choice([-1,1], size=int(N**dimension/2)).astype(np.int32)
       self.B = np.random.choice([-1,1], size=int(N**dimension/2)).astype(np.int32)
       self.C = np.random.choice([-1,1], size=int(N**dimension/2)).astype(np.int32)

       self.mem_A = cl.Buffer(self.context, cl.mem_flags.KERNEL_READ_AND_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf = self.A)
       self.mem_B = cl.Buffer(self.context, cl.mem_flags.KERNEL_READ_AND_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf = self.B)
       self.mem_C = cl.Buffer(self.context, cl.mem_flags.KERNEL_READ_AND_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf = self.C)

   def reset(self):
       N, D = self.N, self.D
       self.A = np.random.choice([-1,1], size=int(N**D/2)).astype(np.int32)
       self.B = np.random.choice([-1,1], size=int(N**D/2)).astype(np.int32)
       self.C = np.random.choice([-1,1], size=int(N**D/2)).astype(np.int32)

   def mag(self): #return mag, which can be used to estimate T_c
       total_mag = np.sum(self.A)+np.sum(self.B)
       return total_mag

   def Energy(self):#Not Tested, maybe doesn't work. Need to check dtypes in kernel call
       A,B,C,mem_A,mem_B,mem_C = self.A, self.B, self.C, self.mem_A, self.mem_B, self.mem_C
       context,queu = self.context, self.queu
       N,J = self.N,self.J

       currentE = np.int32(0)
       mem_E = cl.Buffer(context, cl.mem_flags.KERNEL_READ_AND_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf = currentE)

       inter_E = self.program.Energy
       inter_E.set_scalar_arg_dtypes([None,None,None,np.uint32])
       inter_E(queu,np.int32,None,mem_A,mem_B, mem_E, N)
       queu.finish()
       cl.enqueue_copy(queu,currentE,mem_E)
       return currentE*J #maybe missing a factor
##############################################################################
class Ising2D(Ising):

   def compare(self,which = 0,Tau = 0): #change A,B Globally. Tau is proportional to a unitless Temp

       A,B,C, mem_A,mem_B,mem_C = self.A, self.B, self.C, self.mem_A, self.mem_B,self.mem_C
       context,queu = self.context, self.queue


       N,Jh, Jd, J123 = self.N,self.J

       first_comp = np.float32(np.exp(-J*2/Tau)) #exponentials calculated only once is better
       Rand = self.rng.random(int(N**2/2),dtype = np.float32)
       mem_Rand = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = Rand)

       compare = self.program.Comparison
       compare.set_scalar_arg_dtypes([None,None,None,np.uint32,np.uint32,np.float32])
       compare(queu,A.shape,None,mem_A,mem_B,mem_Rand, N, np.int32(which), first_comp)

       queu.finish()
       if(which == 0):
           cl.enqueue_copy(queu,A,mem_A)
       elif(which ==1):
           cl.enqueue_copy(queu,B,mem_B)

       else:
           cl.enqueue_copy(queu,C,mem_C)

   def combine(self):#TODO: write as kernel to speed up animation
       A,B,C = self.A, self.B, self.C
       N = self.N
       outArr = np.zeros((N,N))
       for i in range(0,int(N),2):
           k = int(i/2)
           outArr[i][::2] = A[N*k:N*k+N][::2]
           outArr[i+1][1::2] = A[N*k:N*k+N][1::2]

           outArr[i][1::2] = B[N*k:N*k+N][1::2]
           outArr[i+1][::2] = B[N*k:N*k+N][::2]
       return outArr
