import numpy as np
import re
import subprocess
import sys
import datetime
import os
def talapas_qsub(scriptname):
  cmd = ['sbatch',scriptname]
  done = False
  while not done:
    with subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,bufsize=1) as p:
      output = ""
      for stuff in p.stdout:
        output = output+ stuff.decode("utf-8")
    if not re.search("error" ,stuff.decode("utf-8")):
      done = True
    print(output,end="")
    if not done:
      print("RESUBMITTING")

def simple_talapas_job(jobstring="",scriptname="",jobname="",time="24",q="short"):
  today = str(datetime.date.today())
  logdir = "/gpfs/home/yl244/logs/"
  with open(scriptname,'w') as f:
    f.write('#!/bin/bash\n')
    f.write('#SBATCH --job-name='+ jobname+ '\n')
    f.write('#SBATCH --output=' + logdir + jobname+'.out\n')
    f.write('#SBATCH --error=' + logdir + jobname+'.err\n')
    f.write('#SBATCH --partition=' + q + "\n")
    f.write('#SBATCH --time='+time+':00:00\n')
    f.write('#SBATCH --mem-per-cpu=100000\n')
    f.write('#SBATCH --nodes=1\n')
    f.write('#SBATCH --ntasks-per-node=1\n')
    f.write('#SBATCH --requeue\n')
    f.write('#SBATCH --account=ajliu\n')
    f.write('')
    #f.write('#SBATCH --gres=gpu:1\n')
    f.write(jobstring)
    f.close()
  talapas_qsub(scriptname)

#numjobs =np.arange(0,50) #np.arange(5,8)
numInd=[0,1] # jump operator index (S_Z, or S_-)
probMsm = 0.1
rate = 1e-5
Nqubits = 10
ens_N = 20 # Make sure your trajectories finish so the files are saved before the time above run out

# You can make more for loops in the vars
for jumpInd in numInd:
    jobstring = "python3 ~/cathyTest/rc.py " + str(probMsm) + " " + str(jumpInd) + " " + str(rate) + " " + str(Nqubits) + " " + str(ens_N)
    simple_talapas_job(jobstring=str(jumpInd),scriptname="rc.srun",jobname="rc"+str(jumpInd))

#cancel jobs: for i in `seq 100 200`; do scancel $i; done
