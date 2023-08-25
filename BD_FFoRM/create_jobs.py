import os
import sys
import numpy as np

# This code creates prandtl submit files to generate training data by
# 1) BD simulations to get I(q,t), uu(t), uuuu(t)
# 2) Batchelor's expression for the stress uu, uuuu -> sigma
# 3) Preprocess data from lab frame to co-rotating phase and phase-aligned scattering intensity
step_size = 0.05
# Sweep parameter - flow rate ratio controls flow type, ext vs shear
FRR = np.arange(-0.7,1.00001,step_size)
# FRR = [0.05]
FRR = FRR[::2]
nF = len(FRR)
nQ = 1
# Boundary conditions - inlet flow rates
# Q2 = np.zeros((nF,nQ))
Q2max = 1.0e-6
Q2min = 1.0e-8
# Q2[:] = np.linspace(Q2min,Q2max,nQ)
Q2arr = np.linspace(Q2min,Q2max,10)
# Q2 = Q2arr[1::2]
# Q2 = np.array([7.8e-7])
Q2 = Q2arr[1:]
print(Q2)
# Q2[:] = [1.2e-7]
Q2 = Q2/1.0e-7
strm = np.arange(0,10)
avg_win = 0.01
nqi = 100
qlim = 0.1
# nthph = 300
nthph = 200
npsi = 20
ntraj = 100000
# nodes = [7,10,11,12,13,14]
# ppn = [16,8,8,8,8,8]
nodes = [40]
ppn = [20]
nnodes = len(nodes)
load = 0

count_threads = 0
node = 0
for f in FRR:
	for q in Q2:
		for st in strm:
			print(f,q,st)
			# jobname = 'BDFF_FRR_%.2f_Q2_%.2f_strm_%d'%(f,q,st)
			jobname = 'IqRot_FRR_%.2f_Q2_%.2f_strm_%d'%(f,q,st)
			jobfname = 'qsub/'+jobname+'.qsub'
			jobfile = open(jobfname,'w+')

			jobfile.write("#!/bin/bash\n\n")
			jobfile.write("#PBS -N ."+jobname+"\n")
			# jobfile.write("#PBS -l nodes=prandtl40:ppn=1\n")
			jobfile.write("#PBS -l nodes=prandtl%d:ppn=1\n"%nodes[node])
			jobfile.write("#PBS -k oe\n")
			jobfile.write("#PBS -j oe\n")
			jobfile.write("#PBS -l walltime=10:00:00:00\n")
			jobfile.write("#PBS -q parallel\n")
			jobfile.write("cd $PBS_O_WORKDIR\n")
			jobfile.write("./ups.out %.2f %.2f %d %d %d\n"%(f,q,st,nthph,ntraj))
			# jobfile.write("./Pq_pp.py %.2f %.2f %d %f %d %.2f %d %d %d\n"%(f,q,st,avg_win,nqi,qlim,nthph,npsi,ntraj))
			# jobfile.write("./PDF_Pq.py %.2f %.2f %d %f %d %.2f %d %d\n"%(f,q,st,avg_win,nqi,qlim,nthph,ntraj))
			jobfile.write("./stress_ups.py %.2f %.2f %d %.2f %d\n"%(f,q,st,avg_win,ntraj))
			jobfile.write("./Pq_rot_l2c_ups.py --FRR %.2f --Q2 %.2f --strm %d --load %d\n"%(f,q,st,load))
			jobfile.write("exit 0\n\n")
			count_threads = count_threads + 1
			if count_threads >= ppn[node]:
				count_threads = 0
				node = node + 1
				if node == nnodes:
					node = 0
