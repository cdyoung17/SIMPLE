import os
import sys
import numpy as np

dPCA = 50
# batch_time = np.array([1,2,3,5,10])
# batch_time = np.array([10])
t = 10
# niters = 4000
# niters = 40000/batch_time
alpha = np.array([0.0,1.0e-4,1.0e-3,1.0e-2])
arch = 0
traj = np.arange(10,13)
load = 0
nodes = [48]
ppn = [24]
nnodes = len(nodes)

data_size = 95119
wdir = 'l2c_smooth_sin'
dh = 5
ae_model = 0

# data_size = 95181
# wdir = 'l2c'
# dh = 8
# ae_model = 2

step_size = 0.05
FRR = np.arange(-0.7,1.00001,step_size)
FRR = FRR[::2]
print(FRR)
nF = len(FRR)
nQ = 1
Q2max = 1.0e-6
Q2min = 1.0e-8
Q2arr = np.linspace(Q2min,Q2max,10)
# Q2 = np.array([7.8e-7])
Q2 = Q2arr[2::2]
Q2 = Q2/1.0e-7
print(Q2)
strm = np.arange(0,10)
movie = 0
# FRR = [1.0]
# Q2 = [10.0]
# strm = [9]

# movie = 1
# FRR = np.array([-0.7,-0.5,0.0,0.3,0.6,1.0])
# # Q2 = np.array([3.4,3.4,3.4,3.4,3.4])
# # strm = np.array([4,4,4,4,4])
# # Q2 = np.zeros(len(FRR)) + 10.0
# Q2 = np.array([1.2,3.4,10.0])
# # strm = np.zeros(len(FRR),dtype=int) + 4
# st = 4

count_threads = 0
node = 0
count = 0
inds = 0

for f in FRR:
	for q in Q2:
		for st in strm:
			for al in alpha:
				for tr in traj:
					# print(tr)
					jobname = 'FFpred_dh%d_t%d_alpha%f_a%d_tr%d_f%.2f_q%.2f_st%d'%(dh,t,al,arch,tr,f,q,st)
					jobfname = 'qsub/'+jobname+'.qsub'
					jobfile = open(jobfname,'w+')

					jobfile.write("#!/bin/bash\n\n")
					jobfile.write("#PBS -N ."+jobname+"\n")
					jobfile.write("#PBS -l nodes=prandtl%d:ppn=1\n"%nodes[node])
					jobfile.write("#PBS -k oe\n")
					jobfile.write("#PBS -j oe\n")
					jobfile.write("#PBS -l walltime=10:00:00:00\n")
					jobfile.write("#PBS -q parallel\n")
					jobfile.write("cd $PBS_O_WORKDIR\n")
					jobfile.write("./l2c_pred_strm.py --batch_time %d --dh %d --alpha %f --dir %s --ae_model %d --data_size %d --arch %d --traj %d --movie %d --FRR %.2f --Q2 %.2f --strm %d\n"%(t,dh,al,wdir,ae_model,data_size,arch,tr,movie,f,q,st))
					jobfile.write("exit 0\n\n")
					count_threads = count_threads + 1
					if count_threads >= ppn[node]:
						count_threads = 0
						node = node + 1
						if node == nnodes:
							node = 0

	# count += 1
