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
nodes = [44]
ppn = [24]
nnodes = len(nodes)
movie = 1
flowType = np.array([0.0,1.0,1.0])
Pe = np.array([10.0,5.1,20.0])

data_size = 95181
wdir = 'l2c'
dh = 8
ae_model = 2

# data_size = 95119
# wdir = 'l2c_smooth'
# dh = 5
# ae_model = 0

count_threads = 0
node = 0
count = 0
inds = 0

for inds in range(len(flowType)):
	f = flowType[inds]
	q = Pe[inds]
	# for t in batch_time:
	for al in alpha:
		for tr in traj:
			# print(tr)
			jobname = 'FFextrap_dh%d_t%d_alpha%f_a%d_tr%d_FT%.2f_Pe%.2f'%(dh,t,al,arch,tr,f,q)
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
			jobfile.write("./l2c_pred_test.py --batch_time %d --dh %d --alpha %f --dir %s --ae_model %d --data_size %d --arch %d --traj %d --movie %d --flowType %.2f --Pe %f\n"%(t,dh,al,wdir,ae_model,data_size,arch,tr,movie,f,q))
			jobfile.write("exit 0\n\n")
			count_threads = count_threads + 1
			if count_threads >= ppn[node]:
				count_threads = 0
				node = node + 1
				if node == nnodes:
					node = 0

		count += 1
