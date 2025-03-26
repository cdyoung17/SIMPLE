import os
import sys
import numpy as np

dPCA = 50
# batch_time = np.array([1,2,3,5,10])
batch_time = np.array([1,2,3])
# niters = 4000
niters = 40000/batch_time
alpha = 0.0
arch = 0
traj = np.arange(3,6)
load = 0
nodes = [44]
ppn = [9]
nnodes = len(nodes)
data_size = 95119
wdir = 'QpE1E2_Ima'
dh = 5

count_threads = 0
node = 0
count = 0
for t in batch_time:
	for tr in traj:
		print(tr)
		jobname = 'FFNODE_dh%d_t%d_a%d_tr%d'%(dh,t,arch,tr)
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
		jobfile.write("./NODE_AE_Qphase2.py --batch_time %d --dh %d --dir %s --data_size %d --niters %d --arch %d --traj %d --load %d\n"%(t,dh,wdir,data_size,niters[count],arch,tr,load))
		jobfile.write("exit 0\n\n")
		count_threads = count_threads + 1
		if count_threads >= ppn[node]:
			count_threads = 0
			node = node + 1
			if node == nnodes:
				node = 0

	count += 1
