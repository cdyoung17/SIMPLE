import sys
import numpy as np

dPCA = 50
# batch_time = np.array([1,2,3,5,10])
# batch_time = np.array([10])
t = 10
# niters = 8000
niters = 80000/t
# alpha = 0.01
alpha = np.array([0.001])
arch = 0
traj = np.arange(8,10)
load = 0
nodes = [43]
ppn = [1]
nnodes = len(nodes)
data_size = 106663
wdir = 'l2c_ups_smooth'
dh = 5
ae_model = 1

count_threads = 0
node = 0
count = 0
# for t in batch_time:
for al in alpha:
	for tr in traj:
		print(tr)
		jobname = 'FFNODE_dh%d_t%d_a%d_alpha%.6f_tr%d'%(dh,t,arch,al,tr)
		jobfname = 'qsub/'+jobname+'.qsub'
		jobfile = open(jobfname,'w+')

		jobfile.write("#!/bin/bash\n\n")
		jobfile.write("#PBS -N ."+jobname+"\n")
		jobfile.write("#PBS -l nodes=prandtl%d:ppn=2\n"%nodes[node])
		jobfile.write("#PBS -k oe\n")
		jobfile.write("#PBS -j oe\n")
		jobfile.write("#PBS -l walltime=10:00:00:00\n")
		jobfile.write("#PBS -q parallel\n")
		jobfile.write("cd $PBS_O_WORKDIR\n")
		jobfile.write("./NODE_AE_l2c_ups.py --batch_time %d --ae_model %d --dh %d --dir %s --data_size %d --niters %d --arch %d --alpha %f --traj %d --load %d\n"%(t,ae_model,dh,wdir,data_size,niters,arch,al,tr,load))
		jobfile.write("exit 0\n\n")
		count_threads = count_threads + 1
		if count_threads >= ppn[node]:
			count_threads = 0
			node = node + 1
			if node == nnodes:
				node = 0

	count += 1


