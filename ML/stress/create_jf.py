import os
import sys
import numpy as np

# arch = 200
load = 0
t_sp = 1
dPCA = np.arange(1,31)
arch = 2
# dd = np.array([3])
model = range(1,10)
reg = 0.0001
load = 0
# node = int(sys.argv[1])

nodes = [38,39,40,41]
ppn = [20,20,20,20]
nnodes = len(nodes)

count_threads = 0
node = 0
for dp in dPCA:
	for mod in model:
		jobname = 'str_recon_dPCA%d_mod%d_a%d'%(dp,mod,arch)
		jobfname = 'qsub/'+jobname+'.qsub'
		jobfile = open(jobfname,'w+')

		jobfile.write("#!/bin/bash\n\n")
		jobfile.write("#PBS -N ."+jobname+"\n")
		# jobfile.write("#PBS -l nodes=prandtl%d:ppn=1\n"%node)
		jobfile.write("#PBS -l nodes=prandtl%d:ppn=1\n"%nodes[node])
		jobfile.write("#PBS -k oe\n")
		jobfile.write("#PBS -j oe\n")
		jobfile.write("#PBS -l walltime=5:00:00:00\n")
		jobfile.write("#PBS -q parallel\n")
		jobfile.write("cd $PBS_O_WORKDIR\n")
		jobfile.write("./stress_PCA.py --dPCA %d --arch %d --model %d --reg %.6f --load %d\n"%(dp,arch,mod,reg,load))
		jobfile.write("exit 0\n\n")
		count_threads = count_threads + 1
		if count_threads >= ppn[node]:
			count_threads = 0
			node = node + 1
			if node == nnodes:
				node = 0
