import sys
import numpy as np

dPCA = 50
wtdc = np.array([1e-5,1e-4,1e-3,5e-3,1e-2])
nlin = np.array([2,3,4,5])
niters = 2000
model = np.arange(0,3)
train = 1
downs = 5
optim = 'AdamW'
nodes = [35,37,38]
ppn = [14,14,10]
nnodes = len(nodes)
data_size = 106663
dh = 5
# ae_model = 1

count_threads = 0
node = 0
count = 0
# for t in batch_time:
for wt in wtdc:
	for nl in nlin:
		for mod in model:
			# print(tr)
			jobname = 'FFAE_%s_w%.2e_l%d_mod%d'%(optim,wt,nl,mod)
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
			jobfile.write("./IRMAE.py --train %d --downs %d --model %d --wtdc %e --nlin %d --optim %s\n"%(train,downs,mod,wt,nl,optim))
			jobfile.write("exit 0\n\n")
			count_threads = count_threads + 1
			if count_threads >= ppn[node]:
				count_threads = 0
				node = node + 1
				if node == nnodes:
					node = 0

	count += 1


