import os
import sys
import numpy as np

flowType = float(sys.argv[1])
# strain_max = float(sys.argv[2])

Pelist = np.logspace(0,np.log10(50.0),10)
low = np.array([0.1,0.5])
Pelist = np.append(low,Pelist)
print(Pelist)
avg_win = np.array([1.0,1.0,0.5,0.5,0.5,0.1,0.1,0.1,0.1,0.01,0.01,0.01])
strain_max = np.zeros(len(Pelist)) + 20.0
strain_max[0] = 1.0
strain_max[1] = 5.0
# strain_max = 100.0

# avg_win = 0.01
nqi = 100
qlim = 0.1
# nthph = 300
nthph = 200
npsi = 20
ntraj = 100000
nodes = [48]
ppn = [24]
nnodes = len(nodes)
load = 0

count_threads = 0
node = 0
count = 0

for Pe in Pelist:
	# print(Pe)
	# jobname = 'BDFF_FRR_%.2f_Q2_%.2f_strm_%d'%(f,q,st)
	jobname = 'Iqgen_FT_%.2f_Pe_%.2e'%(flowType,Pe)
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
	jobfile.write("./create_strm_steady.py %.2f %.2e %f\n"%(flowType,Pe,strain_max[count]))
	jobfile.write("./run.out %.2f %.2f %d %d %.3f\n"%(flowType,Pe,nthph,ntraj,avg_win[count]))
	# jobfile.write("./Pq_pp.py %.2f %.2f %d %f %d %.2f %d %d %d\n"%(f,q,st,avg_win,nqi,qlim,nthph,npsi,ntraj))
	# jobfile.write("./PDF_Pq.py %.2f %.2e %f %d %.2f %d %d %d\n"%(flowType,Pe,avg_win,nqi,qlim,nthph,ntraj,load))
	jobfile.write("./Pq_rot_l2c.py --flowType %.2f --Pe %.2e --avg_win %.3f --load %d\n"%(flowType,Pe,avg_win[count],load))
	jobfile.write("exit 0\n\n")
	count_threads = count_threads + 1
	if count_threads >= ppn[node]:
		count_threads = 0
		node = node + 1
		if node == nnodes:
			node = 0
	count += 1
