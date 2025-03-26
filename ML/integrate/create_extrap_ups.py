import os
import sys
import numpy as np

# flowType = float(sys.argv[1])
# strain_max = float(sys.argv[2])

fTlist = np.array([0.0,1.0])
Pelist = np.logspace(0,np.log10(50.0),10)
low = np.array([0.1,0.5])
Pelist = np.append(low,Pelist)
print(Pelist)
avg_win = np.array([1.0,1.0,0.5,0.5,0.5,0.1,0.1,0.1,0.1,0.01,0.01,0.01])

# avg_win = 0.01
nqi = 100
qlim = 0.1
# nthph = 300
nthph = 200
npsi = 20
ntraj = 100000
nodes = [8,9,10,11,13,14]
ppn = [14,6,6,6,6,6]
# nodes = [43,44]
# ppn = [6,4]
nnodes = len(nodes)
load = 0

alpha = np.array([0.0,1.0e-3])
# alpha = np.array([0.0,1.0e-4,1.0e-3,1.0e-2])
traj = np.arange(0,10)
# alpha = np.array([0.001])
# traj = np.array([11])
arch = 0
data_size = 106663
wdir = 'l2c_ups_smooth'
dh = 5
ae_model = 1
t = 10
movie = 1

count_threads = 0
node = 0
count = 0

for flowType in fTlist:
	count = 0
	for Pe in Pelist:
		for al in alpha:
			for tr in traj:
				# print(tr)
				jobname = 'FFextrap_dh%d_t%d_alpha%f_a%d_tr%d_FT%.2f_Pe%.2e'%(dh,t,al,arch,tr,flowType,Pe)
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
				jobfile.write("./pred_ups_test.py --batch_time %d --dh %d --alpha %f --dir %s --ae_model %d --data_size %d --arch %d --traj %d --movie %d --flowType %.2f --Pe %f --dt %f\n"%(t,dh,al,wdir,ae_model,data_size,arch,tr,movie,flowType,Pe,avg_win[count]))
				jobfile.write("exit 0\n\n")
				count_threads = count_threads + 1
				if count_threads >= ppn[node]:
					count_threads = 0
					node = node + 1
					if node == nnodes:
						node = 0

		count += 1
