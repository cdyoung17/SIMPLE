#! /home/cyoung37/anaconda3/bin/python3

import os
import sys
import math
import numpy as np
from distutils.dir_util import copy_tree

# This code generates the submit files for simulating the FFoRM geometry with a Newtonian fluid at varying operating conditions

def write_BCs(Ustr,Q1,Q2):
	with open(Ustr,'w') as Ufile:
		Ufile.write('FoamFile\n')
		Ufile.write('{\n')
		Ufile.write('    version     2.0;\n')
		Ufile.write('    format      ascii;\n')
		Ufile.write('    class       volVectorField;\n')
		Ufile.write('    object      U;\n')
		Ufile.write('}\n')
		Ufile.write('dimensions      [0 1 -1 0 0 0 0];\n')
		Ufile.write('internalField   uniform (0 0 0);\n')
		Ufile.write('boundaryField\n')
		Ufile.write('{\n')
		Ufile.write('    walls\n')
		Ufile.write('    {\n')
		Ufile.write('        type            fixedValue;\n')
		Ufile.write('        value           uniform (0 0 0);\n')
		Ufile.write('    }\n')
		Ufile.write('    inlet_westbottom\n')
		Ufile.write('    {\n')
		Ufile.write('        type            flowRateInletVelocity;\n')
		Ufile.write('        volumetricFlowRate    	%.4e;\n'%Q2)
		Ufile.write('        value           $internalField;\n')
		Ufile.write('    }\n')
		Ufile.write('    inlet_easttop\n')
		Ufile.write('    {\n')
		Ufile.write('        type            flowRateInletVelocity;\n')
		Ufile.write('        volumetricFlowRate    	%.4e;\n'%Q2)
		Ufile.write('        value           $internalField;\n')
		Ufile.write('    }\n')
		Ufile.write('    outlet_northleft\n')
		Ufile.write('    {\n')
		Ufile.write('        type            flowRateInletVelocity;\n')
		Ufile.write('        volumetricFlowRate    	%.4e;\n'%Q1)
		Ufile.write('        value           $internalField;\n')
		Ufile.write('    }\n')
		Ufile.write('    outlet_southright\n')
		Ufile.write('    {\n')
		Ufile.write('        type            flowRateInletVelocity;\n')
		Ufile.write('        volumetricFlowRate    	%.4e;\n'%Q1)
		Ufile.write('        value           $internalField;\n')
		Ufile.write('    }\n')
		Ufile.write('    pressure_northright\n')
		Ufile.write('    {\n')
		Ufile.write('        type            zeroGradient;')
		Ufile.write('    }\n')
		Ufile.write('    pressure_southleft\n')
		Ufile.write('    {\n')
		Ufile.write('        type            zeroGradient;')
		Ufile.write('    }\n')
		Ufile.write('    pressure_westtop\n')
		Ufile.write('    {\n')
		Ufile.write('        type            zeroGradient;')
		Ufile.write('    }\n')
		Ufile.write('    pressure_eastbottom\n')
		Ufile.write('    {\n')
		Ufile.write('        type            zeroGradient;')
		Ufile.write('    }\n')
		Ufile.write('    frontAndBack\n')
		Ufile.write('    {\n')
		Ufile.write('        type            empty;\n')
		Ufile.write('    }\n')
		Ufile.write('}\n')
	Ufile.close()
	# exit()
def write_qsub(FRR,Q2):

	jobname = 'FFoRM_FRR_%.2f_Q2_%.2f'%(FRR,Q2/1.0e-07)
	jobfname = 'qsub/'+jobname+'.qsub'
	jobfile = open(jobfname,'w+')

	jobfile.write("#!/bin/sh -f\n\n")
	jobfile.write("#PBS -N ."+jobname+"\n")
	jobfile.write("#PBS -l nodes=prandtl40:ppn=5\n")
	jobfile.write("#PBS -k oe\n")
	jobfile.write("#PBS -j oe\n")
	jobfile.write("#PBS -l walltime=10:00:00:00\n")
	jobfile.write("#PBS -q parallel\n")
	jobfile.write("cd $PBS_O_WORKDIR\n")
	jobfile.write("cd FRR_%.2f/Q2_%.2f\n"%(FRR,Q2/1.0e-7))
	jobfile.write("docker container run --rm -v $PWD:/data -w /data myrheotool:v5 ./Allrun\n")
	# jobfile.write("chmod -R 777 *\n\n")
	jobfile.write("exit 0\n\n")

if __name__=='__main__':

	dir1 = 'template'
	step_size = 0.05
	# Sweep parameter - flow rate = Q1/Q2 ratio controls flow type, ext vs shear
	FRR = np.arange(-0.7,1.00001,step_size)
	# ind_remove = [13,14,15]
	# FRR = np.delete(FRR,ind_remove)
	# print(FRR)
	# exit()
	nF = len(FRR)
	nQ = 10
	# Boundary conditions - inlet and outlet flow rates
	Q1 = np.zeros((nF,nQ))
	Q2 = np.zeros((nF,nQ))
	Q2max = 1.0e-6
	Q2min = 1.0e-8
	Q2[:] = np.linspace(Q2min,Q2max,nQ)
	# print(Q2)
	# exit()
	# Increase total flow rate at lower values of FRR to maintain similar strain rate at stagnation pt
	# 5x increase in Q2 from FRR = 1.0 to FRR = -0.7 used in Corona et al. Phys. Rev. Mat. 2022
	# Q2[-1] = 8.33333e-7
	# Q2[0] = Q2[-1]*5
	# Linear interpolation to find other total flow rates
	# Q2[1:-1] = Q2[0] + (FRR[1:-1] - FRR[0])*(Q2[-1] - Q2[0])/(FRR[-1] - FRR[0])

	for i in range(nF):
		Q1[i] = -Q2[i]*FRR[i]

	# print(Q1)
	# Create directories. Copy contents and rewrite 0/U for new BC
	for i in range(nF):
		for j in range(nQ):
			# print(i,FRR[i],Q1[i],Q2[i])
			dir2 = 'FRR_%.2f/Q2_%.2f'%(FRR[i],Q2[i,j]/1.0e-7)
			print('i %d FRR %.5f Q1 %.4e Q2 %.4e dir2 %s'%(i,FRR[i],Q1[i,j]/1.0e-7,Q2[i,j]/1.0e-7,dir2))
			# if(os.path.exists(dir2)==0):
				# os.makedirs(dir2)
			copy_tree(dir1,dir2)
			Ustr = dir2+'/0/U'
			write_BCs(Ustr,Q1[i,j],Q2[i,j])
			write_qsub(FRR[i],Q2[i,j])
