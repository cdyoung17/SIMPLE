#ifndef Main_h
#define Main_h

// Header files for various libraries used in the code

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
// #include <mkl.h>
#include <string.h>
#include <time.h>
#include <getopt.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

// Constants for the random number generator

#define IM1 2147483563
#define IM2 2147483399
#define AM (1.0/IM1)
#define IMM1 (IM1-1)
#define IA1 40014
#define IA2 40692
#define IQ1 53668
#define IQ2 52774
#define IR1 12211
#define IR2 3791
#define NTAB 32
#define EPS 1.2e-7
#define RNMX (1.0-EPS)
#define NDIV (1+IMM1/NTAB)

// User defined macros

#define max(a,b) ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a > _b ? _a : _b; })
#define min(a,b) ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a < _b ? _a : _b; })

long initRan();
// float ran1(long *idum); // Uniform RNG [0,1) from numerical recipes textbook
// float gasdev();
float ran1(long *idum); // Uniform RNG [0,1) from numerical recipes textbook
// float ran1(long idum); // Uniform RNG [0,1) from numerical recipes textbook
float gasdev();
// float ran1(unsigned long *idum); // Uniform RNG [0,1) from numerical recipes textbook
// float gasdev(unsigned long *idum);
double* cyl_pq(int nq,double *qx,double *qy,double *qz,double Rc,double L,double gamm,double beta,double alpha);
double cyl_pqi(double qx,double qy,double qz,double Rc,double L,double gamm,double beta,double alpha);

int i,j,k,l,m,nt,traj_time_print,result,avg_count;
double wall_time;
double FT,Pe,FRR,Q2,Dr,temp,temp2,temp3,dt,tend,p,t_elap,sum,umag,phi0,theta0,sqrtm8,thetai,phii;
int strm,traj_s,traj_e,ntraj,flow_steps,count,t,tmax,f_step;
char flow_input[100],output[100],outdir[200],odirbase[200];
FILE *flowfile,*outfile;

double *tf,*flow_type,*flow_mag,**Gd,**ut,*ui,**thetat,**phit,**uu_avg,**u4_avg;
long *idum; // RNG seed
// unsigned long *idum;
// long idum;
double eye[9],u[3],un[3],uu[9],dF[3],dW[3],R[3],eye_m_uu_x_Gd[9],sqrtm[9],A[9];

int samp_rate,steps_p_win,n_win,nq,qxnum,qynum,nang,win,pbin,tbin,ycount,nqi;
double qlim,avg_win,L,Rc,volfrac,dsld,bkgd,qxmin,qxmax,qxstep,qymin,qymax,qystep,bin_size,bin_area,phimod,dphi,dtheta,gamm,Pi;
double **P,*Pj,*qxlist,*qylist,***PDF,*qx,*qy,*qz;
long int *hcount,***h;

#endif // Main_h
