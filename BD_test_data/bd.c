#include "main.h" // function definitions and global variable delcarations are here

int main(int argc,const char *argv[]){
	sscanf(argv[1],"%lf",&FT);
	sscanf(argv[2],"%lf",&Pe);
	sscanf(argv[3],"%d",&nang);
	sscanf(argv[4],"%d",&ntraj);
	sscanf(argv[5],"%lf",&avg_win);

	sprintf(flow_input,"Gamma_input/FT%.2f_Pe%.1e.txt",FT,Pe);
	printf("input file %s\n",flow_input);
	flowfile = fopen(flow_input,"r");
	fscanf(flowfile,"Time ftype fmag G_00 G_01 G_02 G_10 G_11 G_12 G_20 G_21 G_22\n");
	fscanf(flowfile,"---------------------\n");
	flow_steps = 0;
	while(!feof(flowfile)){
		fscanf(flowfile,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf \n",&temp,&temp,&temp,&temp,&temp,&temp,&temp,&temp,&temp,&temp,&temp,&temp);
		flow_steps++;
	}
	fclose(flowfile);
	// exit(1);
	printf("flow_steps %d\n",flow_steps);
	tf = calloc(flow_steps,sizeof(double));
	flow_type = calloc(flow_steps,sizeof(double));
	flow_mag = calloc(flow_steps,sizeof(double));
	Gd = calloc(flow_steps,sizeof(double));
	for(i=0;i<flow_steps;++i){
		Gd[i] = calloc(9,sizeof(double));
	}
	flowfile = fopen(flow_input,"r");
	fscanf(flowfile,"Time ftype fmag G_00 G_01 G_02 G_10 G_11 G_12 G_20 G_21 G_22\n");
	fscanf(flowfile,"---------------------\n");
	count = 0;
	while(!feof(flowfile)){
		// printf("count %d\n",count);
		fscanf(flowfile,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf \n",&tf[count],&flow_type[count],&flow_mag[count],&Gd[count][0],&Gd[count][1],&Gd[count][2],&Gd[count][3],&Gd[count][4],&Gd[count][5],&Gd[count][6],&Gd[count][7],&Gd[count][8]);
		// printf("tf %lf ftp %lf G %lf G00 %lf G01 %lf\n",tf[count],flow_type[count],flow_mag[count],Gd[count][0],Gd[count][1]);
		count++;
	}
	fclose(flowfile);
	// for(i=0;i<flow_steps;++i){
		// printf("step %d t %lf ftp %f G %f G00 %f\n",i,tf[i],flow_type[i],flow_mag[i],Gd[i][0]);
	// }
	// exit(1);
	// Create output directory
	getcwd(outdir,4096);
	printf("output dir %s\n",outdir);
	// strcat(outdir,"/output/FRR_%.2f/Q2_%.2f/strm_%d/",FRR,Q2,strm);
	sprintf(output,"/output/FT%.2f_Pe%.1e",FT,Pe);
	strcat(outdir,output);
	result = mkdir(outdir, 0777);
	// sprintf(output,"/Pq_nt%d",ntraj);
	// strcat(outdir,output);
	// result = mkdir(outdir, 0777);
	// sprintf(output,"/h_na_%d_nq_%d_qlim_%.2f.txt",nang,nqi,qlim);
	sprintf(output,"/h_na_%d_%d.txt",ntraj,nang);
	strcat(outdir,output);
	Dr = 2.0;
	for(i=0;i<flow_steps;++i){
		tf[i] *= Dr;
		flow_mag[i] /= Dr;
		for(j=0;j<9;++j){
			Gd[i][j] /= Dr;
		}
	}
	dt = 0.0005;
	tend = tf[flow_steps - 1];
	tmax = (int)(tend/dt);
	printf("No. time steps %d\n",tmax);
	traj_time_print = 1000;
	// ut = calloc(ntraj,sizeof(double));
	// for(i=0;i<ntraj;++i){
		// ut[i] = calloc(3*tmax,sizeof(double));
	// }
	ui = calloc(3*tmax,sizeof(double));
	uu_avg = calloc(tmax,sizeof(double));
	u4_avg = calloc(tmax,sizeof(double));
	for(i=0;i<tmax;++i){
		uu_avg[i] = calloc(9,sizeof(double));
		u4_avg[i] = calloc(81,sizeof(double));
	}
	avg_count = 0;
	for(i=0;i<9;++i){
		eye[i] = 0.0;
	}
	eye[0] = 1.0; eye[4] = 1.0; eye[8] = 1.0;
	p = sqrt(2.0*dt);
	wall_time = omp_get_wtime();
	idum = malloc(sizeof(long));
	*idum = initRan();

	printf("test\n");
	// Allocate PDF arrays for on the fly averaging
	samp_rate = 1;
	// avg_win = 0.01;
	// avg_win = 0.1;
	steps_p_win = (int)(avg_win/(dt*samp_rate));
	n_win = (int)(ceil(tend/avg_win));
	printf("tend %f avg_win %f n_win %d\n",tend,avg_win,n_win);
	// nang = 300; // Number of angles for 2D pdf
	bin_size = M_PI/nang;
	bin_area = bin_size*bin_size;
	h = calloc(n_win,sizeof(long int));
	hcount = calloc(n_win,sizeof(long int));
	PDF = calloc(n_win,sizeof(double));
	for(i=0;i<n_win;++i){
		h[i] = calloc(nang,sizeof(long int));
		PDF[i] = calloc(nang,sizeof(double));
		for(j=0;j<nang;++j){
			h[i][j] = calloc(nang,sizeof(long int));
			PDF[i][j] = calloc(nang,sizeof(double));
		}
	}

	for(nt=0;nt<ntraj;++nt){
		// outfile = fopen(outdir,"a");
		// fprintf(outfile,"test traj %d\n",nt);
		// fclose(outfile);
		phi0 = 2.0*M_PI*ran1(idum);
		theta0 = acos(2.0*ran1(idum)-1.0);
		ui[0] = sin(theta0)*cos(phi0);
		ui[1] = sin(theta0)*sin(phi0);
		ui[2] = cos(theta0);
		// t_elap = 0.0;
		f_step = 0;
		for(t=1;t<tmax;++t){
			// outfile = fopen(outdir,"a");
			// fprintf(outfile,"test ts %d\n",t);
			// fclose(outfile);
			u[0] = ui[3*(t-1)];
			u[1] = ui[3*(t-1) + 1];
			u[2] = ui[3*(t-1) + 2];
			for(i=0;i<3;++i){
				for(j=0;j<3;++j){
					uu[3*i + j] = u[i]*u[j];
				}
			}
			for(i=0;i<9;++i){
				A[i] = eye[i] - uu[i];
			}
			if(t*dt > tf[f_step]){
				f_step++;
			}
			for(i=0;i<3;++i){
				for(j=0;j<3;++j){
					sum = 0.0;
					for(k=0;k<3;++k){
						sum += A[3*i + k]*Gd[f_step][3*k + j];
					}
					eye_m_uu_x_Gd[3*i + j] = sum;
				}
			}
			for(i=0;i<3;++i){
				sum = 0.0;
				for(j=0;j<3;++j){
					sum += eye_m_uu_x_Gd[3*i + j]*u[j];
				}
				dF[i] = dt*sum;
			}
			for(i=0;i<3;++i){
				R[i] = gasdev(idum);
			}
			// Note the projection operator is its own square root, PP = P
			for(i=0;i<3;++i){
				sum = 0.0;
				for(j=0;j<3;++j){
					sum += A[3*i + j]*R[j];
				}
				dW[i] = p*sum;
			}
			umag = 0.0;
			for(i=0;i<3;++i){
				un[i] = u[i] + dF[i] + dW[i];
				// un[i] = u[i] + dF[i] + p*R[i];
				umag += un[i]*un[i];
			}
			umag = sqrt(umag);
			for(i=0;i<3;++i){
				ui[3*t + i] = un[i]/umag;
			}
		}
		for(i=0;i<tmax;++i){
			thetai = atan2(sqrt(ui[3*i]*ui[3*i] + ui[3*i+1]*ui[3*i+1]),ui[3*i+2]);
			phii = atan2(ui[3*i+1],ui[3*i]) + M_PI;
			win = (int)(i/steps_p_win);
			tbin = (int)(thetai/bin_size);
			phimod = phii - M_PI*floor(phii/M_PI);
			pbin = (int)(phimod/bin_size);
			if(tbin > nang || pbin > nang){
				printf("Outside bin range tbin %d pbin %d theta %lf phi %lf\n",tbin,pbin,thetai,phii);
			}
			h[win][tbin][pbin]++;
			hcount[win]++;
		}
		for(i=0;i<tmax;++i){
			for(j=0;j<3;++j){
				for(k=0;k<3;++k){
					uu_avg[i][3*j + k] += ui[3*i + j]*ui[3*i + k];
				}
			}
		}
		for(i=0;i<tmax;++i){
			for(j=0;j<3;++j){
				for(k=0;k<3;++k){
					for(l=0;l<3;++l){
						for(m=0;m<3;++m){
							u4_avg[i][27*j + 9*k + 3*l + m] += ui[3*i + j]*ui[3*i + k]*ui[3*i + l]*ui[3*i + m];
						}
					}
				}
			}
		}
		avg_count++;
		if(nt%traj_time_print==0){
			printf("nt %d, %d traj in %lf s\n",nt,traj_time_print,omp_get_wtime()-wall_time);
			wall_time = omp_get_wtime();
		}
	}
	// outfile = fopen(outdir,"a");
	// fprintf(outfile,"before PDF calc\n");
	// fclose(outfile);
	// for(i=0;i<n_win;++i){
	// 	printf("i %d hcount %d\n",i,hcount[i]);
	// }
	outfile = fopen(outdir,"w");
	for(i=0;i<n_win;++i){
		// sprintf(output,"output/ht_%.2f_%.2f_%d_%d_%d.txt",FRR,Q2,strm,ntraj,i);
		// outfile = fopen(output,"w");
		for(j=0;j<nang;++j){
			for(k=0;k<nang;++k){
				PDF[i][j][k] = (double)(h[i][j][k])/(bin_area*(double)(hcount[i]));
				// h[i][j][k] /= (double)hcount[i];
				// fprintf(outfile,"%e ",(double)(h[i][j][k])/(double)(hcount[i]));
				fprintf(outfile,"%e ",PDF[i][j][k]);
				// fprintf(outfile,"%d ",h[i][j][k]);
			}
			// fprintf(outfile,"\n");
		}
		fprintf(outfile,"\n");
		// fclose(outfile);
	}
	fclose(outfile);
	getcwd(outdir,4096);
	printf("output dir %s\n",outdir);
	sprintf(output,"/output/FT%.2f_Pe%.1e",FT,Pe);
	strcat(outdir,output);
	// strcat(outdir,output);
	// result = mkdir(outdir, 0777);
	// sprintf(output,"/h_na_%d_nq_%d_qlim_%.2f.txt",nang,nqi,qlim);
	sprintf(output,"/uu_%d.txt",ntraj);
	strcat(outdir,output);
	outfile = fopen(outdir,"w");
	fprintf(outfile,"t");
	for(i=0;i<3;++i){
		for(j=0;j<3;++j){
			fprintf(outfile," u_%d%d",i,j);
		}
	}
	fprintf(outfile,"\n");
	for(i=0;i<tmax;++i){
		fprintf(outfile,"%lf",i*dt);
		for(j=0;j<3;++j){
			for(k=0;k<3;++k){
				fprintf(outfile," %lf",uu_avg[i][3*j + k]/avg_count);
			}
		}
		fprintf(outfile,"\n");
	}
	fclose(outfile);
	getcwd(outdir,4096);
	printf("output dir %s\n",outdir);
	sprintf(output,"/output/FT%.2f_Pe%.1e",FT,Pe);
	strcat(outdir,output);
	sprintf(output,"/u4_%d.txt",ntraj);
	strcat(outdir,output);
	outfile = fopen(outdir,"w");
	fprintf(outfile,"t");
	for(i=0;i<3;++i){
		for(j=0;j<3;++j){
			for(k=0;k<3;++k){
				for(l=0;l<3;++l){
					fprintf(outfile," u_%d%d%d%d",i,j,k,l);
				}
			}
		}
	}
	fprintf(outfile,"\n");
	for(i=0;i<tmax;++i){
		fprintf(outfile,"%lf",i*dt);
		for(j=0;j<3;++j){
			for(k=0;k<3;++k){
				for(l=0;l<3;++l){
					for(m=0;m<3;++m){
						fprintf(outfile," %lf",u4_avg[i][27*j + 9*k + 3*l + m]/avg_count);
					}
				}
			}
		}
		fprintf(outfile,"\n");
	}
	fclose(outfile);
	exit(1);
	// outfile = fopen(outdir,"a");
	// fprintf(outfile,"after PDF calc\n");
	// fclose(outfile);
	// L = 9200;
	// Rc = 33;
	L = 1500.0;
	Rc = 100.0;
	volfrac = 0.1;
	gamm = 0.0;

	qxmin = -qlim;
	qxmax = qlim;
	qxnum = nqi;
	qxstep = (qxmax - qxmin)/(qxnum - 1);
	qxlist = calloc(qxnum,sizeof(double));
	for(i=0;i<qxnum;++i){
		qxlist[i] = qxmin + i*qxstep;
	}

	qymin = -qlim;
	qymax = qlim;
	qynum = nqi;
	qystep = (qymax - qymin)/(qynum - 1);
	qylist = calloc(qynum,sizeof(double));
	for(i=0;i<qynum;++i){
		qylist[i] = qymin + i*qystep;
	}

	nq = qxnum*qynum;
	qx = calloc(nq,sizeof(double));
	qy = calloc(nq,sizeof(double));
	qz = calloc(nq,sizeof(double));
	count = 0;
	for(i=0;i<qxnum;++i){
		ycount = 0;
		for(j=0;j<qynum;++j){
			qx[count + qxnum*ycount] = qxlist[i];
			qy[count + qxnum*ycount] = qylist[j];
			ycount++;
		}
		count++;
	}
	P = calloc(n_win,sizeof(double));
	for(i=0;i<n_win;++i){
		P[i] = calloc(nq,sizeof(double));
	}

	// outfile = fopen(outdir,"a");
	// fprintf(outfile,"before Pq loop\n");
	// fprintf(outfile,"nq %d nwin*nq %d\n",nq,n_win*nq);
	// fclose(outfile);
	// Pj = calloc(nq,sizeof(double));
	printf("nq %d nwin*nq %d\n",nq,n_win*nq);
	wall_time = omp_get_wtime();
	for(i=0;i<nang;++i){
		// printf("theta %d\n",i);
		dtheta = (i + 0.5)*bin_size;
		for(j=0;j<nang;++j){
			dphi = (j + 0.5)*bin_size;
			Pj = cyl_pq(nq,qx,qy,qz,Rc,L,gamm,dtheta,dphi);
			for(k=0;k<n_win;++k){
				for(l=0;l<nq;++l){
					P[k][l] += bin_area*sin(dtheta)*PDF[k][i][j]*Pj[l];
				}
			}
			free(Pj);
			// for(k=0;k<nq;++k){
			// 	Pi = cyl_pqi(qx[k],qy[k],qz[k],Rc,L,gamm,dtheta,dphi);
			// 	for(l=0;l<n_win;++l){
			// 		P[l][k] += bin_area*sin(dtheta)*PDF[l][i][j]*Pi;
			// 	}
			// }
		}
		if(i%10==0){
			printf("i %d dtheta %lf in %lf s\n",i,dtheta,omp_get_wtime()-wall_time);
			// fprintf(outfile,"i %d dtheta %lf in %lf s\n",i,dtheta,omp_get_wtime()-wall_time);
			wall_time = omp_get_wtime();
		}
	}
	printf("calc Pq done\n");
	// outfile = fopen(outdir,"a");
	// fprintf(outfile,"after Pq loop\n");
	// fclose(outfile);
	outfile = fopen(outdir,"w");
	for(i=0;i<n_win;++i){
		for(j=0;j<nq;++j){
			fprintf(outfile,"%e ",P[i][j]);
		}
		fprintf(outfile,"\n");
	}
	fclose(outfile);
	return 0;
}
long initRan(){
    //time_t seconds;
    //time(&seconds);
    //return -1*(unsigned long)(seconds/12345); This is bad.  :(

    //This will hopefully allow us to have a unique seed even if executed multiple times a second-Got from Mike
    //http://stackoverflow.com/questions/322938/recommended-way-to-initialize-srand
    unsigned long a = clock();
    unsigned long b = time(NULL);
    unsigned long c = getpid();
    a=a-b;  a=a-c;  a=a^(c >> 13);
    b=b-c;  b=b-a;  b=b^(a << 8);
    c=c-a;  c=c-b;  c=c^(b >> 13);
    a=a-b;  a=a-c;  a=a^(c >> 12);
    b=b-c;  b=b-a;  b=b^(a << 16);
    c=c-a;  c=c-b;  c=c^(b >> 5);
    a=a-b;  a=a-c;  a=a^(c >> 3);
    b=b-c;  b=b-a;  b=b^(a << 10);
    c=c-a;  c=c-b;  c=c^(b >> 15);

    return c%1000000000; //careful here.  Another 0 might break the ran1 (long long instead of just long)
}
float ran1(long *idum){
// float ran1(unsigned long *idum){
	int j;
	long k;
	static long idum2 = 123456789;
	static long iy=0;
	static long iv[NTAB];
	float temp;

	if(*idum <= 0)
	{
		if (-(*idum) < 1) *idum=1;
		else *idum = -(*idum);
		idum2 = (*idum);
		for(j=NTAB+7;j>=0;--j)
		{
			k=(*idum)/IQ1;
			*idum=IA1*(*idum-k*IQ1)-k*IR1;
			if(*idum<0) *idum+=IM1;
			if(j<NTAB) iv[j] = *idum;
		}
		iy = iv[0];
	}
	k = (*idum)/IQ1;
	*idum=IA1*(*idum-k*IQ1)-k*IR1;
	if(*idum<0) *idum += IM1;
	k=idum2/IQ2;
	if(*idum<0) idum2+= IM2;
	j=iy/NDIV;
	iy=iv[j]-idum2;
	iv[j] = *idum;
	if(iy<1) iy += IMM1;
	if((temp=AM*iy) > RNMX) return RNMX;
	else return temp;
}
// float gasdev(long *idum){
float gasdev(long *idum){
// float gasdev(unsigned long *idum){
	float ran1(long *idum);
	// float ran1(unsigned long *idum);
	static int iset=0;
	static float gset;
	float fac,rsq,v1,v2;

	if (*idum < 0) iset = 0;
	if (iset == 0)
	{
		do
		{
			v1 = 2.0*ran1(idum)-1.0;
			v2 = 2.0*ran1(idum)-1.0;
			rsq = v1*v1+v2*v2;
		}
		while(rsq >= 1.0 || rsq == 0.0);
		fac=sqrt(-2.0*log(rsq)/rsq);
		gset=v1*fac;
		iset=1;
		return v2*fac;
	}
	else
	{
		iset = 0;
		return gset;
	}
}
double* cyl_pq(int nq,double *qx,double *qy,double *qz,double Rc,double L,double gamm,double beta,double alpha){
	int i;
	double *Pq = calloc(nq,sizeof(double));
	for(i=0;i<nq;++i){
		Pq[i] = pow(
				4.0*j1(Rc*sqrt(pow(qy[i]*cos(alpha) - qx[i]*sin(alpha),2) + pow(cos(beta)*(qx[i]*cos(alpha) + qy[i]*sin(alpha)) - qz[i]*sin(beta),2)))*sin(0.5*L*(qz[i]*cos(beta) + (qx[i]*cos(alpha) + qy[i]*sin(alpha))*sin(beta))) / 
				(L*Rc*(qz[i]*cos(beta) + (qx[i]*cos(alpha) + qy[i]*sin(alpha))*sin(beta))*sqrt(pow((qy[i]*cos(alpha) - qx[i]*sin(alpha)),2) + pow(cos(beta)*(qx[i]*cos(alpha) + qy[i]*sin(alpha)) - qz[i]*sin(beta),2)))
			,2);
	}
	return Pq;
	free(Pq);
}
double cyl_pqi(double qx,double qy,double qz,double Rc,double L,double gamm,double beta,double alpha){
	double Pqi;
	Pqi = pow(
			4.0*j1(Rc*sqrt(pow(qy*cos(alpha) - qx*sin(alpha),2) + pow(cos(beta)*(qx*cos(alpha) + qy*sin(alpha)) - qz*sin(beta),2)))*sin(0.5*L*(qz*cos(beta) + (qx*cos(alpha) + qy*sin(alpha))*sin(beta))) / 
			(L*Rc*(qz*cos(beta) + (qx*cos(alpha) + qy*sin(alpha))*sin(beta))*sqrt(pow((qy*cos(alpha) - qx*sin(alpha)),2) + pow(cos(beta)*(qx*cos(alpha) + qy*sin(alpha)) - qz*sin(beta),2)))
		,2);
	return Pqi;
}
