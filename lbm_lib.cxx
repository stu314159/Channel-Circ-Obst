

void initialize_ubc_pois3D(float * u_bc, const float u_bc_max, 
			   const int * inl, const int * onl,
			   const int Nx, const int Ny,
			   const int Nz){

  int tid;
  float b = ((float)Ny-1.)/2.;
  float h;
  for(int z = 0; z<Nz; z++){
    for(int y = 0; y< Ny; y++){
      for( int x = 0; x<Nx;x++){
	tid = x+y*Nx+z*Nx*Ny;
	if((inl[tid]==1)|(onl[tid]==1)){
	  //compute u_bc[tid] based on y value.
	  h = ((float)y - b)/b;
	  u_bc[tid]=u_bc_max*(1.-(h*h));	  
	}else{
	  u_bc[tid]=0.;
	}
      }
    }
  }

}


void initialize_lattice_partitionT(float * fIn, const float rho_init,
				   const float * w, const int Nx,
				   const int Ny, const int Nz, const int numSpd){
  int tid;
  int nnodes=Nx*Ny*Nz;
  for(int spd=0;spd<numSpd;spd++){
    for(int z=0;z<Nz;z++){
      for(int y = 0; y<Ny;y++){
	for(int x=0;x<Nx;x++){
	    tid = x+y*Nx+z*Nx*Ny;
	    fIn[tid+spd*nnodes]=rho_init*w[spd];

	  }
      }
    }
  }
	  

}

void initialize_lattice_partition(float * fIn, const float rho_init,
				  const float * w, const int Nx,
				  const int Ny, const int Nz, const int numSpd){

   int tid;
   for(int z=0;z<Nz;z++){
    for(int y=0;y<Ny;y++){
      for(int x=0;x<Nx;x++){
	tid=x+y*Nx+z*Nx*Ny;
	for(int spd=0;spd<numSpd;spd++){
	  fIn[spd+tid*numSpd]=rho_init*w[spd];
	}
      }
    }
  }
}

void initialize_snl_partition(int * snl, const int Nx, const int Ny,
			      const int Nz, const int numSpd){
 int nnodes = Nx*Ny*Nz;
  int tid;
  for(int z=0;z<Nz;z++){
    for(int y=0;y<Ny;y++){
      for(int x=0;x<Nx;x++){
	tid=x+y*Nx+z*Nx*Ny;
	if((y==0)||(y==(Ny-1))){
	  snl[tid]=1;
	}else{
	  snl[tid]=0;
	}
      }
    }
  }

}

void ts_pois3D_D3Q15_LBGK_simple(float * fIn, float * fOut,const int * snl, 
		     const int * inl, const int * onl, 
		     const float * u_bc, const float omega,
		     const int Nx, const int Ny,const int firstSlice,
				 const int lastSlice){

  float ex[15]={0,1,-1,0,0,0,0,1,-1,1,-1,1,-1,1,-1};
  float ey[15]={0,0,0,1,-1,0,0,1,1,-1,-1,1,1,-1,-1};
  float ez[15]={0,0,0,0,0,1,-1,1,1,1,1,-1,-1,-1,-1};
  float w[15]={2./9.,1./9.,1./9,1./9.,1./9.,1./9.,1./9.,
	       1./72.,1./72.,1./72.,1./72.,
	       1./72.,1./72.,1./72.,1./72.};

  int bb_spd[15]={0,2,1,4,3,6,5,14,13,12,11,10,9,8,7};
  const int numSpd=15;

  int Nz = lastSlice - firstSlice;
  int nnodes=Nx*Ny*Nz;
  float rho,ux,uy,uz,f_tmp,cu;

  for(int z=firstSlice;z<lastSlice;z++){
    for(int y=0;y<Ny;y++){
      for(int x=0;x<Nx;x++){
	int nd=x+y*Nx+z*Nx*Ny;
	rho=0.;ux=0.;uy=0.;uz=0.;
	
	for(int spd=0;spd<numSpd;spd++){
	  f_tmp=fIn[spd+nd*numSpd];
	  rho+=f_tmp; ux+=f_tmp*ex[spd]; uy+=f_tmp*ey[spd]; uz+=f_tmp*ez[spd];
	}
	ux/=rho; uy/=rho; uz/=rho;
	if((inl[nd]==1) || (onl[nd]==1)){
	  float dx,dy,dz;
	  dx=-ux;
	  dy=-uy;
	  dz=u_bc[nd]-uz;
	  for(int spd=1;spd<numSpd;spd++){//start on the second speed
	    cu = 3.0*(ex[spd]*dx+ey[spd]*dy+ez[spd]*dz);
	    fIn[spd+nd*numSpd]+=w[spd]*rho*cu;
	  }
	  uz=u_bc[nd]; uy=0.; ux=0.;
	}

	if(snl[nd]==0){
	  f_tmp=0.; //temporary for fEq...
	  for(int spd=0;spd<numSpd;spd++){
	    cu=3.0*(ex[spd]*ux+ey[spd]*uy+ez[spd]*uz);
	    f_tmp=w[spd]*rho*(1.0+cu+(0.5)*(cu*cu)-(1.5)*(ux*ux+uy*uy+uz*uz));
	    fIn[spd+nd*numSpd]=fIn[spd+nd*numSpd]-omega*(fIn[spd+nd*numSpd]-f_tmp);
	  }

	}else{ //if a solid node, bounce-back to generate 
	  for(int spd=0;spd<numSpd;spd++){
	    // fOut[bb_spd[spd]+nd*numSpd]=fIn[spd+nd*numSpd];
	    f_tmp=fIn[bb_spd[spd]+nd*numSpd];
	    fIn[bb_spd[spd]+nd*numSpd]=fIn[spd+nd*numSpd];
	    fIn[spd+nd*numSpd]=f_tmp;
	  }
	}


      }
    }
  }

  int Xt,Yt,Zt,tid_t;
  int tid;


  for(int Z=firstSlice;Z<lastSlice;Z++){
    for(int Y=0;Y<Ny;Y++){
      for(int X=0;X<Nx;X++){
	tid = X+Y*Nx+Z*Nx*Ny;
	for(int spd=0;spd<numSpd;spd++){
	 
	  Xt = X+(int)ex[spd];Yt=Y+(int)ey[spd];Zt=Z+(int)ez[spd];
	
	  if(Xt<0) Xt=(Nx-1);
	  if(Yt<0) Yt=(Ny-1);
	  //if(Zt<0) Zt=(Nz-1);
	  if(Xt==Nx) Xt=0;
	  if(Yt==Ny) Yt=0;
	  //if(Zt==Nz) Zt=0;
	  tid_t = Xt+Yt*Nx+Zt*Nx*Ny;
	  
	  fOut[tid_t*numSpd+spd]=fIn[tid*numSpd+spd];
	
	}
      }
    }
  }




}


void ts_pois3D_D3Q15_LBGK(const float * fIn, float * fOut,const int * snl, 
		     const int * inl, const int * onl, 
		     const float * u_bc, const float omega,
		     const int Nx, const int Ny,const int totalSlices,
		     const int HALO){

  float f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14;
  float ft1,ft2,ft3,ft4,ft5; //<-- temps for unrolling
  float fe1,fe2,fe3,fe4,fe5;
  const float w1=2./9.;
  const float w2=1./9.;
  const float w3=1./72.;
  float rho,ux,uy,uz,dx,dy,dz;
  int tid;
  const int numSpd=15;
  //loop over applicable lattice points
  for(int z=HALO;z<(totalSlices-HALO);z++){
    for(int y=0;y<Ny;y++){
      for(int x=0;x<Nx;x++){ 
	//load data into registers
	tid=x+y*Nx+z*Nx*Ny;
	f0=fIn[0+tid*numSpd];
	f1=fIn[1+tid*numSpd];
	f2=fIn[2+tid*numSpd];
	f3=fIn[3+tid*numSpd];
	f4=fIn[4+tid*numSpd];
	f5=fIn[5+tid*numSpd];
	f6=fIn[6+tid*numSpd];
	f7=fIn[7+tid*numSpd];
	f8=fIn[8+tid*numSpd];
	f9=fIn[9+tid*numSpd];
	f10=fIn[10+tid*numSpd];
	f11=fIn[11+tid*numSpd];
	f12=fIn[12+tid*numSpd];
	f13=fIn[13+tid*numSpd];
	f14=fIn[14+tid*numSpd];
	//compute macroscopic density and velocity
	rho=f0+f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+f11+f12+f13+f14;

	//compute fEq and collide for each speed
	ux=f1-f2+f7-f8+f9-f10+f11-f12+f13-f14; ux=ux/rho;
	uy=f3-f4+f7+f8-f9-f10+f11+f12-f13-f14; uy=uy/rho;
	uz=f5-f6+f7+f8+f9+f10-f11-f12-f13-f14; uz=uz/rho;

	//for nodes on the inlet our outlet node list, apply velocity bc
	if((inl[tid]==1)||(onl[tid]==1)){
	  //no update to speed 0

	  //speed 1 ex=1 ey=ez=0. w=1./9.
	  //speed 2 ex=-1 ey=ez=0. w=1./9.
	  //speed 3 ey=1; ex=ez=0; w=1./9.
	  //speed 4 ey=-1; ex=ez=0; w=1./9.
	  //speed 5 ex=ey=0; ez=1; w=1./9.
	  //speed 6 ex=ey=0; ez=-1; w=1./9.
	  //speed 7 ex=ey=ez=1; w=1./72.
	  //speed 8 ex=-1 ey=ez=1; w=1./72.
	  //speed 9 ex=1 ey=-1 ez=1
	  //speed 10 ex=-1 ey=-1 ez=1
	  //speed 11 ex=1 ey=1 ez=-1
	  //speed 12 ex=-1 ey=1 ez=-1
	  //speed 13 ex=1 ey=-1 ez=-1 w=1./72.
	  //speed 14 ex=ey=ez=-1 w=1./72.

	}

	//for nodes on snl, bounce back if not on snl, compute fEq and collide.
	if(snl[tid]==1){


	}else{

	  //speed 1 ex=1 ey=ez=0. w=1./9.
	  //speed 2 ex=-1 ey=ez=0. w=1./9.
	  //speed 3 ey=1; ex=ez=0; w=1./9.
	  //speed 4 ey=-1; ex=ez=0; w=1./9.
	  //speed 5 ex=ey=0; ez=1; w=1./9.
	  //speed 6 ex=ey=0; ez=-1; w=1./9.
	  //speed 7 ex=ey=ez=1; w=1./72.
	  //speed 8 ex=-1 ey=ez=1; w=1./72.
	  //speed 9 ex=1 ey=-1 ez=1
	  //speed 10 ex=-1 ey=-1 ez=1
	  //speed 11 ex=1 ey=1 ez=-1
	  //speed 12 ex=-1 ey=1 ez=-1
	  //speed 13 ex=1 ey=-1 ez=-1 w=1./72.
	  //speed 14 ex=ey=ez=-1 w=1./72.


	}	
	//stream to fOut

	//speed 1 ex=1 ey=ez=0. w=1./9.
	//speed 2 ex=-1 ey=ez=0. w=1./9.
	//speed 3 ey=1; ex=ez=0; w=1./9.
	//speed 4 ey=-1; ex=ez=0; w=1./9.
	//speed 5 ex=ey=0; ez=1; w=1./9.
	//speed 6 ex=ey=0; ez=-1; w=1./9.
	//speed 7 ex=ey=ez=1; w=1./72.
	//speed 8 ex=-1 ey=ez=1; w=1./72.
	//speed 9 ex=1 ey=-1 ez=1
	//speed 10 ex=-1 ey=-1 ez=1
	//speed 11 ex=1 ey=1 ez=-1
	//speed 12 ex=-1 ey=1 ez=-1
	//speed 13 ex=1 ey=-1 ez=-1 w=1./72.
	//speed 14 ex=ey=ez=-1 w=1./72.

      }
    }
  }

}


void stream_out_collect(const float* fIn_b,float * buff_out,
			const int numSpeeds, const int numStreamSpeeds,
			const int * streamSpeeds,
			const int Nx, const int Ny,
			const int Nz, const int HALO){
  //see comments in stream_in_distribute
  int tid_l, stream_spd;
  for(int z=0;z<HALO;z++){
    for(int y=0;y<Ny;y++){
      for(int x=0;x<Nx;x++){
	for(int spd=0;spd<numStreamSpeeds;spd++){
	  tid_l = x+y*Nx+z*Nx*Ny;
	  stream_spd=streamSpeeds[spd];
	  buff_out[tid_l*numStreamSpeeds+spd]=fIn_b[tid_l*numSpeeds+stream_spd];
	}
      }
    }
  }

}

void stream_in_distribute(float * fIn_b,const float * buff_in,
			  const int numSpeeds,const int numStreamSpeeds,
			  const int * streamSpeeds,
			  const int Nx, const int Ny, 
			  const int Nz,const int HALO){

  //fIn_b is a pointer to the first memory location of the boundary lattice points -- all of which are assumed to be consecutive lattice points.
  //buff_in holds the numStreamSpeeds*Nx*Ny*HALO values that will be distributed to 
  //the streamSpeeds of fIn_b.
  //for this computation it is assumed that each lattice point has it's density distributions stored consecutively...for MPI/CUDA cases, this will be different. 
  //I don't usse Nz for this case, but for the CUDA case I will and I want to keep the
  //interface consistent.
  int tid_l,stream_spd;  
  for(int z=0;z<HALO;z++){
    for(int y=0;y<Ny;y++){
      for(int x=0;x<Nx;x++){
	for(int spd=0;spd<numStreamSpeeds;spd++){
	  tid_l=x+y*Nx+z*Nx*Ny;
	  stream_spd=streamSpeeds[spd];
	  fIn_b[tid_l*numSpeeds+stream_spd]=buff_in[tid_l*numStreamSpeeds+spd];
	}
      
      }
    }
  }
    

}


void initialize_inl_partition(){}

void initialize_onl_partition(){}

void ts_pois3D_D3Q15_LBGK_r2(const float * fIn, float * fOut,const int * snl, 
			     const int * inl, const int * onl, 
			     const float * u_bc, const float omega,
			     const int Nx, const int Ny,const int firstSlice,
			     const int lastSlice){

  float f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14;
  float cu,rho,ux,uy,uz,fEq,dz;
  int X_t,Y_t,Z_t,tid_t,tid,Nz;

  //Nz=lastSlice-firstSlice;
  const int numSpd=15;
  for(int Z=firstSlice;Z<lastSlice;Z++){
    for(int Y=0;Y<Ny;Y++){
      for(int X=0;X<Nx;X++){

	tid=X+Y*Nx+Z*Nx*Ny;

	//load the data into registers
	// f0=fIn[tid]; f1=fIn[Nx*Ny*Nz+tid];
	// f2=fIn[2*Nx*Ny*Nz+tid]; f3=fIn[3*Nx*Ny*Nz+tid];
	// f4=fIn[4*Nx*Ny*Nz+tid]; f5=fIn[5*Nx*Ny*Nz+tid];
	// f6=fIn[6*Nx*Ny*Nz+tid]; f7=fIn[7*Nx*Ny*Nz+tid];
	// f8=fIn[8*Nx*Ny*Nz+tid]; f9=fIn[9*Nx*Ny*Nz+tid];
	// f10=fIn[10*Nx*Ny*Nz+tid]; f11=fIn[11*Nx*Ny*Nz+tid];
	// f12=fIn[12*Nx*Ny*Nz+tid]; f13=fIn[13*Nx*Ny*Nz+tid];
	// f14=fIn[14*Nx*Ny*Nz+tid];
	f0=fIn[tid*numSpd]; f1=fIn[tid*numSpd+1];
	f2=fIn[tid*numSpd+2]; f3=fIn[tid*numSpd+3];
	f4=fIn[tid*numSpd+4]; f5=fIn[tid*numSpd+5];
	f6=fIn[tid*numSpd+6]; f7=fIn[tid*numSpd+7];
	f8=fIn[tid*numSpd+8]; f9=fIn[tid*numSpd+9];
	f10=fIn[tid*numSpd+10]; f11=fIn[tid*numSpd+11];
	f12=fIn[tid*numSpd+12]; f13=fIn[tid*numSpd+13];
	f14=fIn[tid*numSpd+14];

	//compute density
	rho = f0+f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+f11+f12+f13+f14;
	ux=f1-f2+f7-f8+f9-f10+f11-f12+f13-f14; ux/=rho;
	uy=f3-f4+f7+f8-f9-f10+f11+f12-f13-f14; uy/=rho;
	uz=f5-f6+f7+f8+f9+f10-f11-f12-f13-f14; uz/=rho;
	
	//if it's on the inl or onl, update

	if((inl[tid]==1)||(onl[tid]==1)){

	  dz=u_bc[tid]-uz;
	  //speed 1 ex=1 ey=ez=0. w=1./9.
	  cu=3.*(1.)*(-ux);
	  f1+=(1./9.)*rho*cu;

	  //speed 2 ex=-1 ey=ez=0. w=1./9.
	  cu=3.*(-1.)*(-ux);
	  f2+=(1./9.)*rho*cu;

	  //speed 3 ey=1; ex=ez=0; w=1./9.
	  cu=3.*(1.)*(-uy);
	  f3+=(1./9.)*rho*cu;

	  //speed 4 ey=-1; ex=ez=0; w=1./9.
	  cu=3.*(-1.)*(-uy);
	  f4+=(1./9.)*rho*cu;

	  //speed 5 ex=ey=0; ez=1; w=1./9.
	  cu=3.*(1.)*(dz);
	  f5+=(1./9.)*rho*cu;

	  //speed 6 ex=ey=0; ez=-1; w=1./9.
	  cu=3.*(-1.)*(dz);
	  f6+=(1./9.)*rho*cu;

	  //speed 7 ex=ey=ez=1; w=1./72.
	  cu=3.*((1.)*-ux+(1.)*(-uy)+(1.)*dz);
	  f7+=(1./72.)*rho*cu;

	  //speed 8 ex=-1 ey=ez=1; w=1./72.
	  cu=3.*((-1.)*-ux+(1.)*(-uy)+(1.)*dz);
	  f8+=(1./72.)*rho*cu;

	  //speed 9 ex=1 ey=-1 ez=1
	  cu=3.0*((1.)*-ux+(-1.)*(-uy)+(1.)*dz);
	  f9+=(1./72.)*rho*cu;

	  //speed 10 ex=-1 ey=-1 ez=1
	  cu=3.0*((-1.)*-ux+(-1.)*(-uy)+(1.)*dz);
	  f10+=(1./72.)*rho*cu;

	  //speed 11 ex=1 ey=1 ez=-1
	  cu=3.0*((1.)*-ux +(1.)*(-uy)+(-1.)*dz);
	  f11+=(1./72.)*rho*cu;

	  //speed 12 ex=-1 ey=1 ez=-1
	  cu=3.0*((-1.)*-ux+(1.)*(-uy)+(-1.)*dz);
	  f12+=(1./72.)*rho*cu;

	  //speed 13 ex=1 ey=-1 ez=-1 w=1./72.
	  cu=3.0*((1.)*-ux+(-1.)*(-uy)+(-1.)*dz);
	  f13+=(1./72.)*rho*cu;
      
	  //speed 14 ex=ey=ez=-1 w=1./72.
	  cu=3.0*((-1.)*-ux + (-1.)*(-uy) +(-1.)*dz);
	  f14+=(1./72.)*rho*cu;

	  ux=0.; uy=0.; uz=u_bc[tid];
	}

	if(snl[tid]==1){
	  // 1--2
	  cu=f2; f2=f1; f1=cu;
	  //3--4
	  cu=f4; f4=f3; f3=cu;
	  //5--6
	  cu=f6; f6=f5; f5=cu;
	  //7--14
	  cu=f14; f14=f7; f7=cu;
	  //8--13
	  cu=f13; f13=f8; f8=cu;
	  //9--12
	  cu=f12; f12=f9; f9=cu;
	  //10--11
	  cu=f11; f11=f10; f10=cu;


	}else{
	  fEq=rho*(2./9.)*(1.-1.5*(ux*ux+uy*uy+uz*uz));
	  f0=f0-omega*(f0-fEq);

	  //speed 1 ex=1 ey=ez=0 w=1./9.
	  cu=3.*(1.*ux);
	  fEq=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
			   1.5*(ux*ux+uy*uy+uz*uz));
	  f1=f1-omega*(f1-fEq);

	  //speed 2 ex=-1 ey=ez=0 w=1./9.
	  cu=3.*((-1.)*ux);
	  fEq=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
			   1.5*(ux*ux+uy*uy+uz*uz));
	  f2=f2-omega*(f2-fEq);

	  //speed 3 ex=0 ey=1 ez=0 w=1./9.
	  cu=3.*(1.*uy);
	  fEq=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
			   1.5*(ux*ux+uy*uy+uz*uz));
	  f3=f3-omega*(f3-fEq);

	  //speed 4 ex=0 ey=-1 ez=0 w=1./9.
	  cu=3.*(-1.*uy);
	  fEq=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
			   1.5*(ux*ux+uy*uy+uz*uz));
	  f4=f4-omega*(f4-fEq);

	  //speed 5 ex=ey=0 ez=1 w=1./9.
	  cu=3.*(1.*uz);
	  fEq=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
			   1.5*(ux*ux+uy*uy+uz*uz));
	  f5=f5-omega*(f5-fEq);

	  //speed 6 ex=ey=0 ez=-1 w=1./9.
	  cu=3.*(-1.*uz);
	  fEq=rho*(1./9.)*(1.+cu+0.5*(cu*cu)-
			   1.5*(ux*ux+uy*uy+uz*uz));
	  f6=f6-omega*(f6-fEq);

	  //speed 7 ex=ey=ez=1 w=1./72.
	  cu=3.*(ux+uy+uz);
	  fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
			    1.5*(ux*ux+uy*uy+uz*uz));
	  f7=f7-omega*(f7-fEq);

	  //speed 8 ex=-1 ey=ez=1 w=1./72.
	  cu=3.*(-ux+uy+uz);
	  fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
			    1.5*(ux*ux+uy*uy+uz*uz));
	  f8=f8-omega*(f8-fEq);

	  //speed 9 ex=1 ey=-1 ez=1 w=1./72.
	  cu=3.*(ux-uy+uz);
	  fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
			    1.5*(ux*ux+uy*uy+uz*uz));
	  f9=f9-omega*(f9-fEq);

	  //speed 10 ex=-1 ey=-1 ez=1 w=1/72
	  cu=3.*(-ux-uy+uz);
	  fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
			    1.5*(ux*ux+uy*uy+uz*uz));
	  f10=f10-omega*(f10-fEq);

	  //speed 11 ex=1 ey=1 ez=-1 w=1/72
	  cu=3.*(ux+uy-uz);
	  fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
			    1.5*(ux*ux+uy*uy+uz*uz));
	  f11=f11-omega*(f11-fEq);

	  //speed 12 ex=-1 ey=1 ez=-1 w=1/72
	  cu=3.*(-ux+uy-uz);
	  fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
			    1.5*(ux*ux+uy*uy+uz*uz));
	  f12=f12-omega*(f12-fEq);

	  //speed 13 ex=1 ey=ez=-1 w=1/72
	  cu=3.*(ux-uy-uz);
	  fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
			    1.5*(ux*ux+uy*uy+uz*uz));
	  f13=f13-omega*(f13-fEq);

	  //speed 14 ex=ey=ez=-1 w=1/72
	  cu=3.*(-ux-uy-uz);
	  fEq=rho*(1./72.)*(1.+cu+0.5*(cu*cu)-
			    1.5*(ux*ux+uy*uy+uz*uz));
	  f14=f14-omega*(f14-fEq);



	}

	//speed 0 ex=ey=ez=0
	//fOut[tid]=f0;
	fOut[tid*numSpd]=f0;

	//speed 1 ex=1 ey=ez=0
	X_t=X+1; Y_t=Y; Z_t=Z;
	if(X_t==Nx) X_t=0;
	tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
	//	fOut[Nx*Ny*Nz+tid_t]=f1;
	fOut[tid_t*numSpd+1]=f1;

	//speed 2 ex=-1 ey=ez=0;
	X_t=X-1; Y_t=Y; Z_t=Z;
	if(X_t<0) X_t=(Nx-1);
	tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
	//	fOut[2*Nx*Ny*Nz+tid_t]=f2;
	fOut[tid_t*numSpd+2]=f2;

	//speed 3 ex=0 ey=1 ez=0
	X_t=X; Y_t=Y+1; Z_t=Z;
	if(Y_t==Ny) Y_t=0;
	tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
	//	fOut[3*Nx*Ny*Nz+tid_t]=f3;
	fOut[tid_t*numSpd+3]=f3;

	//speed 4 ex=0 ey=-1 ez=0
	X_t=X; Y_t=Y-1; Z_t=Z;
	if(Y_t<0) Y_t=(Ny-1);
	tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
	///	fOut[4*Nx*Ny*Nz+tid_t]=f4;
	fOut[tid_t*numSpd+4]=f4;


	//speed 5 ex=ey=0 ez=1
	X_t=X; Y_t=Y; Z_t=Z+1;
	//	if(Z_t==Nz) Z_t=0;
	tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
	//fOut[5*Nx*Ny*Nz+tid_t]=f5;
	fOut[tid_t*numSpd+5]=f5;

	//speed 6 ex=ey=0 ez=-1
	X_t=X; Y_t=Y; Z_t=Z-1;
	//	if(Z_t<0) Z_t=(Nz-1);
	tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
	//	fOut[6*Nx*Ny*Nz+tid_t]=f6;
	fOut[tid_t*numSpd+6]=f6;

	//speed 7 ex=ey=ez=1
	X_t=X+1; Y_t=Y+1; Z_t=Z+1;
	if(X_t==Nx) X_t=0;
	if(Y_t==Ny) Y_t=0;
	//	if(Z_t==Nz) Z_t=0;
	tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
	//	fOut[7*Nx*Ny*Nz+tid_t]=f7;
	fOut[tid_t*numSpd+7]=f7;

	//speed 8 ex=-1 ey=1 ez=1
	X_t=X-1; Y_t=Y+1; Z_t=Z+1;
	if(X_t<0) X_t=(Nx-1);
	if(Y_t==Ny) Y_t=0;
	//	if(Z_t==Nz) Z_t=0;
	tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
	//	fOut[8*Nx*Ny*Nz+tid_t]=f8;
	fOut[tid_t*numSpd+8]=f8;

	//speed 9 ex=1 ey=-1 ez=1
	X_t=X+1; Y_t=Y-1; Z_t=Z+1;
	if(X_t==Nx) X_t=0;
	if(Y_t<0) Y_t=(Ny-1);
	//	if(Z_t==Nz) Z_t=0;
	tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
	//	fOut[9*Nx*Ny*Nz+tid_t]=f9;
	fOut[tid_t*numSpd+9]=f9;

	//speed 10 ex=-1 ey=-1 ez=1
	X_t=X-1; Y_t=Y-1; Z_t=Z+1;
	if(X_t<0) X_t=(Nx-1);
	if(Y_t<0) Y_t=(Ny-1);
	//	if(Z_t==Nz) Z_t=0;
	tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
	//	fOut[10*Nx*Ny*Nz+tid_t]=f10;
	fOut[tid_t*numSpd+10]=f10;

	//speed 11 ex=1 ey=1 ez=-1
	X_t=X+1; Y_t=Y+1; Z_t=Z-1;
	if(X_t==Nx) X_t=0;
	if(Y_t==Ny) Y_t=0;
	//	if(Z_t<0) Z_t=(Nz-1);
	tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
	//	fOut[11*Nx*Ny*Nz+tid_t]=f11;
	fOut[tid_t*numSpd+11]=f11;

	//speed 12 ex=-1 ey=1 ez=-1
	X_t=X-1; Y_t=Y+1; Z_t=Z-1;
	if(X_t<0) X_t=(Nx-1);
	if(Y_t==Ny) Y_t=0;
	//	if(Z_t<0) Z_t=(Nz-1);
	tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
	//	fOut[12*Nx*Ny*Nz+tid_t]=f12;
	fOut[tid_t*numSpd+12]=f12;

	//speed 13 ex=1 ey=-1 ez=-1
	X_t=X+1; Y_t=Y-1; Z_t=Z-1;
	if(X_t==Nx) X_t=0;
	if(Y_t<0) Y_t=(Ny-1);
	//	if(Z_t<0) Z_t=(Nz-1);
	tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
	//	fOut[13*Nx*Ny*Nz+tid_t]=f13;
	fOut[tid_t*numSpd+13]=f13;

	//speed 14 ex=ey=ez=-1
	X_t=X-1; Y_t=Y-1; Z_t=Z-1;
	if(X_t<0) X_t=(Nx-1);
	if(Y_t<0) Y_t=(Ny-1);
	//	if(Z_t<0) Z_t=(Nz-1);
	tid_t=X_t+Y_t*Nx+Z_t*Nx*Ny;
	//	fOut[14*Nx*Ny*Nz+tid_t]=f14;

	fOut[tid_t*numSpd+14]=f14;



	

      }
    }
  }

}
