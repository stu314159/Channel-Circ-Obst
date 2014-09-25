#ifndef LBM_LIB_H
#define LBM_LIB_H

void initialize_ubc_pois3D(float * u_bc, const float u_bc_max, 
			   const int * inl, const int * onl,
			   const int Nx, const int Ny,
			   const int Nz);

void initialize_lattice_partitionT(float * fIn, const float rho_init,
				   const float * w, const int Nx,
				   const int Ny, const int Nz, const int numSpd);

void initialize_lattice_partition(float * fIn, const float rho_init,
				  const float * w, const int Nx,
				  const int Ny, const int Nz, const int numSpd);

void initialize_snl_partition(int * snl, const int Nx, const int Ny,
			      const int Nz, const int numSpd);

void ts_pois3D_D3Q15_LBGK(const float * fIn, float * fOut,const int * snl, 
		     const int * inl, const int * onl, 
		     const float * u_bc, const float omega,
		     const int Nx, const int Ny,const int firstSlice,
		     const int lastSlice);

void ts_pois3D_D3Q15_LBGK_simple( float * fIn, float * fOut,const int * snl, 
				  const int * inl, const int * onl, 
				  const float * u_bc, const float omega,
				  const int Nx, const int Ny,const int firstSlice,
				  const int lastSlice);

void stream_in_distribute(float * fIn_b,const float * buff_in,
			  const int numSpeeds,const int numStreamSpeeds,
			  const int * streamSpeeds,
			  const int Nx, const int Ny, 
			  const int Nz,const int HALO);

void stream_out_collect(const float* fIn_b,float * buff_out,
			const int numSpeeds, const int numStreamSpeeds,
			const int * streamSpeeds,
			const int Nx, const int Ny,
			const int Nz, const int HALO);

void ts_pois3D_D3Q15_LBGK_r2(const float * fIn, float * fOut,const int * snl, 
			     const int * inl, const int * onl, 
			     const float * u_bc, const float omega,
			     const int Nx, const int Ny,const int firstSlice,
			     const int lastSlice);

#endif
