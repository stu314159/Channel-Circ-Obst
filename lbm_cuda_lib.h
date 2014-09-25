#ifndef LBM_CUDA_LIB_H
#define LBM_CUDA_LIB_H


void ts_pois3D_D3Q15_LBGK_cuda(const float * fIn, float * fOut, const int * snl,
			      const int * inl, const int * onl,
			      const float * u_bc, const float omega,
			      const int Nx, const int Ny,
			      const int firstSlice,
			      const int lastSlice,const int nnodes);

void stream_out_collect_cuda(const float * fIn_b, float * buff_out,
			     const int numStreamSpeeds, 
			     const int * streamSpeeds,
			     const int Nx, const int Ny, 
			     const int Nz, const int HALO);

void stream_in_distribute_cuda(float * fIn_b,const float * buff_in,
			       const int numStreamSpeeds, 
			       const int * streamSpeeds,
			       const int Nx, const int Ny,
			       const int Nz, const int HALO);

#endif
