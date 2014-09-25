MPICC=mpicxx
MPI_FLAGS=-O3
MPI_LD_LIB=/opt/cuda/4.0.17/cuda/lib64
MPI_INCL=/opt/cuda/4.0.17/cuda/include
LIBS=-lcudart

NV=nvcc
NV_FLAGS=-O3 -arch compute_20

CPP_CC=g++
CPP_FLAGS=-O3


SOURCE=lbm_mpi_cuda.cxx
TARGET=pois3D_mpiCuda

$(TARGET):  $(SOURCE) vtk_lib.o lbm_lib.o lbm_cuda_lib.o
	$(MPICC) -o $(TARGET) $(SOURCE) -L$(MPI_LD_LIB) -I$(MPI_INCL)  vtk_lib.o lbm_lib.o lbm_cuda_lib.o $(LIBS)

vtk_lib.o:  vtk_lib.cxx
	$(CPP_CC) -c vtk_lib.cxx $(CPP_FLAGS)

lbm_lib.o: lbm_lib.cxx
	$(CPP_CC) -c lbm_lib.cxx $(CPP_FLAGS)

lbm_cuda_lib.o:  lbm_cuda_lib.cu
	$(NV) -c lbm_cuda_lib.cu $(NV_FLAGS)

clean:
	rm *.o  $(TARGET)