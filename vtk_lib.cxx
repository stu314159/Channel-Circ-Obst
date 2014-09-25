#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <cmath>

using namespace std;

void SaveVTKStructuredGridVectorAndMagnitude_ascii(float * u,
						   float * v,
						   float * w,
						   float * x,
						   float * y,
						   float * z,
						   string fileName,
						   string vectorName,
						   int * dims){
  FILE * pFile;
  float * vecMag;
  int numEl = dims[0]*dims[1]*dims[2];
  vecMag = new float[numEl];
  pFile=fopen(fileName.c_str(),"w");
  fprintf(pFile,"# vtk DataFile Version 3.0 \n");
  fprintf(pFile,"Slow ASCII version...\n");
  fprintf(pFile,"ASCII \n");
  fprintf(pFile,"\n");
  fprintf(pFile,"DATASET STRUCTURED_GRID \n");
  fprintf(pFile,"DIMENSIONS %d  %d  %d \n",dims[0],dims[1],dims[2]);
  fprintf(pFile,"POINTS %d float \n",numEl);
  for(int nd=0;nd<(numEl);nd++){
    fprintf(pFile,"%4.3f  %4.3f  %4.3f \n",x[nd],y[nd],z[nd]);
  }
  fprintf(pFile,"\n POINT_DATA %d \n",numEl);
  fprintf(pFile,"VECTORS  %s  float \n",vectorName.c_str());
  for(int nd=0;nd<(numEl);nd++){
    fprintf(pFile,"%4.3f  %4.3f  %4.3f \n",u[nd],v[nd],w[nd]);
    vecMag[nd]=sqrt(u[nd]*u[nd]+v[nd]*v[nd]+w[nd]*w[nd]);
  }
  string magName = vectorName+"Magnitude";
  fprintf(pFile,"SCALARS %s  float \n",magName.c_str());
  fprintf(pFile,"LOOKUP_TABLE default \n");
  for(int nd=0;nd<numEl;nd++){
    fprintf(pFile,"%4.3f \n",vecMag[nd]);
  }

  fclose(pFile);
  delete [] vecMag;

}

void SaveVTKImageData_ascii(float * array, string fileName, 
			    string scalarName, float * origin,
			    float * spacing, int * dims){

  FILE * pFile;
  pFile = fopen(fileName.c_str(),"w");
  fprintf(pFile,"# vtk DataFile Version 3.0 \n");
  fprintf(pFile,"Slow ASCII version...\n");
  fprintf(pFile,"ASCII \n");
  fprintf(pFile,"\n");
  fprintf(pFile,"DATASET STRUCTURED_POINTS \n");
  fprintf(pFile,"DIMENSIONS %d  %d  %d \n",dims[0],dims[1],dims[2]);
  fprintf(pFile,"\n");
  fprintf(pFile,"ORIGIN  %4.3f   %4.3f  %4.3f \n",origin[0],
	  origin[1],origin[2]);
  fprintf(pFile,"SPACING  %4.3f  %4.3f  %4.3f \n",spacing[0],spacing[1],
	  spacing[2]);
  fprintf(pFile,"\n");
  fprintf(pFile,"POINT_DATA %d  \n",dims[0]*dims[1]*dims[2]);
  fprintf(pFile,"SCALARS \t %s  \t float \n",scalarName.c_str());
  fprintf(pFile,"LOOKUP_TABLE default \n \n");

  for(int z=0;z<dims[2];z++){
    for(int y=0;y<dims[1];y++){
      for(int x=0;x<dims[0];x++){
  	fprintf(pFile," %g  ",array[x+y*dims[0]+z*dims[0]*dims[1]]);
      }
      fprintf(pFile,"\n");
    }
  }

  fclose(pFile);

}



