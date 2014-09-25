#ifndef VTK_LIB_H
#define VTK_LIB_H

#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>

using namespace std;

void SaveVTKStructuredGridVectorAndMagnitude_ascii(float * u,
						   float * v,
						   float * w,
						   float * x,
						   float * y,
						   float * z,
						   string fileName,
						   string vectorName,
						   int * dims);

void SaveVTKImageData_ascii(float * array, string fileName, 
			    string scalarName, float * origin,
			    float * spacing, int * dims);
#endif
