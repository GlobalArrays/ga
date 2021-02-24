#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include <cuda_runtime.h>

int numDevices()
{
  int ngpus;
  cudaGetDeviceCount(&ngpus);
  return ngpus;
}
