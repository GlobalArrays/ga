#ifndef _TEST_UTILITIES_H
#define _TEST_UTILITIES_H
#include <math.h>
/**
 * Factor communicator size into a processor grid of dimension
 * ndim
 * @param ndim dimension of processor grid
 * @param size number of processors in grid
 * @param pdims array of processor dimensions
 */
void factor(int ndim, int size, int *pdims)
{
  /* find all primes between 2 and sqrt(size) */
  int ip = static_cast<int>(sqrt(static_cast<double>(size)))+1;
  std::vector<int> prime;
  int pmax = 0;
  int i, j;
  bool chk;
  for (i=2; i<=ip; i++) {
    chk = true;
    for (j=0; j<pmax; j++) {
      if (i%prime[j] == 0) {
        chk = false;
        break;
      }
    }
    if (chk) {
      pmax++;
      prime.push_back(i);
    }
  }
  /* find all prime factors of size */
  ip = size;
  int ifac = 0;
  std::vector<int> fac;
  for (i=0; i<pmax; i++) {
    while(ip%prime[i] == 0) {
      ifac++;
      fac.push_back(prime[i]);
      ip = ip/prime[i];
    }
  }
  /* size is prime */
  if (ifac == 0) {
    ifac++;
    fac.push_back(size);
  }
  for (i=0; i<ndim; i++)  {
    pdims[i] = 1;
  }
  for (i=fac.size()-1; i>=0; i--) {
    /* find smallest value in pdims */  
    int il = 0;
    for (j=1; j<ndim; j++) {
      if (pdims[j]<pdims[il]) il = j;
    }
    /* multiply pdims[il] by fac[i] */
    pdims[il] *= fac[i];
  }
  /* check for consistency */
  ifac = 1;
  for (i=0; i<ndim; i++) {
    ifac *= pdims[i];
  }
  if (ifac != size) {
    printf("Grid factorization doesn't match size\n");
  }
}
#endif
