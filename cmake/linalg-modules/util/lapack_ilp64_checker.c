#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif
  void DSYEV_NAME(const char*, const char*, int32_t*, double*, int32_t*, double*, double*, int32_t*, int32_t*);
#ifdef __cplusplus
}
#endif

int main() {

  int32_t fake_two[2] = {2,1}; // Will be interpreted as 2 in LP64 and 2^32+2 in ILP64
  int32_t true_two[2] = {2,0}; // Will be interpreted as 2 in both LP64 and ILP64

  double A[4] = {1.,1.,1.,1.};
  double W[2];
  int32_t LWORK[2] = {-1,-1}; // Will be interpreted as -1 in both LP64 and ILP64
  int32_t INFO[2];           // Will store the correct INFO in INFO[1]
  double WORK;

  int32_t* fake_two_ptr = fake_two;
  int32_t* true_two_ptr = true_two;

  // In an LP64 LAPACK Library, the SYEV will be interpreted as N=LDA=LDA=2
  // In an ILP64 BLAS Library, the GEMM will be interpreted as LDA=2 and N=LARGE
  //   Because LDA < N, etc in ILP64, the DSYEV fails by the standard
  //   Valid for MKL, OpenBLAS, NETLIB (Reference) LAPACK, and ESSL
  DSYEV_NAME( "N","L", fake_two_ptr, A, true_two_ptr, W, &WORK, LWORK, INFO );

  if( INFO[0] == 0 ) return 0;
  else               return -1;
};

