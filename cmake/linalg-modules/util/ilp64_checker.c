#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif
  void DGEMM_NAME(const char*, const char*, int32_t*,int32_t*,int32_t*, double*, double*, int32_t*, double*, int32_t*, double*, double*, int32_t*);
#ifdef __cplusplus
}
#endif

int main() {

  int32_t fake_two[2] = {2,1}; // Will be interpreted as 2 in LP64 and 2^32+2 in ILP64
  int32_t true_two[2] = {2,0}; // Will be interpreted as 2 in both LP64 and ILP64

  double A[4] = {1.,1.,1.,1.};
  double B[4] = {1.,1.,1.,1.};
  double C[4] = {0.,0.,0.,0.};

  double zero = 0., one_d = 1.;
  int32_t* fake_two_ptr = fake_two;
  int32_t* true_two_ptr = true_two;

  // In an LP64 BLAS Library, the GEMM will be interpreted as N=M=K=LDA=LDB=LDA=2
  // In an ILP64 BLAS Library, the GEMM will be interpreted as LDA=LDB=LDC=2 and M=N=K=LARGE
  //   Because LDA < M, etc in ILP64, the GEMM fails by the standard
  //   Valid for MKL, BLIS, OpenBLAS, NETLIB (Reference) BLAS, and ESSL
  DGEMM_NAME( "N","N",
          fake_two_ptr, fake_two_ptr, fake_two_ptr, 
          &one_d, A, true_two_ptr, B, true_two_ptr, 
          &zero, C, true_two_ptr );

  // Print out result, will either be zero or it work print due to GEMM abort
  double sum = 0.;
  int i;
  for( i = 0; i < 4; ++i ) sum += C[i];
  //printf("BLAS LP64 CHECK = %.1f\n", sum);

  if( ((int)sum) == 8 ) return 0;
  else                  return 1;

  return 0;
};

