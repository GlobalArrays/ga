/* $Id: acc.c,v 1.8 2000-04-17 22:31:36 d3h325 Exp $ */
void  L_ACCUMULATE_2D(long* alpha, int* rows, int* cols, long* a, 
                      int* lda, long* b, int* ldb)
{
int i,j;

   for(j=0;j< *cols; j++){
     long *aa = a + j* *lda;
     long *bb = b + j* *ldb;
     for(i=0;i< *rows; i++)
       aa[i] += *alpha * bb[i];
   }
}
   
#ifdef NOFORT

typedef struct {
  float imag;
  float real;
} cmpl_t;

typedef struct {
  double imag;
  double real;
} dcmpl_t;

void  I_ACCUMULATE_2D(int* alpha, int* rows, int* cols, int* a,
                      int* lda, int* b, int* ldb)
{
int i,j;

   for(j=0;j< *cols; j++){
     int *aa = a + j* *lda;
     int *bb = b + j* *ldb;
     for(i=0;i< *rows; i++)
       aa[i] += *alpha * bb[i];
   }
}

void  F_ACCUMULATE_2D(float* alpha, int* rows, int* cols, float* a,
                      int* lda, float* b, int* ldb)
{
int i,j;

   for(j=0;j< *cols; j++){
     float *aa = a + j* *lda;
     float *bb = b + j* *ldb;
     for(i=0;i< *rows; i++)
       aa[i] += *alpha * bb[i];
   }
}


void  D_ACCUMULATE_2D(double* alpha, int* rows, int* cols, double* a,
                      int* lda, double* b, int* ldb)
{
int i,j;

   for(j=0;j< *cols; j++){
     double *aa = a + j* *lda;
     double *bb = b + j* *ldb;
     for(i=0;i< *rows; i++)
       aa[i] += *alpha * bb[i];
   }
}


void  C_ACCUMULATE_2D(cmpl_t* alpha, int* rows, int* cols, cmpl_t* a,
                      int* lda, cmpl_t* b, int* ldb)
{
int i,j;

   for(j=0;j< *cols; j++){
     cmpl_t *aa = a + j* *lda;
     cmpl_t *bb = b + j* *ldb;
     for(i=0;i< *rows; i++){
       aa[i].real  += alpha->real * bb[i].real - alpha->imag * bb[i].imag;
       aa[i].imag  += alpha->imag * bb[i].real + alpha->real * bb[i].imag;
     }
   }
}


void  Z_ACCUMULATE_2D(dcmpl_t* alpha, int* rows, int* cols, dcmpl_t* a,
                      int* lda, dcmpl_t* b, int* ldb)
{
int i,j;


   for(j=0;j< *cols; j++){
     dcmpl_t *aa = a + j* *lda;
     dcmpl_t *bb = b + j* *ldb;
     for(i=0;i< *rows; i++){
       aa[i].real  += alpha->real * bb[i].real - alpha->imag * bb[i].imag;
       aa[i].imag  += alpha->imag * bb[i].real + alpha->real * bb[i].imag;
     }
   }
}


#endif
