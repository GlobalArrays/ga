/* $Id: acc.c,v 1.7 1999-07-28 00:47:52 d3h325 Exp $ */
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
   
