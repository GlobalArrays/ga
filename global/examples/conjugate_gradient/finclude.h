#define f_matvecmul f_matvecmul_
#define f_computeminverse f_computeminverse_
#define f_computeminverser f_computeminverser_
#define f_addvec f_addvec_
#define f_2addvec f_2addvec_

extern void f_matvecmul(double *,double *,double *,int *,int *,int *,int *,int*);
extern void f_computeminverse(double *,double *,int *,int *,int *,int *); 
extern void f_computeminverser(double *,double *,double *,int *,int *); 
extern void f_addvec(double *,double *,double *,double *,double *, int*, int*);
extern void f_2addvec(double *,double *,double *,double *,double *,double *,double *,double *,double *,double *,int *,int *);
