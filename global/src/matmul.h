/* $Id: matmul.h,v 1.15.4.1 2006-12-22 13:05:22 manoj Exp $ */
#ifndef _MATMUL_H_
#define _MATMUL_H_

#include "ga.h"
#include "globalp.h"
#include "message.h"
#include "base.h"

#if HAVE_MATH_H
#   include <math.h>
#endif
#include "armci.h"

#define sgemm_ F77_FUNC(sgemm,SGEMM)
#define dgemm_ F77_FUNC(dgemm,DGEMM)
#define zgemm_ F77_FUNC(zgemm,ZGEMM)
#define cgemm_ F77_FUNC(cgemm,CGEMM)

#if F2C_HIDDEN_STRING_LENGTH_AFTER_ARGS
extern void cgemm_(char*, char*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, int, int);
extern void dgemm_(char*, char*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, int, int);
extern void sgemm_(char*, char*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, int, int);
extern void zgemm_(char*, char*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, int, int);
#else
extern void cgemm_(char*, int, char*, int, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*);
extern void dgemm_(char*, int, char*, int, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*);
extern void sgemm_(char*, int, char*, int, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*);
extern void zgemm_(char*, int, char*, int, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*);
#endif

/* min acceptable amount of memory (in elements) and default chunk size */
#  define MINMEM 64
#  define CHUNK_SIZE 256
#  define MAX_CHUNKS 1024
#  define BLOCK_SIZE 1024 /* temp buf size for pinning */
#  define GA_ASPECT_RATIO 3
#  define NUM_MATS 3 
#  define MINTASKS 10 /* increase this if there is high load imbalance */
#  define EXTRA 4

#define MIN_CHUNK_SIZE 256

#define SET   1
#define UNSET 0

extern void gai_matmul_patch_flag(int flag);

Integer gNbhdlA[2], gNbhdlB[2], gNbhdlC[2];/* for A and B matrix */
typedef struct {
  int lo[2]; /* 2 elements: ilo and klo */
  int hi[2];
  int dim[2];
  int chunkBId;
  short int do_put;
}task_list_t;

#define VECTORCHECK(rank,dims,dim1,dim2, ilo, ihi, jlo, jhi) \
  if(rank>2)  pnga_error("rank is greater than 2",rank); \
  else if(rank==2) {dim1=dims[0]; dim2=dims[1];} \
  else if(rank==1) {if((ihi-ilo)>0) { dim1=dims[0]; dim2=1;} \
                    else { dim1=1; dim2=dims[0];}} \
  else pnga_error("rank must be atleast 1",rank);

#define WAIT_GET_BLOCK(nbhdl) pnga_nbwait(nbhdl)

#endif /* _MATMUL_H_ */
