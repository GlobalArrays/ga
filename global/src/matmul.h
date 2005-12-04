/* $Id: matmul.h,v 1.15 2005-12-04 07:18:21 manoj Exp $ */
#ifndef _MATMUL_H_
#define _MATMUL_H_

#include "ga.h"
#include "global.h"
#include "globalp.h"
#include "message.h"
#include "base.h"

#include <math.h>
#ifdef WIN32
#include <armci.h>
#else
#include <../../armci/src/armci.h>
#endif

#ifdef KSR
#  define dgemm_ sgemm_
#  define zgemm_ cgemm_
#endif

#ifdef CRAY
#      include <fortran.h>
#      define  DGEMM SGEMM
#      define  ZGEMM CGEMM
#elif defined(WIN32)
extern void FATR DGEMM(char*,int, char*,int, Integer*, Integer*, Integer*,
		       void*, void*, Integer*, void*, Integer*, void*,
		       void*, Integer*);
extern void FATR ZGEMM(char*,int, char*,int, Integer*, Integer*, Integer*,
		       DoubleComplex*, DoubleComplex*, Integer*, DoubleComplex*,
		       Integer*, DoubleComplex*, DoubleComplex*, Integer*);
#elif defined(F2C2__)
#      define DGEMM dgemm__
#      define ZGEMM zgemm__
#elif defined(HITACHI)
#      define dgemm_ DGEMM
#      define zgemm_ ZGEMM
#endif

#if defined(CRAY) || defined(WIN32)
#   define cptofcd(fcd)  _cptofcd((fcd),1)
#else
#      define cptofcd(fcd) (fcd)
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

static int _gai_matmul_patch_flag = 0; 
Integer gNbhdlA[2], gNbhdlB[2], gNbhdlC[2];/* for A and B matrix */
typedef struct {
  int lo[2]; /* 2 elements: ilo and klo */
  int hi[2];
  int dim[2];
  int chunkBId;
  short int do_put;
}task_list_t;

extern void FATR  ga_nbget_(Integer *g_a, Integer *ilo, Integer *ihi, 
			    Integer *jlo, Integer *jhi, Void *buf, 
			    Integer *ld, Integer *nbhdl);

#define VECTORCHECK(rank,dims,dim1,dim2, ilo, ihi, jlo, jhi) \
  if(rank>2)  ga_error("rank is greater than 2",rank); \
  else if(rank==2) {dim1=dims[0]; dim2=dims[1];} \
  else if(rank==1) {if((ihi-ilo)>0) { dim1=dims[0]; dim2=1;} \
                    else { dim1=1; dim2=dims[0];}} \
  else ga_error("rank must be atleast 1",rank);

#define WAIT_GET_BLOCK(nbhdl) ga_nbwait_(nbhdl)

#endif /* _MATMUL_H_ */
