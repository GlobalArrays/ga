/* $Id: matmul.c,v 1.33 2003-10-23 14:43:18 manoj Exp $ */
/*===========================================================
 *
 *         GA_Dgemm(): Parallel Matrix Multiplication
 *              (i.e.  C = alpha*A*B + beta*C)
 *
 *===========================================================*/

#include "matmul.h"

/* some optimization macros */
#define KCHUNK_OPTIMIZATION 0 /* This Opt performing well only for m=1000;n=1000'k=2000 kinda cases and not for the opposite*/

/* Optimization flags: Initialized everytime in ga_matmul() */
static short int CYCLIC_DISTR_OPT_FLAG  = SET;
static short int CONTIG_CHUNKS_OPT_FLAG = SET;
static short int DIRECT_ACCESS_OPT_FLAG = SET;

static int max3(int ichunk, int jchunk, int kchunk) {
  if(ichunk>jchunk) return MAX(ichunk,kchunk);
  else return MAX(jchunk, kchunk);
}

static void GET_BLOCK(Integer *g_x, task_list_t *chunk, void *buf, 
		      char *trans, Integer *xilo, Integer *xjlo, 
		      Integer *dim_next, Integer *nbhdl) {

    Integer i0, i1, j0, j1;

    if(*trans == 'n' || *trans == 'N') {
       *dim_next = chunk->dim[0];
       i0= *xilo+chunk->lo[0]; i1= *xilo+chunk->hi[0];
       j0= *xjlo+chunk->lo[1]; j1= *xjlo+chunk->hi[1];
    }
    else {
       *dim_next = chunk->dim[1];
       i0= *xjlo+chunk->lo[1]; i1= *xjlo+chunk->hi[1];
       j0= *xilo+chunk->lo[0]; j1= *xilo+chunk->hi[0];
    }

    ga_nbget_(g_x, &i0, &i1, &j0, &j1, buf, dim_next, nbhdl);
}

static int gai_nxtask(int irregular, int g_t) {
    if(irregular) {
       int subscript = 0;
       return NGA_Read_inc(g_t, &subscript, 1);
    }
    else {
       return (++gTaskId);
    }
}

static int set_task_id(short int irregular, Integer nproc) {
    if(irregular) {
       int g_t, ihi, ilo, value=1;  
       g_t = NGA_Create(C_INT, 1, &value, "Atomic Task", NULL);
       if(!g_t) ga_error("Task array creation failed", 0L);
       if(!ga_nodeid_()) {  /* Initialize the task array */
	  value=(int)nproc; 
	  ilo = ihi = 0; 
	  NGA_Put(g_t,&ilo,&ihi,&value,&ihi); 
       }
       ga_sync_();
       return g_t;
    }
    else gTaskId=0; /* Note: this is a static variable */
    return 0;
}

static short int
gai_get_task_list(task_list_t *taskListA, task_list_t *taskListB, 
		  task_list_t *state, Integer istart, Integer jstart,
		  Integer kstart, Integer iend, Integer jend, Integer kend, 
		  Integer Ichunk, Integer Jchunk, Integer Kchunk, 
		  int *max_tasks, Integer *g_a) {
    
    int ii, jj, nloops=0;
    short int do_put, more_chunks_left=0, recovery=0;
    Integer ilo, ihi, jlo, jhi, klo, khi, get_new_B;
    Integer jstart_=jstart, kstart_=kstart;
    
    if(state->lo[0] != -1) recovery = 1;

    nloops = (iend-istart+1)/Ichunk + ( ((iend-istart+1)%Ichunk)?1:0 );
    if(nloops>MAX_CHUNKS) GA_Error("Increase MAX_CHUNKS value in matmul.h",0L);

    if(recovery) jstart_ = state->lo[0]; /* recovering the previous state */
    for(ii=jj=0, jlo = jstart_; jlo <= jend; jlo += Jchunk) {
       jhi = MIN(jend, jlo+Jchunk-1);

       if(recovery) {
	  do_put = state->do_put;
	  kstart_ =  state->lo[1];
       }
       else do_put = SET; /* for 1st shot we can put, instead of accumulate */
       
       for(klo = kstart_; klo <= kend; klo += Kchunk) {
	  khi = MIN(kend, klo+Kchunk-1); 
	  get_new_B = TRUE;
	  
	  /* set it back after the first loop */
	  recovery = 0;
	  jstart_ = jstart;
	  kstart_ = kstart;
	  
	  /* save CURRENT STATE. Saving state before "i" loop helps to avoid 
	     tracking get_new_B, which is hassle in ga_matmul_regular() */
	  if(ii+nloops >= MAX_CHUNKS) {
	     more_chunks_left = 1;
	     state->lo[0]  = jlo;
	     state->lo[1]  = klo;
	     state->do_put   = do_put;
	     break;
	  }
	  
	  for(ilo = istart; ilo <= iend; ilo += Ichunk){ 	     
	     ihi = MIN(iend, ilo+Ichunk-1);
	     taskListA[ii].dim[0] = ihi - ilo + 1; 
	     taskListA[ii].dim[1] = khi - klo + 1;
	     taskListA[ii].lo[0]  = ilo; taskListA[ii].hi[0] = ihi;
	     taskListA[ii].lo[1]  = klo; taskListA[ii].hi[1] = khi;
	     taskListA[ii].do_put = do_put;
	     if(get_new_B) { /* B matrix */
		ihi = MIN(iend, ilo+Ichunk-1);
		taskListB[jj].dim[0] = khi - klo + 1; 
		taskListB[jj].dim[1] = jhi - jlo + 1;
		taskListB[jj].lo[0]  = klo; taskListB[jj].hi[0] = khi;
		taskListB[jj].lo[1]  = jlo; taskListB[jj].hi[1] = jhi;
		get_new_B = FALSE; /* Until J or K change again */
		taskListA[ii].chunkBId = jj;
		++jj;
	     }
	     else taskListA[ii].chunkBId = taskListA[ii-1].chunkBId;
	     ++ii;
	  }
	  if (more_chunks_left) break;
	  do_put = UNSET;
       }
       if (more_chunks_left) break;
    }

    *max_tasks = ii;

    /* Optimization disabled if chunks exceeds buffer space */
    if(more_chunks_left) CYCLIC_DISTR_OPT_FLAG = UNSET;

    if(CYCLIC_DISTR_OPT_FLAG) { /* should not be called for irregular matmul */
       int prow, pcol, offset, me = ga_nodeid_();
       prow = GA[GA_OFFSET + *g_a].nblock[0];
       pcol = GA[GA_OFFSET + *g_a].nblock[1];
       offset = (me/prow + me%prow) % pcol;
       for(jj=0, ilo = istart; ilo <= iend; jj++, ilo += Ichunk)
	  taskListA[jj].do_put = UNSET;
       for(jj=0, ilo = istart; ilo <= iend; jj++, ilo += Ichunk)
	  taskListA[jj+offset].do_put = SET;
    }

    return more_chunks_left;
}

static void gai_get_chunk_size(int irregular,Integer *Ichunk,Integer *Jchunk,
			       Integer *Kchunk,Integer *elems,Integer atype, 
			       Integer m,Integer n,Integer k, short int nbuf,
			       short int memory_flag) {
    double temp;
    Integer min_tasks = MINTASKS; /* Increase tasks if there is load imbalance.
				     This controls the granularity of chunks */
    Integer  max_chunk, nproc=ga_nnodes_(), tmpa, tmpb, tmpc;

    tmpa = *Ichunk;
    tmpb = *Jchunk;
    tmpc = *Kchunk;
    
    if(irregular) {
       temp = (k*(double)(m*(double)n)) / (min_tasks * nproc);
       max_chunk = (Integer)pow(temp, (1.0/3.0) );
       if (max_chunk < MIN_CHUNK_SIZE) max_chunk = MIN_CHUNK_SIZE;  
    }
    else
       max_chunk = (Integer) max3(*Ichunk, *Jchunk, *Kchunk);
    
    if ( max_chunk > CHUNK_SIZE/nbuf) {
       /*if memory if very limited, performance degrades for large matrices
	 as chunk size is very small, which leads to communication overhead)*/
       Integer avail = ga_memory_avail(atype);
       ga_igop(GA_TYPE_GOP, &avail, (Integer)1, "min");
      if(avail<MINMEM && ga_nodeid_()==0) ga_error("NotEnough memory",avail);
      *elems = (Integer)(avail*0.9); /* Donot use every last drop */
      
      /* MAX: get the maximum chunk (or, block) size i.e  */
      max_chunk=MIN(max_chunk, (Integer)(sqrt( (double)((*elems-nbuf*NUM_MATS)/NUM_MATS))));

      if(!irregular && memory_flag==SET) 
	 max_chunk = *Ichunk = *Jchunk = *Kchunk = BLOCK_SIZE;
    
      if(irregular) {
	 /* NOTE:enable this part for regular cases, if later 
	    part of the code is buggy or inefficient */
	 *Ichunk = MIN(m,max_chunk);
	 *Jchunk = MIN(n,max_chunk);
	 *Kchunk = MIN(k,max_chunk);      
      }
      else { /* This part of the code takes care of rectangular chunks and
		most probably gives optimum rectangular chunk size */
	 temp = max_chunk*max_chunk;
	 if(*Ichunk < max_chunk && *Kchunk > max_chunk) {
	    *Kchunk = MIN(*Kchunk,(Integer)(temp/(*Ichunk)));
	    *Jchunk = MIN(*Jchunk,(Integer)(temp/(*Kchunk)));
	 }
	 else if(*Kchunk < max_chunk && *Ichunk > max_chunk) {
	    temp *= 1.0/(*Kchunk);
	    *Ichunk = MIN(*Ichunk,(Integer)temp);
	    *Jchunk = MIN(*Jchunk,(Integer)temp);
	 }
	 else *Ichunk = *Jchunk = *Kchunk = max_chunk;
      }
    }
    else 
       *Ichunk = *Jchunk = *Kchunk = CHUNK_SIZE/nbuf;

    /* Try to use 1-d data transfer & take advantage of zero-copy protocol */
    if(!irregular) {
       if(*Ichunk > tmpa && *Jchunk > tmpb) {
	  *Ichunk = tmpa;
	  *Jchunk = tmpb;
	  *Kchunk = MIN(*Ichunk,*Jchunk);
       }
       else if(CONTIG_CHUNKS_OPT_FLAG) { /* select a contiguous piece */
	  int i=1;/* i should be >=1 , to avoid divide by zero error */
	  temp = max_chunk*max_chunk;
	  if(temp > tmpa) {
	     *Ichunk = tmpa;
	     *Jchunk = (Integer)(temp/(*Ichunk));
	     if(*Jchunk < tmpb) {
		while(tmpb/i > *Jchunk) ++i;
		*Jchunk = tmpb/i;
	     }
	     else *Jchunk = tmpb;
	     *Kchunk = MIN(*Ichunk, *Jchunk);
	  }
       }
    }

    /* Total elements "NUM_MAT" extra elems for safety - just in case */
    *elems = ( nbuf*(*Ichunk)*(*Kchunk) + nbuf*(*Kchunk)*(*Jchunk) + 
	       (*Ichunk)*(*Jchunk) );
    *elems += nbuf*NUM_MATS*sizeof(DoubleComplex)/GAsizeofM(atype);
}

static DoubleComplex* 
gai_get_armci_memory(Integer Ichunk, Integer Jchunk, Integer Kchunk,
		     short int nbuf, Integer atype) {

    DoubleComplex *tmp = NULL;
    Integer elems;

    elems = (Integer) pow((double)BLOCK_SIZE,(double)2);
    elems = nbuf*elems + nbuf*elems + elems; /* A,B,C temporary buffers */
    
    /* add extra elements for safety */
    elems += nbuf*NUM_MATS*sizeof(DoubleComplex)/GAsizeofM(atype);

    /* allocate temporary storage using ARMCI_Malloc */
    if( (Integer) (((double)nbuf)*(Ichunk* Kchunk) + 
		   ((double)nbuf)*(Kchunk* Jchunk) + 
		   Ichunk* Jchunk ) < elems) {
       tmp=(DoubleComplex*)ARMCI_Malloc_local(elems*GAsizeofM(atype));
    }
    return tmp;
}
	  
/************************************
 * Sequential DGEMM 
 *      i.e. BLAS dgemm Routines
 ************************************/

static void GAI_DGEMM(Integer atype, char *transa, char *transb, 
		      Integer idim, Integer jdim, Integer kdim, void *alpha, 
		      DoubleComplex *a, Integer adim, DoubleComplex *b, 
		      Integer bdim, DoubleComplex *c, Integer cdim) {

    int idim_t, jdim_t, kdim_t, adim_t, bdim_t, cdim_t;
    DoubleComplex ZERO;
    
    idim_t=idim; jdim_t=jdim; kdim_t=kdim;
    adim_t=adim; bdim_t=bdim; cdim_t=cdim;
    ZERO.real = 0.; ZERO.imag = 0.;
    
# if (defined(CRAY) || defined(WIN32)) && !defined(GA_C_CORE)
    switch(atype) {
       case C_FLOAT:
	  xb_sgemm(transa, transb, &idim_t, &jdim_t, &kdim_t,
		   (float *)alpha, (float *)a, &adim_t, (float *)b, 
		   &bdim_t, (float *)&ZERO,  (float *)c, &cdim_t);
	  break;
       case C_DBL:
	  DGEMM(cptofcd(transa), cptofcd(transb), &idim, &jdim, &kdim,
		alpha, (double*)a, &adim, (double*)b, &bdim, &ZERO, 
		(double*)c, &cdim);
	  break;
       case C_DCPL:
	  ZGEMM(cptofcd(transa), cptofcd(transb), &idim, &jdim, &kdim,
		(DoubleComplex*)alpha, a, &adim, b, &bdim, &ZERO,c,&cdim);
	  break;
       default:
	  ga_error("ga_matmul_patch: wrong data type", atype);
    }
# else 
    switch(atype) {
       case C_FLOAT:
	  xb_sgemm(transa, transb, &idim_t, &jdim_t, &kdim_t,
		   (float *)alpha, (float *)a, &adim_t, (float *)b, 
		   &bdim_t, (float *)&ZERO,  (float *)c, &cdim_t);
	  break;
       case C_DBL:
#   ifdef GA_C_CORE
	  xb_dgemm(transa, transb, &idim_t, &jdim_t, &kdim_t,
		   alpha, (double *)a, &adim_t, (double *)b, &bdim_t, 
		   (double *)&ZERO,  (double *)c, &cdim_t);
#   else
	  
#     if !defined(HAS_BLAS) && defined(EXT_INT)
	  dgemm_(transa, transb, &idim, &jdim, &kdim,
		 alpha, a, &adim, b, &bdim, &ZERO, c, &cdim, 1, 1);
#     else
	  dgemm_(transa, transb, &idim_t, &jdim_t, &kdim_t,
		 alpha, a, &adim_t, b, &bdim_t, &ZERO, c, &cdim_t, 1, 1);
#     endif
	  
#   endif
	  break;
       case C_DCPL:
#   ifdef GA_C_CORE 
	  xb_zgemm(transa, transb, &idim_t, &jdim_t, &kdim_t,
		   (DoubleComplex *)alpha, a, &adim_t, b, &bdim_t, 
		   &ZERO,  c, &cdim_t);
#   else
	  zgemm_(transa, transb, &idim, &jdim, &kdim,
		 (DoubleComplex*)alpha, a, &adim, b, &bdim, &ZERO, c, 
		 &cdim, 1, 1);
#   endif
      break;
       default:
	  ga_error("ga_matmul_patch: wrong data type", atype);
    }
# endif
}



static void gai_matmul_shmem(transa, transb, alpha, beta, atype,
			     g_a, ailo, aihi, ajlo, ajhi,
			     g_b, bilo, bihi, bjlo, bjhi,
			     g_c, cilo, cihi, cjlo, cjhi,
			     Ichunk, Kchunk, Jchunk, a,b,c, need_scaling)
     
     Integer *g_a, *ailo, *aihi, *ajlo, *ajhi;    /* patch of g_a */
     Integer *g_b, *bilo, *bihi, *bjlo, *bjhi;    /* patch of g_b */
     Integer *g_c, *cilo, *cihi, *cjlo, *cjhi;    /* patch of g_c */
     Integer Ichunk, Kchunk, Jchunk, atype;
     void    *alpha, *beta;
     char    *transa, *transb;
     DoubleComplex *a, *b, *c;
     short int need_scaling;
{

    Integer me = ga_nodeid_();
    Integer get_new_B, loC[2]={0,0}, hiC[2]={0,0}, ld[2];
    Integer i0, i1, j0, j1;
    Integer ilo, ihi, idim, jlo, jhi, jdim, klo, khi, kdim, adim, bdim, cdim;
    int istart, jstart, kstart, iend, jend, kend;
    short int do_put=UNSET, single_task_flag=UNSET;
    DoubleComplex ONE;
    float ONE_F = 1.0;
    ONE.real =1.; ONE.imag =0.; 

    GA_PUSH_NAME("ga_matmul_shmem");

    /* to skip accumulate and exploit data locality:
       get chunks according to "C" matrix distribution*/
    nga_distribution_(g_c, &me, loC, hiC);
    istart = loC[0]-1; iend = hiC[0]-1;
    jstart = loC[1]-1; jend = hiC[1]-1;
    kstart = 0       ; kend = *ajhi-*ajlo;

    if(DIRECT_ACCESS_OPT_FLAG) {
       /* check if there is only one task. If so, then it is contiguous */
       if( (iend-istart+1 <= Ichunk) && (jend-jstart+1 <= Jchunk) &&
	   (kend-kstart+1 <= Kchunk) ) {
	  single_task_flag = SET;
	  nga_access_ptr(g_c, loC, hiC, &c, ld);
       }
    }

    /* loop through columns of g_c patch */
    for(jlo = jstart; jlo <= jend; jlo += Jchunk) { 
       jhi  = MIN(jend, jlo+Jchunk-1);
       jdim = jhi - jlo +1;
     
       /* if beta=0,then for first shot we can do put,instead of accumulate */
       if(need_scaling == UNSET) do_put = SET;
   
       /* loop cols of g_a patch : loop rows of g_b patch*/
       for(klo = kstart; klo <= kend; klo += Kchunk) { 
	  khi = MIN(kend, klo+Kchunk-1);
	  kdim= khi - klo +1;
	  get_new_B = TRUE; /* Each pass thru' outer 2 loops means we 
			       need a different patch of B.*/
	  /*loop through rows of g_c patch */
	  for(ilo = istart; ilo <= iend; ilo += Ichunk){ 
	     ihi = MIN(iend, ilo+Ichunk-1);
	     idim= cdim = ihi - ilo +1;
	 
	     /* STEP1(a): get matrix "A" chunk */
	     i0= *ailo+ilo; i1= *ailo+ihi;
	     j0= *ajlo+klo; j1= *ajlo+khi;
	     if (*transa == 'n' || *transa == 'N'){
		adim=idim; ga_get_(g_a, &i0, &i1, &j0, &j1, a, &idim);
	     }else{
		adim=kdim; ga_get_(g_a, &j0, &j1, &i0, &i1, a, &kdim);
	     }

	     /* STEP1(b): get matrix "B" chunk*/
	     if(get_new_B) {/*Avoid rereading B if same patch as last time*/
		i0= *bilo+klo; i1= *bilo+khi;
		j0= *bjlo+jlo; j1= *bjlo+jhi;
		if (*transb == 'n' || *transb == 'N'){ 
		   bdim=kdim; ga_get_(g_b, &i0, &i1, &j0, &j1, b, &kdim);  
		}else {
		   bdim=jdim; ga_get_(g_b, &j0, &j1, &i0, &i1, b, &jdim);
		}
		get_new_B = FALSE; /* Until J or K change again */
	     }
	 
	     /* STEP2: Do the sequential matrix multiply - i.e.BLAS dgemm */
	     GAI_DGEMM(atype, transa, transb, idim, jdim, kdim, alpha, 
		       a, adim, b, bdim, c, cdim);

	     /* STEP3: put/accumulate into "C" matrix */
	     i0= *cilo+ilo; i1= *cilo+ihi;   
	     j0= *cjlo+jlo; j1= *cjlo+jhi;	 
	     /* if single_task_flag is SET (i.e =1), then there is no need to 
		update "C" matrix, as we use pointer directly in GAI_DGEMM */
	     if(single_task_flag != SET) {
		switch(atype) {
		   case C_FLOAT:
		      if(do_put==SET) /* i.e.beta == 0.0 */
			 ga_put_(g_c, &i0, &i1, &j0, &j1, (float *)c, &cdim);
		      else
			 ga_acc_(g_c, &i0, &i1, &j0, &j1, (float *)c, &cdim, 
				 &ONE_F);
		      break;
		   default:
		      if(do_put==SET) /* i.e.beta == 0.0 */
			 ga_put_(g_c, &i0, &i1, &j0, &j1, (DoublePrecision*)c,
				 &cdim);
		      else
			 ga_acc_(g_c, &i0, &i1, &j0, &j1, (DoublePrecision*)c,
				 &cdim, (DoublePrecision*)&ONE);
		      break;
		}
	     }
	  }
	  do_put = UNSET; /* In the second loop, accumulate should be done */
       }
    }
    GA_POP_NAME;
}




static void gai_matmul_regular(transa, transb, alpha, beta, atype,
			       g_a, ailo, aihi, ajlo, ajhi,
			       g_b, bilo, bihi, bjlo, bjhi,
			       g_c, cilo, cihi, cjlo, cjhi,
			       Ichunk, Kchunk, Jchunk, a_ar,b_ar,c_ar, 
			       need_scaling, irregular) 
     
     Integer *g_a, *ailo, *aihi, *ajlo, *ajhi;    /* patch of g_a */
     Integer *g_b, *bilo, *bihi, *bjlo, *bjhi;    /* patch of g_b */
     Integer *g_c, *cilo, *cihi, *cjlo, *cjhi;    /* patch of g_c */
     Integer Ichunk, Kchunk, Jchunk, atype;
     void    *alpha, *beta;
     char    *transa, *transb;
     DoubleComplex **a_ar, **b_ar, **c_ar;
     short int need_scaling, irregular;
{
  
    Integer me= ga_nodeid_(), nproc=ga_nnodes_();
    Integer get_new_B, i, i0, i1, j0, j1;
    Integer ilo, ihi, idim, jlo, jhi, jdim, klo, khi, kdim;
    Integer n, m, k, adim, bdim, cdim, adim_next, bdim_next;
    Integer loC[2]={1,1}, hiC[2]={1,1}, ld[2];
    int g_t, max_tasks=0, shiftA=0, shiftB=0;
    int currA, nextA, currB, nextB; /* "current" and "next" task Ids */
    task_list_t taskListA[MAX_CHUNKS], taskListB[MAX_CHUNKS], state; 
    short int do_put=UNSET, single_task_flag=UNSET, chunks_left=0;
    DoubleComplex ONE, *a, *b, *c;
    float ONE_F = 1.0;
    int offset=0;

    GA_PUSH_NAME("ga_matmul_regular");
    if(irregular) ga_error("irregular flag set", 0L);

    ONE.real =1.; ONE.imag =0.;   
    m = *aihi - *ailo +1;
    n = *bjhi - *bjlo +1;
    k = *ajhi - *ajlo +1;
    state.lo[0] = -1; /* just for first do-while loop */

    do {

       /* Inital Settings */
       a = a_ar[0];
       b = b_ar[0];
       c = c_ar[0];
       do_put = single_task_flag = UNSET;
       offset = 0;
       
       /*****************************************************************
	* Task list: Collect information of all chunks. Matmul using 
	* Non-blocking call needs this list 
	*****************************************************************/
       g_t = set_task_id(irregular, nproc);
       
       /* to skip accumulate and exploit data locality:
	  get chunks according to "C" matrix distribution*/
       nga_distribution_(g_c, &me, loC, hiC);
       chunks_left=gai_get_task_list(taskListA, taskListB, &state,loC[0]-1,
				     loC[1]-1, 0, hiC[0]-1, hiC[1]-1, k-1,
				     Ichunk,Jchunk,Kchunk, &max_tasks,g_a);
       currA = nextA = 0;
       
       if(chunks_left) { /* then turn OFF this optimization */
	  if(DIRECT_ACCESS_OPT_FLAG) {
	     /* check if there is only one task.If so,then it is contiguous */
	     if(max_tasks == 1) {
		if( !((hiC[0]-loC[0]+1 <= Ichunk) &&(hiC[1]-loC[1]+1 <=Jchunk)
		      && (k <= Kchunk))) 
		   ga_error("Invalid task list", 0L);
		single_task_flag = SET;
		nga_access_ptr(g_c, loC, hiC, &c, ld);
	     }
	  }
       }
       
       if(CYCLIC_DISTR_OPT_FLAG) {
	  int prow,pcol;
	  prow = GA[GA_OFFSET + *g_a].nblock[0];
	  pcol = GA[GA_OFFSET + *g_a].nblock[1];
	  offset = (me/prow + me%prow) % pcol;
	  currA = nextA = nextA + offset;
       }
       
       /*************************************************
	* Do the setup & issue non-blocking calls to get 
	* the first block/chunk I'm gonna work 
	*************************************************/
       shiftA=0; shiftB=0;
       if(nextA < max_tasks) {
	  currB = nextB = taskListA[currA].chunkBId;
	  
	  GET_BLOCK(g_a, &taskListA[nextA], a_ar[shiftA], transa, 
		    ailo, ajlo, &adim_next, &gNbhdlA[shiftA]);
	  
	  GET_BLOCK(g_b, &taskListB[nextB], b_ar[shiftB], transb,
		    bilo, bjlo, &bdim_next, &gNbhdlB[shiftB]);
	  
	  adim=adim_next; bdim=bdim_next;
	  get_new_B = TRUE;
       }
       
       /*************************************************************
	* Main Parallel DGEMM Loop.
	*************************************************************/
       while(nextA < max_tasks) {
	  currA = nextA;
	  currB = taskListA[currA].chunkBId;
	  
	  idim = cdim = taskListA[currA].dim[0];
	  jdim = taskListB[currB].dim[1];
	  kdim = taskListA[currA].dim[1];
	  bdim=bdim_next;
	  
	  /* if beta=0.0 (i.e.if need_scaling=UNSET), then for first shot,
	     we can do put, instead of accumulate */
	  if(need_scaling == UNSET) do_put = taskListA[currA].do_put; 
	  
	  nextA = gai_nxtask(irregular, g_t); /* get the next task id */
	  
	  if(CYCLIC_DISTR_OPT_FLAG && nextA < max_tasks) 
	     nextA = (offset+nextA) % max_tasks;
	  
	  
	  /* ---- WAIT till we get the current A & B block ---- */
	  a = a_ar[shiftA];
	  WAIT_GET_BLOCK(&gNbhdlA[shiftA]);
	  if(get_new_B){/*Avoid rereading B if it is same patch as last time*/
	     get_new_B = FALSE;
	     b = b_ar[shiftB];
	     WAIT_GET_BLOCK(&gNbhdlB[shiftB]);
	  }
	  
	  /* ---- GET the next A & B block ---- */
	  if(nextA < max_tasks) {
	     GET_BLOCK(g_a, &taskListA[nextA], a_ar[(shiftA+1)%2], transa, 
		       ailo, ajlo, &adim_next, &gNbhdlA[(shiftA+1)%2]);
	     
	     nextB = taskListA[nextA].chunkBId;
	     if(currB != nextB) {
		shiftB=((shiftB+1)%2);
		
		GET_BLOCK(g_b, &taskListB[nextB], b_ar[shiftB], transb, 
			  bilo, bjlo, &bdim_next, &gNbhdlB[shiftB]);
	     }
	  }
	  if(currB != nextB) get_new_B = TRUE;
	  
	  /* Do the sequential matrix multiply - i.e.BLAS dgemm */
	  GAI_DGEMM(atype, transa, transb, idim, jdim, kdim, alpha, 
		    a, adim, b, bdim, c, cdim);
	  
	  /* Non-blocking Accumulate Operation. Note: skip wait in 1st loop*/
	  i0 = *cilo + taskListA[currA].lo[0];
	  i1 = *cilo + taskListA[currA].hi[0];
	  j0 = *cjlo + taskListB[currB].lo[1];
	  j1 = *cjlo + taskListB[currB].hi[1];
	  
	  if(currA < max_tasks) {
	     if (single_task_flag != SET) {
		switch(atype) {
		   case C_FLOAT:
		      if(do_put==SET) /* Note:do_put is UNSET, if beta!=0.0*/
			 ga_put_(g_c, &i0, &i1, &j0, &j1, (float *)c, &cdim);
		      else
			 ga_acc_(g_c, &i0, &i1, &j0, &j1, (float *)c, 
				 &cdim, &ONE_F);
		      break;
		   default:
		      if(do_put==SET) /* i.e.beta ==0.0 */
			 ga_put_(g_c, &i0, &i1, &j0, &j1, (DoublePrecision*)c, 
				 &cdim);
		      else
			 ga_acc_(g_c, &i0, &i1, &j0, &j1, (DoublePrecision*)c, 
				 &cdim,(DoublePrecision*)&ONE);
		      break;
		}
	     }
	  }
	  
	  /* shift next buffer..toggles between 0 and 1: as we use 2 buffers, 
	     one for computation and the other for communication (overlap) */
	  shiftA = ((shiftA+1)%2); 
	  adim = adim_next;
       }
    } while(chunks_left);
   
    GA_POP_NAME;
}



static void gai_matmul_irreg(transa, transb, alpha, beta, atype,
			     g_a, ailo, aihi, ajlo, ajhi,
			     g_b, bilo, bihi, bjlo, bjhi,
			     g_c, cilo, cihi, cjlo, cjhi,
			     Ichunk, Kchunk, Jchunk, a_ar,b_ar,c_ar, 
			     need_scaling, irregular) 
     
     Integer *g_a, *ailo, *aihi, *ajlo, *ajhi;    /* patch of g_a */
     Integer *g_b, *bilo, *bihi, *bjlo, *bjhi;    /* patch of g_b */
     Integer *g_c, *cilo, *cihi, *cjlo, *cjhi;    /* patch of g_c */
     Integer Ichunk, Kchunk, Jchunk, atype;
     void    *alpha, *beta;
     char    *transa, *transb;
     DoubleComplex **a_ar, **b_ar, **c_ar;
     short int need_scaling, irregular;
{
  
    Integer me= ga_nodeid_(), nproc=ga_nnodes_();
    Integer get_new_B, i, i0, i1, j0, j1;
    Integer ilo, ihi, idim, jlo, jhi, jdim, klo, khi, kdim, ijk=0;
    Integer n, m, k, adim, bdim, cdim;
    Integer idim_prev, jdim_prev, kdim_prev, adim_prev, bdim_prev, cdim_prev;
    task_list_t taskListC; 
    short int compute_flag=0, shiftA=0, shiftB=0;
    DoubleComplex ONE, *a, *b, *c;
    float ONE_F = 1.0;
 
    GA_PUSH_NAME("ga_matmul_irreg");

    ONE.real =1.; ONE.imag =0.;
   
    m = *aihi - *ailo +1;
    n = *bjhi - *bjlo +1;
    k = *ajhi - *ajlo +1;
    a = a_ar[0];
    b = b_ar[0];
    c = c_ar[0];
    
    if(need_scaling) ga_scale_patch_(g_c, cilo, cihi, cjlo, cjhi, beta);
    else  ga_fill_patch_(g_c, cilo, cihi, cjlo, cjhi, beta);

    /* take care of the last chunk */
    compute_flag=0;
    for(jlo = 0; jlo < n; jlo += Jchunk){ /* loop thru columns of g_c patch */
       jhi = MIN(n-1, jlo+Jchunk-1);
       jdim= jhi - jlo +1;

       for(klo = 0; klo < k; klo += Kchunk){    /* loop cols of g_a patch */
	  khi = MIN(k-1, klo+Kchunk-1);          /* loop rows of g_b patch */
	  kdim= khi - klo +1;                                     
	   
	  /** Each pass through the outer two loops means we need a
	      different patch of B.*/
	  get_new_B = TRUE;
	   
	  for(ilo = 0; ilo < m; ilo+=Ichunk){ /* loop thru rows of g_c patch */
	        
	     if(ijk%nproc == me){

		ihi = MIN(m-1, ilo+Ichunk-1);
		idim= cdim = ihi - ilo +1;


		if (*transa == 'n' || *transa == 'N'){ 
		   adim = idim;
		   i0= *ailo+ilo; i1= *ailo+ihi;   
		   j0= *ajlo+klo; j1= *ajlo+khi;
		   ga_nbget_(g_a, &i0, &i1, &j0, &j1, a_ar[shiftA], 
			     &idim, &gNbhdlA[shiftA]);
		}else{
		   adim = kdim;
		   i0= *ajlo+klo; i1= *ajlo+khi;   
		   j0= *ailo+ilo; j1= *ailo+ihi;
		   ga_nbget_(g_a, &i0, &i1, &j0, &j1, a_ar[shiftA],
			     &kdim, &gNbhdlA[shiftA]);
		}
		
		/* Avoid rereading B if it is same patch as last time. */
		if(get_new_B) { 
		   if (*transb == 'n' || *transb == 'N'){ 
		      bdim = kdim;
		      i0= *bilo+klo; i1= *bilo+khi;
		      j0= *bjlo+jlo; j1= *bjlo+jhi;
		      ga_nbget_(g_b, &i0, &i1, &j0, &j1, b_ar[shiftB], 
				&kdim, &gNbhdlB[shiftB]);
		   }else{
		      bdim = jdim;
		      i0= *bjlo+jlo; i1= *bjlo+jhi;   
		      j0= *bilo+klo; j1= *bilo+khi;
		      ga_nbget_(g_b, &i0, &i1, &j0, &j1, b_ar[shiftB], 
				&jdim, &gNbhdlB[shiftB]);
		   }
		}

		if(compute_flag) { /* compute loop */
		   
		   if(atype == C_FLOAT) 
		      for(i=0;i<idim_prev*jdim_prev;i++) *(((float*)c)+i)=0;
		   else if(atype ==  C_DBL)
		      for(i=0;i<idim_prev*jdim_prev;i++) *(((double*)c)+i)=0;
		   else for(i=0;i<idim_prev*jdim_prev;i++) {
		           c[i].real=0;c[i].imag=0; }
		   
		   /* wait till we get the previous block */
		   a = a_ar[shiftA^1];
		   WAIT_GET_BLOCK(&gNbhdlA[shiftA^1]);
		   if(taskListC.chunkBId) {
		      b = b_ar[shiftB^1];
		      WAIT_GET_BLOCK(&gNbhdlB[shiftB^1]);
		   }
		   
		   /* Do the sequential matrix multiply - i.e.BLAS dgemm */
		   GAI_DGEMM(atype, transa, transb, idim_prev, jdim_prev, 
			     kdim_prev, alpha, a, adim_prev, b, bdim_prev, 
			     c, cdim_prev);
		   
		   i0= *cilo + taskListC.lo[0];
		   i1= *cilo + taskListC.hi[0];
		   j0= *cjlo + taskListC.lo[1];
		   j1= *cjlo + taskListC.hi[1];

		   if(atype == C_FLOAT)
		      ga_acc_(g_c, &i0, &i1, &j0, &j1, (float *)c, 
			      &cdim_prev, &ONE_F);
		   else
		      ga_acc_(g_c, &i0, &i1, &j0, &j1, (DoublePrecision*)c, 
			      &cdim_prev, (DoublePrecision*)&ONE);
		}
		compute_flag=1;

		/* meta-data of current block for next compute loop */
		taskListC.lo[0] = ilo; taskListC.hi[0] = ihi;
		taskListC.lo[1] = jlo; taskListC.hi[1] = jhi;
		taskListC.chunkBId = get_new_B;
		idim_prev = idim;   adim_prev = adim;
		jdim_prev = jdim;   bdim_prev = bdim;
		kdim_prev = kdim;   cdim_prev = cdim;

		/* shift bext buffer */
		shiftA ^= 1;
		if(get_new_B) shiftB ^= 1;

		get_new_B = FALSE; /* Until J or K change again */
	     }
	     ++ijk;
	  }
       }
    }

    /* -------- compute the last chunk --------- */
    if(compute_flag) {
       if(atype == C_FLOAT) 
	  for(i=0;i<idim_prev*jdim_prev;i++) *(((float*)c)+i)=0;
       else if(atype ==  C_DBL)
	  for(i=0;i<idim_prev*jdim_prev;i++) *(((double*)c)+i)=0;
       else for(i=0;i<idim_prev*jdim_prev;i++) {
	  c[i].real=0;c[i].imag=0; }
       
       /* wait till we get the previous block */
       a = a_ar[shiftA^1];
       WAIT_GET_BLOCK(&gNbhdlA[shiftA^1]);
       if(taskListC.chunkBId) {
	  b = b_ar[shiftB^1];
	  WAIT_GET_BLOCK(&gNbhdlB[shiftB^1]);
       }
       
       
       /* Do the sequential matrix multiply - i.e.BLAS dgemm */
       GAI_DGEMM(atype, transa, transb, idim_prev, jdim_prev, 
		 kdim_prev, alpha, a, adim_prev, b, bdim_prev, 
		 c, cdim_prev);
       
       i0= *cilo + taskListC.lo[0];
       i1= *cilo + taskListC.hi[0];
       j0= *cjlo + taskListC.lo[1];
       j1= *cjlo + taskListC.hi[1];
       
       if(atype == C_FLOAT)
	  ga_acc_(g_c, &i0, &i1, &j0, &j1, (float *)c, 
		  &cdim_prev, &ONE_F);
       else
	  ga_acc_(g_c, &i0, &i1, &j0, &j1, (DoublePrecision*)c, 
		  &cdim_prev, (DoublePrecision*)&ONE);
    }
    /* ----------------------------------------- */
    GA_POP_NAME;
}



/******************************************
 * PARALLEL DGEMM
 *     i.e.  C = alpha*A*B + beta*C
 ******************************************/
void ga_matmul(transa, transb, alpha, beta,
	       g_a, ailo, aihi, ajlo, ajhi,
	       g_b, bilo, bihi, bjlo, bjhi,
	       g_c, cilo, cihi, cjlo, cjhi)
     
     Integer *g_a, *ailo, *aihi, *ajlo, *ajhi;    /* patch of g_a */
     Integer *g_b, *bilo, *bihi, *bjlo, *bjhi;    /* patch of g_b */
     Integer *g_c, *cilo, *cihi, *cjlo, *cjhi;    /* patch of g_c */
     void    *alpha, *beta;
     char    *transa, *transb;
{
#ifdef STATBUF /* Using static (memory) buffers */
    /* approx. sqrt(2) ratio in chunk size to use the same buffer space */
    DoubleComplex a[ICHUNK*KCHUNK], b[KCHUNK*JCHUNK], c[ICHUNK*JCHUNK];
    DoubleComplex a_ar[2][ICHUNK*KCHUNK], b_ar[2][ICHUNK*KCHUNK],
      c_ar[2][ICHUNK*KCHUNK];
#else
    DoubleComplex *a, *b, *c, *a_ar[2], *b_ar[2], *c_ar[2];
#endif
    Integer adim1, adim2, bdim1, bdim2, cdim1, cdim2, dims[2];
    Integer atype, btype, ctype, rank, me= ga_nodeid_(), nproc = ga_nnodes_();
    Integer n, m, k, Ichunk, Kchunk, Jchunk, ZERO_I = 0, inode, iproc;
    Integer loA[2]={0,0}, hiA[2]={0,0};
    Integer loB[2]={0,0}, hiB[2]={0,0};
    Integer loC[2]={0,0}, hiC[2]={0,0};
    int local_sync_begin,local_sync_end;
    short int need_scaling=SET,use_NB_matmul=SET;
    short int irregular=UNSET, memory_flag=UNSET;
    
    /* OPTIMIZATIONS FLAGS. To unset an optimization, replace SET by UNSET) */
    CYCLIC_DISTR_OPT_FLAG  = SET;
    CONTIG_CHUNKS_OPT_FLAG = SET;
    DIRECT_ACCESS_OPT_FLAG = SET;

    local_sync_begin = _ga_sync_begin; local_sync_end = _ga_sync_end;
    _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
    if(local_sync_begin)ga_sync_();

    GA_PUSH_NAME("ga_matmul");

    /**************************************************
     * Do All Sanity Checks 
     **************************************************/

    /* Check to make sure all global arrays are of the same type */
    if (!(ga_is_mirrored_(g_a) == ga_is_mirrored_(g_b) &&
	  ga_is_mirrored_(g_a) == ga_is_mirrored_(g_c))) {
       ga_error_("Processors do not match for all arrays",ga_nnodes_());
    }
#if 0
    if (ga_is_mirrored_(g_a)) {
       inode = ga_cluster_nodeid_();
       nproc = ga_cluster_nprocs_(&inode);
       iproc = me - ga_cluster_procid_(&inode, &ZERO_I);
    } else {
       nproc = ga_nnodes_();
       iproc = me;
    }
#endif

    /* check if ranks are O.K. */
    nga_inquire_internal_(g_a, &atype, &rank, dims); 
    VECTORCHECK(rank, dims, adim1, adim2, *ailo, *aihi, *ajlo, *ajhi);
    nga_inquire_internal_(g_b, &btype, &rank, dims); 
    VECTORCHECK(rank, dims, bdim1, bdim2, *bilo, *bihi, *bjlo, *bjhi);
    nga_inquire_internal_(g_c, &ctype, &rank, dims); 
    VECTORCHECK(rank, dims, cdim1, cdim2, *cilo, *cihi, *cjlo, *cjhi);

    /* check for data-types mismatch */
    if(atype != btype || atype != ctype ) ga_error(" types mismatch ", 0L);
    if(atype != C_DCPL && atype != C_DBL && atype != C_FLOAT) 
       ga_error(" type error",atype);
   
    /* check if patch indices and dims match */
    if (*transa == 'n' || *transa == 'N'){
       if (*ailo <= 0 || *aihi > adim1 || *ajlo <= 0 || *ajhi > adim2)
	  ga_error("  g_a indices out of range ", *g_a);
    }else
       if (*ailo <= 0 || *aihi > adim2 || *ajlo <= 0 || *ajhi > adim1)
	  ga_error("  g_a indices out of range ", *g_a);
   
    if (*transb == 'n' || *transb == 'N'){
       if (*bilo <= 0 || *bihi > bdim1 || *bjlo <= 0 || *bjhi > bdim2)
	  ga_error("  g_b indices out of range ", *g_b);
    }else
       if (*bilo <= 0 || *bihi > bdim2 || *bjlo <= 0 || *bjhi > bdim1)
	  ga_error("  g_b indices out of range ", *g_b);
   
    if (*cilo <= 0 || *cihi > cdim1 || *cjlo <= 0 || *cjhi > cdim2)
       ga_error("  g_c indices out of range ", *g_c);

    /* verify if patch dimensions are consistent */
    m = *aihi - *ailo +1;
    n = *bjhi - *bjlo +1;
    k = *ajhi - *ajlo +1;
    if( (*cihi - *cilo +1) != m) ga_error(" a & c dims error",m);
    if( (*cjhi - *cjlo +1) != n) ga_error(" b & c dims error",n);
    if( (*bihi - *bilo +1) != k) ga_error(" a & b dims error",k);


    /* switch to various matmul algorithms here. more to come */
    if( GA[GA_OFFSET + *g_c].irreg == 1 ||
	GA[GA_OFFSET + *g_b].irreg == 1 ||
	GA[GA_OFFSET + *g_a].irreg == 1 ||
	_gai_matmul_patch_flag == SET) irregular = SET;
    if(!irregular) {
       if((adim1=GA_Cluster_nnodes()) > 1) use_NB_matmul = SET;
       else use_NB_matmul = UNSET;
#    if defined(__crayx1) || defined(NEC)
       use_NB_matmul = UNSET;
#    endif
    }

    /****************************************************************
     * Get the memory (i.e.static or dynamic) for temporary buffers 
     ****************************************************************/

    /* to skip accumulate and exploit data locality:
       get chunks according to "C" matrix distribution*/
    nga_distribution_(g_a, &me, loA, hiA);
    nga_distribution_(g_b, &me, loB, hiB);
    nga_distribution_(g_c, &me, loC, hiC);

#ifdef STATBUF /* Using static memory */
    if(atype ==  C_DBL || atype == C_FLOAT)
       Ichunk=D_CHUNK, Kchunk=D_CHUNK, Jchunk=D_CHUNK;
    else 
       Ichunk=ICHUNK; Kchunk=KCHUNK; Jchunk=JCHUNK;
#else /* Using "Memory Allocator's" dynamic memory */
       {
	  Integer elems, factor=sizeof(DoubleComplex)/GAsizeofM(atype);
	  short int nbuf=1;
	  DoubleComplex *tmp = NULL;

	  Ichunk = MIN( (hiC[0]-loC[0]+1), (hiA[0]-loA[0]+1) );
	  Jchunk = MIN( (hiC[1]-loC[1]+1), (hiB[1]-loB[1]+1) );
	  Kchunk = MIN( (hiA[1]-loA[1]+1), (hiB[0]-loB[0]+1) );

#if KCHUNK_OPTIMIZATION /*works great for m=1000,n=1000,k=4000 kinda cases*/
	  nga_distribution_(g_a, &me, loC, hiC);
	  Kchunk = hiC[1]-loC[1]+1;
	  nga_distribution_(g_b, &me, loC, hiC);
	  Kchunk = MIN(Kchunk, (hiC[0]-loC[0]+1));
#endif
	  /* If non-blocking, we need 2 temporary buffers for A and B matrix */
	  if(use_NB_matmul) nbuf = 2; 
	  
	  if(!irregular) {
	     tmp = a_ar[0] =a=gai_get_armci_memory(Ichunk,Jchunk,Kchunk,
						   nbuf, atype);
	     if(tmp != NULL) memory_flag = SET;
	  }
	  
	  /* get ChunkSize (i.e.BlockSize), that fits in temporary buffer */
	  gai_get_chunk_size(irregular, &Ichunk, &Jchunk, &Kchunk, &elems, 
			     atype, m, n, k, nbuf, memory_flag);
	  
	  if(tmp == NULL) { /* try once again from armci for new chunk sizes */
	     tmp = a_ar[0] =a=gai_get_armci_memory(Ichunk,Jchunk,Kchunk,
						   nbuf, atype);
	     if(tmp != NULL) memory_flag = SET;
	  }

	  if(tmp == NULL) { /*if armci malloc fails again, then get from MA */
	     tmp = a_ar[0] = a =(DoubleComplex*) ga_malloc(elems,atype,
							   "GA mulmat bufs");
	  }

	  if(use_NB_matmul) tmp = a_ar[1] = a_ar[0] + (Ichunk*Kchunk)/factor+1;
	  
	  tmp = b_ar[0] = b = tmp + (Ichunk*Kchunk)/factor + 1;
	  if(use_NB_matmul) tmp = b_ar[1] = b_ar[0] + (Kchunk*Jchunk)/factor+1;
	  
	  c_ar[0] = c = tmp + (Kchunk*Jchunk)/factor + 1;
       }
#endif  
       
       /** check if there is a need for scaling the data. 
	   Note: if beta=0, then need_scaling=0  */
       if(atype==C_DCPL){
	  if((((DoubleComplex*)beta)->real == 0) && 
	     (((DoubleComplex*)beta)->imag ==0)) need_scaling =0;} 
       else if((atype==C_DBL)){
	  if(*(DoublePrecision *)beta == 0) need_scaling =0;}
       else if( *(float*)beta ==0) need_scaling =0;

       if(need_scaling) ga_scale_patch_(g_c, cilo, cihi, cjlo, cjhi, beta);

       /********************************************************************
	* Parallel Matrix Multiplication Starts Here.
	* 3 Steps:
	*    1. Get a chunk of A and B matrix, and store it in local buffer.
	*    2. Do sequential dgemm.
	*    3. Put/accumulate the result into C matrix.
	*********************************************************************/

       /* if only one node, then enable the optimized shmem code */
       if(use_NB_matmul==UNSET) { 
	  gai_matmul_shmem(transa, transb, alpha, beta, atype,
			   g_a, ailo, aihi, ajlo, ajhi,
			   g_b, bilo, bihi, bjlo, bjhi,
			   g_c, cilo, cihi, cjlo, cjhi,
			   Ichunk, Kchunk, Jchunk, a,b,c, need_scaling);
       }
       else {
	  if(irregular)
	     gai_matmul_irreg(transa, transb, alpha, beta, atype,
			      g_a, ailo, aihi, ajlo, ajhi,
			      g_b, bilo, bihi, bjlo, bjhi,
			      g_c, cilo, cihi, cjlo, cjhi,
			      Ichunk, Kchunk, Jchunk, a_ar, b_ar, c_ar,
			      need_scaling, irregular);
	  else
	     gai_matmul_regular(transa, transb, alpha, beta, atype,
				g_a, ailo, aihi, ajlo, ajhi,
				g_b, bilo, bihi, bjlo, bjhi,
				g_c, cilo, cihi, cjlo, cjhi,
				Ichunk, Kchunk, Jchunk, a_ar, b_ar, c_ar, 
				need_scaling, irregular);
       }
	     
#ifndef STATBUF
       a = a_ar[0];
       if(memory_flag == SET) ARMCI_Free_local(a);
       else ga_free(a);
#endif
   
       GA_POP_NAME;   
       if(local_sync_end)ga_sync_();
}

/* This is the old matmul code. It is enadle now for mirrored matrix multiply. 
   It also work for normal matrix/vector multiply with no changes */
void ga_matmul_mirrored(transa, transb, alpha, beta,
			g_a, ailo, aihi, ajlo, ajhi,
			g_b, bilo, bihi, bjlo, bjhi,
			g_c, cilo, cihi, cjlo, cjhi)

     Integer *g_a, *ailo, *aihi, *ajlo, *ajhi;    /* patch of g_a */
     Integer *g_b, *bilo, *bihi, *bjlo, *bjhi;    /* patch of g_b */
     Integer *g_c, *cilo, *cihi, *cjlo, *cjhi;    /* patch of g_c */
     void    *alpha, *beta;
     char    *transa, *transb;
{

    #ifdef STATBUF
  /* approx. sqrt(2) ratio in chunk size to use the same buffer space */
   DoubleComplex a[ICHUNK*KCHUNK], b[KCHUNK*JCHUNK], c[ICHUNK*JCHUNK];
#else
   DoubleComplex *a, *b, *c;
#endif
Integer atype, btype, ctype, adim1, adim2, bdim1, bdim2, cdim1, cdim2, dims[2], rank;
Integer me= ga_nodeid_(), nproc;
Integer i, ijk = 0, i0, i1, j0, j1;
Integer ilo, ihi, idim, jlo, jhi, jdim, klo, khi, kdim;
Integer n, m, k, adim, bdim, cdim;
Integer Ichunk, Kchunk, Jchunk;
DoubleComplex ONE, ZERO;

DoublePrecision chunk_cube;
Integer min_tasks = 10, max_chunk;
int need_scaling=1;
Integer ZERO_I = 0, inode, iproc;
float ONE_F = 1.0, ZERO_F = 0.0;
double ZERO_D = 0.0;
Integer get_new_B;
int local_sync_begin,local_sync_end;
int idim_t, jdim_t, kdim_t, adim_t, bdim_t, cdim_t;

   ONE.real =1.; ZERO.real =0.;
   ONE.imag =0.; ZERO.imag =0.;

   local_sync_begin = _ga_sync_begin; local_sync_end = _ga_sync_end;
   _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
   if(local_sync_begin)ga_sync_();

   GA_PUSH_NAME("ga_matmul_patch");

   /* Check to make sure all global arrays are of the same type */
   if (!(ga_is_mirrored_(g_a) == ga_is_mirrored_(g_b) &&
        ga_is_mirrored_(g_a) == ga_is_mirrored_(g_c))) {
     ga_error_("Processors do not match for all arrays",ga_nnodes_());
   }
   if (ga_is_mirrored_(g_a)) {
     inode = ga_cluster_nodeid_();
     nproc = ga_cluster_nprocs_(&inode);
     iproc = me - ga_cluster_procid_(&inode, &ZERO_I);
   } else {
     nproc = ga_nnodes_();
     iproc = me;
   }

   nga_inquire_internal_(g_a, &atype, &rank, dims); 
   VECTORCHECK(rank, dims, adim1, adim2, *ailo, *aihi, *ajlo, *ajhi);
   nga_inquire_internal_(g_b, &btype, &rank, dims); 
   VECTORCHECK(rank, dims, bdim1, bdim2, *bilo, *bihi, *bjlo, *bjhi);
   nga_inquire_internal_(g_c, &ctype, &rank, dims); 
   VECTORCHECK(rank, dims, cdim1, cdim2, *cilo, *cihi, *cjlo, *cjhi);

   if(atype != btype || atype != ctype ) ga_error(" types mismatch ", 0L);
   if(atype != C_DCPL && atype != C_DBL && atype != C_FLOAT) 
     ga_error(" type error",atype);
   
   
   
   /* check if patch indices and dims match */
   if (*transa == 'n' || *transa == 'N'){
     if (*ailo <= 0 || *aihi > adim1 || *ajlo <= 0 || *ajhi > adim2)
       ga_error("  g_a indices out of range ", *g_a);
   }else
     if (*ailo <= 0 || *aihi > adim2 || *ajlo <= 0 || *ajhi > adim1)
       ga_error("  g_a indices out of range ", *g_a);
   
   if (*transb == 'n' || *transb == 'N'){
     if (*bilo <= 0 || *bihi > bdim1 || *bjlo <= 0 || *bjhi > bdim2)
       ga_error("  g_b indices out of range ", *g_b);
   }else
     if (*bilo <= 0 || *bihi > bdim2 || *bjlo <= 0 || *bjhi > bdim1)
       ga_error("  g_b indices out of range ", *g_b);
   
   if (*cilo <= 0 || *cihi > cdim1 || *cjlo <= 0 || *cjhi > cdim2)
     ga_error("  g_c indices out of range ", *g_c);
   
   /* verify if patch dimensions are consistent */
   m = *aihi - *ailo +1;
   n = *bjhi - *bjlo +1;
   k = *ajhi - *ajlo +1;
   if( (*cihi - *cilo +1) != m) ga_error(" a & c dims error",m);
   if( (*cjhi - *cjlo +1) != n) ga_error(" b & c dims error",n);
   if( (*bihi - *bilo +1) != k) ga_error(" a & b dims error",k);
   

   /* In 32-bit platforms, k*m*n might exceed the "long" range(2^31), 
      eg:k=m=n=1600. So casting the temporary value to "double" helps */
   chunk_cube = (k*(double)(m*n)) / (min_tasks * nproc);
   max_chunk = (Integer)pow(chunk_cube, (DoublePrecision)(1.0/3.0) );
   if (max_chunk < 32) max_chunk = 32;

#ifdef STATBUF
   if(atype ==  C_DBL || atype == C_FLOAT){
      Ichunk=D_CHUNK, Kchunk=D_CHUNK, Jchunk=D_CHUNK;
   }else{
      Ichunk=ICHUNK; Kchunk=KCHUNK; Jchunk=JCHUNK;
   }
#else
   {
     /**
      * Find out how much memory we can grab.  It will be used in
      * three chunks, and the result includes only the first one.
      */
     
     Integer elems, factor = sizeof(DoubleComplex)/GAsizeofM(atype);
     Ichunk = Jchunk = Kchunk = CHUNK_SIZE;
     
     if ( max_chunk > Ichunk) {       
       /*if memory if very limited, performance degrades for large matrices
	 as chunk size is very small, which leads to communication overhead)*/
       Integer avail = ga_memory_avail(atype);
       ga_igop(GA_TYPE_GOP, &avail, (Integer)1, "min");
       if(avail<MINMEM && ga_nodeid_()==0) ga_error("NotEnough memory",avail);
       elems = (Integer)(avail*0.9); /* Donot use every last drop */
       
       max_chunk=MIN(max_chunk, (Integer)(sqrt( (double)((elems-EXTRA)/3))));
       Ichunk = MIN(m,max_chunk);
       Jchunk = MIN(n,max_chunk);
       Kchunk = MIN(k,max_chunk);
     }
     else /* "EXTRA" elems for safety - just in case */
       elems = 3*Ichunk*Jchunk + EXTRA*factor;
     
     a = (DoubleComplex*) ga_malloc(elems, atype, "GA mulmat bufs");
     b = a + (Ichunk*Kchunk)/factor + 1; 
     c = b + (Kchunk*Jchunk)/factor + 1;
   }
#endif

   if(atype==C_DCPL){if((((DoubleComplex*)beta)->real == 0) &&
	       (((DoubleComplex*)beta)->imag ==0)) need_scaling =0;} 
   else if((atype==C_DBL)){if(*(DoublePrecision *)beta == 0) need_scaling =0;}
   else if( *(float*)beta ==0) need_scaling =0;

   if(need_scaling) ga_scale_patch_(g_c, cilo, cihi, cjlo, cjhi, beta);
   else  ga_fill_patch_(g_c, cilo, cihi, cjlo, cjhi, beta);

   for(jlo = 0; jlo < n; jlo += Jchunk){ /* loop through columns of g_c patch */
       jhi = MIN(n-1, jlo+Jchunk-1);
       jdim= jhi - jlo +1;

       for(klo = 0; klo < k; klo += Kchunk){    /* loop cols of g_a patch */
	 khi = MIN(k-1, klo+Kchunk-1);          /* loop rows of g_b patch */
	 kdim= khi - klo +1;                                     
	 
	 /** Each pass through the outer two loops means we need a
	     different patch of B.*/
	 get_new_B = TRUE;
	 
	 for(ilo = 0; ilo < m; ilo += Ichunk){ /*loop through rows of g_c patch */
	   
	   if(ijk%nproc == iproc){

	     ihi = MIN(m-1, ilo+Ichunk-1);
	     idim= cdim = ihi - ilo +1;
	     
	     if(atype == C_FLOAT) 
	       for (i = 0; i < idim*jdim; i++) *(((float*)c)+i)=0;
	     else if(atype ==  C_DBL)
	       for (i = 0; i < idim*jdim; i++) *(((double*)c)+i)=0;
	     else
	       for (i = 0; i < idim*jdim; i++){ c[i].real=0;c[i].imag=0;}
	     
	     if (*transa == 'n' || *transa == 'N'){ 
	       adim = idim;
	       i0= *ailo+ilo; i1= *ailo+ihi;   
	       j0= *ajlo+klo; j1= *ajlo+khi;
	       ga_get_(g_a, &i0, &i1, &j0, &j1, a, &idim);
	     }else{
	       adim = kdim;
	       i0= *ajlo+klo; i1= *ajlo+khi;   
	       j0= *ailo+ilo; j1= *ailo+ihi;
	       ga_get_(g_a, &i0, &i1, &j0, &j1, a, &kdim);
	     }


	     /* Avoid rereading B if it is the same patch as last time. */
	     if(get_new_B) { 
	       if (*transb == 'n' || *transb == 'N'){ 
		 bdim = kdim;
		 i0= *bilo+klo; i1= *bilo+khi;   
		 j0= *bjlo+jlo; j1= *bjlo+jhi;
		 ga_get_(g_b, &i0, &i1, &j0, &j1, b, &kdim);
	       }else{
		 bdim = jdim;
		 i0= *bjlo+jlo; i1= *bjlo+jhi;   
		 j0= *bilo+klo; j1= *bilo+khi;
		 ga_get_(g_b, &i0, &i1, &j0, &j1, b, &jdim);
	       }
	       get_new_B = FALSE; /* Until J or K change again */
	     }

	     
	     idim_t=idim; jdim_t=jdim; kdim_t=kdim;
	     adim_t=adim; bdim_t=bdim; cdim_t=cdim;

#	   if (defined(CRAY) || defined(WIN32)) && !defined(GA_C_CORE)
	     switch(atype) {
	     case C_FLOAT:
	       xb_sgemm(transa, transb, &idim_t, &jdim_t, &kdim_t,
			(float *)alpha, (float *)a, &adim_t, (float *)b, 
			&bdim_t, &ZERO_F,  (float *)c, &cdim_t);
	       break;
	     case C_DBL:
	       DGEMM(cptofcd(transa), cptofcd(transb), &idim, &jdim, &kdim,
		     alpha, (double*)a, &adim, (double*)b, &bdim, &ONE, 
		     (double*)c, &cdim);
	       break;
	     case C_DCPL:
	       ZGEMM(cptofcd(transa), cptofcd(transb), &idim, &jdim, &kdim,
		     (DoubleComplex*)alpha, a, &adim, b, &bdim, &ONE,c,&cdim);
	       break;
	     default:
	       ga_error("ga_matmul_patch: wrong data type", atype);
	     }
#          else 
	     switch(atype) {
	     case C_FLOAT:
	       xb_sgemm(transa, transb, &idim_t, &jdim_t, &kdim_t,
			(float *)alpha, (float *)a, &adim_t, (float *)b, 
			&bdim_t, &ZERO_F,  (float *)c, &cdim_t);
	       break;
	     case C_DBL:
#            ifdef GA_C_CORE
	       xb_dgemm(transa, transb, &idim_t, &jdim_t, &kdim_t,
			alpha, (double *)a, &adim_t, (double *)b, &bdim_t, 
			&ZERO_D,  (double *)c, &cdim_t);
#            else
	       dgemm_(transa, transb, &idim, &jdim, &kdim,
		      alpha, a, &adim, b, &bdim, &ONE, c, &cdim, 1, 1);
#            endif
	       break;
	     case C_DCPL:
#            ifdef GA_C_CORE
	       xb_zgemm(transa, transb, &idim_t, &jdim_t, &kdim_t,
			(DoubleComplex *)alpha, a, &adim_t, b, &bdim_t, 
			&ZERO,  c, &cdim_t);
#            else
	       zgemm_(transa, transb, &idim, &jdim, &kdim,
		      (DoubleComplex*)alpha, a, &adim, b, &bdim, &ONE, c, 
		      &cdim, 1, 1);
#            endif
	       break;
	     default:
	       ga_error("ga_matmul_patch: wrong data type", atype);
	     }
#          endif
	     
	     i0= *cilo+ilo; i1= *cilo+ihi;   j0= *cjlo+jlo; j1= *cjlo+jhi;
	     if(atype == C_FLOAT) 
	       ga_acc_(g_c, &i0, &i1, &j0, &j1, (float *)c, 
		       &cdim, &ONE_F);
	     else
	       ga_acc_(g_c, &i0, &i1, &j0, &j1, (DoublePrecision*)c, 
		       &cdim, (DoublePrecision*)&ONE);
	   }
	   ++ijk;
	 }
       }
   }
   
#ifndef STATBUF
   ga_free(a);
#endif

   GA_POP_NAME;
   if(local_sync_end)ga_sync_();

}


void ga_matmul_patch(transa, transb, alpha, beta,
		     g_a, ailo, aihi, ajlo, ajhi,
		     g_b, bilo, bihi, bjlo, bjhi,
		     g_c, cilo, cihi, cjlo, cjhi)

     Integer *g_a, *ailo, *aihi, *ajlo, *ajhi;    /* patch of g_a */
     Integer *g_b, *bilo, *bihi, *bjlo, *bjhi;    /* patch of g_b */
     Integer *g_c, *cilo, *cihi, *cjlo, *cjhi;    /* patch of g_c */
     void    *alpha, *beta;
     char    *transa, *transb;
{
    if(ga_is_mirrored_(g_a)) 
       ga_matmul_mirrored(transa, transb, alpha, beta,
			  g_a, ailo, aihi, ajlo, ajhi,
			  g_b, bilo, bihi, bjlo, bjhi,
			  g_c, cilo, cihi, cjlo, cjhi);
    else {
       _gai_matmul_patch_flag = SET;
       ga_matmul(transa, transb, alpha, beta,
		 g_a, ailo, aihi, ajlo, ajhi,
		 g_b, bilo, bihi, bjlo, bjhi,
		 g_c, cilo, cihi, cjlo, cjhi);
       _gai_matmul_patch_flag = UNSET;
    }
}


/*\ select the 2d plane to be used in matrix multiplication                     
  \*/
static void  gai_setup_2d_patch(Integer rank, Integer dims[],
                                Integer lo[], Integer hi[],
                                Integer* ilo, Integer* ihi,
                                Integer* jlo, Integer* jhi,
                                Integer* dim1, Integer* dim2,
                                int* ipos, int* jpos)
{
    int d,e=0;

    for(d=0; d<rank; d++)
       if( (hi[d]-lo[d])>0 && ++e>2 ) ga_error("3-D Patch Detected", 0L);
    *ipos = *jpos = -1;
    for(d=0; d<rank; d++){
       if( (*ipos <0) && (hi[d]>lo[d]) ) { *ipos =d; continue; }
       if( (*ipos >=0) && (hi[d]>lo[d])) { *jpos =d; break; }
    }

    /*    if(*ipos >*jpos){Integer t=*ipos; *ipos=*jpos; *jpos=t;} 
     */
    
    /* single element case (trivial) */
    if((*ipos <0) && (*jpos <0)){ *ipos =0; *jpos=1; }
    else{
       
       /* handle almost trivial case of only one dimension with >1 elements */
       if(*ipos == rank-1) (*ipos)--; /* i cannot be the last dimension */
       if(*ipos <0) *ipos = *jpos-1; /* select i dimension based on j */
       if(*jpos <0) *jpos = *ipos+1; /* select j dimenison based on i */
       
    }
    
    *ilo = lo[*ipos]; *ihi = hi[*ipos];
    *jlo = lo[*jpos]; *jhi = hi[*jpos];
    *dim1 = dims[*ipos];
    *dim2 = dims[*jpos];
}

#define  SETINT(tmp,val,n) {int _i; for(_i=0;_i<n; _i++)tmp[_i]=val;}

/*\ MATRIX MULTIPLICATION for 2d patches of multi-dimensional arrays 
 *  
 *  C[lo:hi,lo:hi] = alpha*op(A)[lo:hi,lo:hi] * op(B)[lo:hi,lo:hi]        
 *                 + beta *C[lo:hi,lo:hi]
 *
 *  where:
 *          op(A) = A or A' depending on the transpose flag
 *  [lo:hi,lo:hi] - patch indices _after_ op() operator was applied
 *
\*/
void nga_matmul_patch(char *transa, char *transb, void *alpha, void *beta, 
		      Integer *g_a, Integer alo[], Integer ahi[], 
                      Integer *g_b, Integer blo[], Integer bhi[], 
		      Integer *g_c, Integer clo[], Integer chi[])
{
#ifdef STATBUF
   DoubleComplex a[ICHUNK*KCHUNK], b[KCHUNK*JCHUNK], c[ICHUNK*JCHUNK];
#else
   DoubleComplex *a, *b, *c;
#endif
Integer atype, btype, ctype, adim1, adim2, bdim1, bdim2, cdim1, cdim2;
Integer me= ga_nodeid_(), nproc, inode, iproc;
Integer i, ijk = 0, i0, i1, j0, j1;
Integer ilo, ihi, idim, jlo, jhi, jdim, klo, khi, kdim;
Integer n, m, k, adim, bdim, cdim, arank, brank, crank;
int aipos, ajpos, bipos, bjpos,cipos, cjpos, need_scaling=1;
Integer Ichunk, Kchunk, Jchunk;
Integer ailo, aihi, ajlo, ajhi;    /* 2d plane of g_a */
Integer bilo, bihi, bjlo, bjhi;    /* 2d plane of g_b */
Integer cilo, cihi, cjlo, cjhi;    /* 2d plane of g_c */
Integer adims[GA_MAX_DIM],bdims[GA_MAX_DIM],cdims[GA_MAX_DIM],tmpld[GA_MAX_DIM];
Integer *tmplo = adims, *tmphi =bdims; 
DoubleComplex ONE, ZERO;
float ONE_F = 1.0, ZERO_F = 0.0;
double ZERO_D = 0.0;
Integer ZERO_I = 0;
Integer get_new_B;
DoublePrecision chunk_cube;
Integer min_tasks = 10, max_chunk;
int local_sync_begin,local_sync_end;
int idim_t, jdim_t, kdim_t, adim_t, bdim_t, cdim_t;

   ONE.real =1.; ZERO.real =0.;
   ONE.imag =0.; ZERO.imag =0.;
   
   local_sync_begin = _ga_sync_begin; local_sync_end = _ga_sync_end;
   _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
   if(local_sync_begin)ga_sync_();

   GA_PUSH_NAME("nga_matmul_patch");

   /* Check to make sure all global arrays are of the same type */
   if (!(ga_is_mirrored_(g_a) == ga_is_mirrored_(g_b) &&
        ga_is_mirrored_(g_a) == ga_is_mirrored_(g_c))) {
     ga_error_("Processors do not match for all arrays",ga_nnodes_());
   }
   if (ga_is_mirrored_(g_a)) {
     inode = ga_cluster_nodeid_();
     nproc = ga_cluster_nprocs_(&inode);
     iproc = me - ga_cluster_procid_(&inode, &ZERO_I);
   } else {
     nproc = ga_nnodes_();
     iproc = me;
   }

   nga_inquire_internal_(g_a, &atype, &arank, adims);
   nga_inquire_internal_(g_b, &btype, &brank, bdims);
   nga_inquire_internal_(g_c, &ctype, &crank, cdims);

   if(arank<2)  ga_error("rank of A must be at least 2",arank);
   if(brank<2)  ga_error("rank of B must be at least 2",brank);
   if(crank<2)  ga_error("rank of C must be at least 2",crank);

   if(atype != btype || atype != ctype ) ga_error(" types mismatch ", 0L);
   if(atype != C_DCPL && atype != C_DBL && atype != C_FLOAT) 
     ga_error(" type error",atype);
   
   gai_setup_2d_patch(arank, adims, alo, ahi, &ailo, &aihi, &ajlo, &ajhi, 
		                  &adim1, &adim2, &aipos, &ajpos);
   gai_setup_2d_patch(brank, bdims, blo, bhi, &bilo, &bihi, &bjlo, &bjhi, 
		                  &bdim1, &bdim2, &bipos, &bjpos);
   gai_setup_2d_patch(crank, cdims, clo, chi, &cilo, &cihi, &cjlo, &cjhi, 
		                  &cdim1, &cdim2, &cipos, &cjpos);

   /* check if patch indices and dims match */
   if (*transa == 'n' || *transa == 'N'){
      if (ailo <= 0 || aihi > adim1 || ajlo <= 0 || ajhi > adim2)
         ga_error("  g_a indices out of range ", *g_a);
   }else
      if (ailo <= 0 || aihi > adim2 || ajlo <= 0 || ajhi > adim1)
         ga_error("  g_a indices out of range ", *g_a);

   if (*transb == 'n' || *transb == 'N'){
      if (bilo <= 0 || bihi > bdim1 || bjlo <= 0 || bjhi > bdim2)
          ga_error("  g_b indices out of range ", *g_b);
   }else
      if (bilo <= 0 || bihi > bdim2 || bjlo <= 0 || bjhi > bdim1)
          ga_error("  g_b indices out of range ", *g_b);

   if (cilo <= 0 || cihi > cdim1 || cjlo <= 0 || cjhi > cdim2)
       ga_error("  g_c indices out of range ", *g_c);

  /* verify if patch dimensions are consistent */
   m = aihi - ailo +1;
   n = bjhi - bjlo +1;
   k = ajhi - ajlo +1;
   if( (cihi - cilo +1) != m) ga_error(" a & c dims error",m);
   if( (cjhi - cjlo +1) != n) ga_error(" b & c dims error",n);
   if( (bihi - bilo +1) != k) ga_error(" a & b dims error",k);

   
   chunk_cube = (k*(double)(m*n)) / (min_tasks * nproc);
   max_chunk = (Integer)pow(chunk_cube, (DoublePrecision)(1.0/3.0) );
   if (max_chunk < 32) max_chunk = 32;
   
#ifdef STATBUF
   if(atype ==  C_DBL || atype == C_FLOAT){
      Ichunk=D_CHUNK, Kchunk=D_CHUNK, Jchunk=D_CHUNK;
   }else{
      Ichunk=ICHUNK; Kchunk=KCHUNK; Jchunk=JCHUNK;
   }
#else
   {
     Integer elems, factor = sizeof(DoubleComplex)/GAsizeofM(atype);
     Ichunk = Jchunk = Kchunk = CHUNK_SIZE;
     
     if ( max_chunk > Ichunk) {       
       /*if memory if very limited, performance degrades for large matrices
	 as chunk size is very small, which leads to communication overhead)*/
       Integer avail = ga_memory_avail(atype);
       ga_igop(GA_TYPE_GOP, &avail, (Integer)1, "min");
       if(avail<MINMEM && ga_nodeid_()==0) ga_error("Not enough memory",avail);
       elems = (Integer)(avail*0.9);/* Donot use every last drop */
       
       max_chunk=MIN(max_chunk, (Integer)(sqrt( (double)((elems-EXTRA)/3))));
       Ichunk = MIN(m,max_chunk);
       Jchunk = MIN(n,max_chunk);
       Kchunk = MIN(k,max_chunk);
     }
     else /* "EXTRA" elems for safety - just in case */
       elems = 3*Ichunk*Jchunk + EXTRA*factor;

     a = (DoubleComplex*) ga_malloc(elems, atype, "GA mulmat bufs");     
     b = a + (Ichunk*Kchunk)/factor + 1; 
     c = b + (Kchunk*Jchunk)/factor + 1;
   }
#endif

   if(atype==C_DCPL){if((((DoubleComplex*)beta)->real == 0) &&
	       (((DoubleComplex*)beta)->imag ==0)) need_scaling =0;} 
   else if((atype==C_DBL)){if(*(DoublePrecision *)beta == 0)need_scaling =0;}
   else if( *(float*)beta ==0) need_scaling =0;

   if(need_scaling) nga_scale_patch_(g_c, clo, chi, beta);
   else      nga_fill_patch_(g_c, clo, chi, beta);
  
   for(jlo = 0; jlo < n; jlo += Jchunk){ /* loop through columns of g_c patch */
       jhi = MIN(n-1, jlo+Jchunk-1);
       jdim= jhi - jlo +1;
       
       for(klo = 0; klo < k; klo += Kchunk){    /* loop cols of g_a patch */
	 khi = MIN(k-1, klo+Kchunk-1);        /* loop rows of g_b patch */
	 kdim= khi - klo +1;               

	 get_new_B = TRUE;
	 
	 for(ilo = 0; ilo < m; ilo += Ichunk){ /*loop through rows of g_c patch */
	   
	   if(ijk%nproc == iproc){
	     ihi = MIN(m-1, ilo+Ichunk-1);
	     idim= cdim = ihi - ilo +1;
	     
	     if(atype == C_FLOAT) 
	       for (i = 0; i < idim*jdim; i++) *(((float*)c)+i)=0;
	     else if(atype ==  C_DBL)
	       for (i = 0; i < idim*jdim; i++) *(((double*)c)+i)=0;
	     else
	       for (i = 0; i < idim*jdim; i++){ c[i].real=0;c[i].imag=0;}
	     
	     if (*transa == 'n' || *transa == 'N'){ 
	       adim = idim;
	       i0= ailo+ilo; i1= ailo+ihi;   
	       j0= ajlo+klo; j1= ajlo+khi;
	     }else{
	       adim = kdim;
	       i0= ajlo+klo; i1= ajlo+khi;   
	       j0= ailo+ilo; j1= ailo+ihi;
	     }

	     /* ga_get_(g_a, &i0, &i1, &j0, &j1, a, &adim); */
	     memcpy(tmplo,alo,arank*sizeof(Integer));
	     memcpy(tmphi,ahi,arank*sizeof(Integer));
	     SETINT(tmpld,1,arank-1);
	     tmplo[aipos]=i0; tmphi[aipos]=i1;
	     tmplo[ajpos]=j0; tmphi[ajpos]=j1;
	     tmpld[aipos]=i1-i0+1;
	     nga_get_(g_a,tmplo,tmphi,a,tmpld);
	     
	     if(get_new_B) {
	       if (*transb == 'n' || *transb == 'N'){ 
		 bdim = kdim;
		 i0= bilo+klo; i1= bilo+khi;   
		 j0= bjlo+jlo; j1= bjlo+jhi;
	       }else{
		 bdim = jdim;
		 i0= bjlo+jlo; i1= bjlo+jhi;   
		 j0= bilo+klo; j1= bilo+khi;
	       }
	       /* ga_get_(g_b, &i0, &i1, &j0, &j1, b, &bdim); */
	       memcpy(tmplo,blo,brank*sizeof(Integer));
	       memcpy(tmphi,bhi,brank*sizeof(Integer));
	       SETINT(tmpld,1,brank-1);
	       tmplo[bipos]=i0; tmphi[bipos]=i1;
	       tmplo[bjpos]=j0; tmphi[bjpos]=j1;
	       tmpld[bipos]=i1-i0+1;
	       nga_get_(g_b,tmplo,tmphi,b,tmpld);
	       get_new_B = FALSE;
	     }

	     idim_t=idim; jdim_t=jdim; kdim_t=kdim;
	     adim_t=adim; bdim_t=bdim; cdim_t=cdim;

#	     if (defined(CRAY) || defined(WIN32)) && !defined(GA_C_CORE)
		  switch(atype) {
		  case C_FLOAT:
		    xb_sgemm(transa, transb, &idim_t, &jdim_t, &kdim_t,
			     (float *)alpha, (float *)a, &adim_t, (float *)b, 
			     &bdim_t, &ZERO_F,  (float *)c, &cdim_t);
		    break;		    
		  case C_DBL:
                    DGEMM(cptofcd(transa), cptofcd(transb), &idim, &jdim, &kdim,
                          alpha, (double*)a, &adim, (double*)b, &bdim, &ONE, 
			  (double*)c, &cdim);
		    break;
		  case C_DCPL:
                    ZGEMM(cptofcd(transa), cptofcd(transb), &idim, &jdim, &kdim,
                          (DoubleComplex*)alpha, a, &adim, b, &bdim, &ONE,c,&cdim);
		    break;
		  default:
		    ga_error("ga_matmul_patch: wrong data type", atype);
		  }
#            else 
		  switch(atype) {
		  case C_FLOAT:
		    xb_sgemm(transa, transb, &idim_t, &jdim_t, &kdim_t,
			     (float *)alpha, (float *)a, &adim_t, (float *)b, &bdim_t, 
			     &ZERO_F,  (float *)c, &cdim_t);
		    break;
		  case C_DBL:
#                 ifdef GA_C_CORE
		    
		    xb_dgemm(transa, transb, &idim_t, &jdim_t, &kdim_t,
			     alpha, (double *)a, &adim_t, (double *)b, &bdim_t, 
			     &ZERO_D,  (double *)c, &cdim_t);
#                 else
#                 if !defined(HAS_BLAS) && defined(EXT_INT)
		    dgemm_(transa, transb, &idim, &jdim, &kdim,
			   alpha, a, &adim, b, &bdim, &ONE, c, &cdim, 1, 1);
#                   else
		    dgemm_(transa, transb, &idim_t, &jdim_t, &kdim_t,
			   alpha, a, &adim_t, b, &bdim_t, &ONE,c,&cdim_t,1,1);
#                   endif
#                 endif
		    break;
		  case C_DCPL:
#                 ifdef GA_C_CORE
		    xb_zgemm(transa, transb, &idim_t, &jdim_t, &kdim_t,
			     (DoubleComplex *)alpha, a, &adim_t, b, &bdim_t, 
			     &ZERO,  c, &cdim_t);
#                 else
		    zgemm_(transa, transb, &idim, &jdim, &kdim,
			   (DoubleComplex*)alpha, a, &adim, b, &bdim, &ONE, c, 
			   &cdim, 1, 1);
#                 endif
		    break;
		  default:
		    ga_error("ga_matmul_patch: wrong data type", atype);
		  }
#            endif

                  i0= cilo+ilo; i1= cilo+ihi;   j0= cjlo+jlo; j1= cjlo+jhi;
                  /* ga_acc_(g_c, &i0, &i1, &j0, &j1, (DoublePrecision*)c, 
                                            &cdim, (DoublePrecision*)&ONE); */
		  memcpy(tmplo,clo,crank*sizeof(Integer));
		  memcpy(tmphi,chi,crank*sizeof(Integer));
		  SETINT(tmpld,1,crank-1);
		  tmplo[cipos]=i0; tmphi[cipos]=i1;
		  tmplo[cjpos]=j0; tmphi[cjpos]=j1;
		  tmpld[cipos]=i1-i0+1;
		  if(atype == C_FLOAT) 
		    nga_acc_(g_c,tmplo,tmphi,(float *)c,tmpld, &ONE_F);
		  else
		    nga_acc_(g_c,tmplo,tmphi,c,tmpld,(DoublePrecision*)&ONE);
               }
	   ++ijk;
	 }
       }
   }

#ifndef STATBUF
   ga_free(a);
#endif
   
   GA_POP_NAME;
   if(local_sync_end)ga_sync_(); 
}

/*\ MATRIX MULTIPLICATION for patches 
 *  Fortran interface
\*/
void FATR nga_matmul_patch_(transa, transb, alpha, beta, g_a, alo, ahi, 

                       g_b, blo, bhi, g_c, clo, chi)

                      void *alpha, *beta;
		      Integer *g_a, alo[], ahi[]; 
                      Integer *g_b, blo[], bhi[]; 
		      Integer *g_c, clo[], chi[];

#if defined(CRAY) || defined(WIN32)
     _fcd   transa, transb;
{    
     nga_matmul_patch(_fcdtocp(transa), _fcdtocp(transb), alpha, beta, g_a, alo,
                      ahi, g_b, blo, bhi, g_c, clo, chi);
#else
     char    *transa, *transb;
{    
	nga_matmul_patch(transa, transb, alpha, beta, g_a, alo, ahi,
                         g_b, blo, bhi, g_c, clo, chi);
#endif
}

void FATR ga_matmul_patch_(transa, transb, alpha, beta,
                      g_a, ailo, aihi, ajlo, ajhi,
                      g_b, bilo, bihi, bjlo, bjhi,
                      g_c, cilo, cihi, cjlo, cjhi)

     Integer *g_a, *ailo, *aihi, *ajlo, *ajhi;    /* patch of g_a */
     Integer *g_b, *bilo, *bihi, *bjlo, *bjhi;    /* patch of g_b */
     Integer *g_c, *cilo, *cihi, *cjlo, *cjhi;    /* patch of g_c */
     DoublePrecision      *alpha, *beta;

#if defined(CRAY) || defined(WIN32)
     _fcd   transa, transb;
     {   
	_gai_matmul_patch_flag = SET;
	ga_matmul(_fcdtocp(transa), _fcdtocp(transb), alpha, beta,
		  g_a, ailo, aihi, ajlo, ajhi,
		  g_b, bilo, bihi, bjlo, bjhi,
		  g_c, cilo, cihi, cjlo, cjhi);
	_gai_matmul_patch_flag = UNSET;
     }
#else
     char    *transa, *transb;
{    
#if 0
Integer alo[2], ahi[2]; 
Integer blo[2], bhi[2];
Integer clo[2], chi[2];
        alo[0]=*ailo; ahi[0]=*aihi; alo[1]=*ajlo; ahi[1]=*ajhi;
        blo[0]=*bilo; bhi[0]=*bihi; blo[1]=*bjlo; bhi[1]=*bjhi;
        clo[0]=*cilo; chi[0]=*cihi; clo[1]=*cjlo; chi[1]=*cjhi;
	nga_matmul_patch(transa, transb, alpha, beta, g_a, alo, ahi,
                         g_b, blo, bhi, g_c, clo, chi);
#else
	if(ga_is_mirrored_(g_a)) 
	   ga_matmul_mirrored(transa, transb, alpha, beta,
			      g_a, ailo, aihi, ajlo, ajhi,
			      g_b, bilo, bihi, bjlo, bjhi,
			      g_c, cilo, cihi, cjlo, cjhi);
	else {
	   _gai_matmul_patch_flag = SET;
	   ga_matmul(transa, transb, alpha, beta,
		     g_a, ailo, aihi, ajlo, ajhi,
		     g_b, bilo, bihi, bjlo, bjhi,
		     g_c, cilo, cihi, cjlo, cjhi);
	   _gai_matmul_patch_flag = UNSET;
	}
#endif
}
#endif




/*********************** Fortran warppers for ga_Xgemm ***********************/


#ifdef USE_SUMMA
void ga_dgemm_(char *transa, char *transb, Integer *m, Integer *n, Integer *k,
               double *alpha, Integer *g_a, Integer *g_b,
               double *beta, Integer *g_c) {
  /**
   * ga_summa calls ga_ga_dgemm to handle cases it does not cover
   */
  _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
  ga_summa_(transa, transb, m, n, k, alpha, g_a, g_b, beta, g_c);
}
#  define GA_DGEMM ga_ga_dgemm_
#else
#  define GA_DGEMM ga_dgemm_
#endif


#define  SET_GEMM_INDICES\
  Integer ailo = 1;\
  Integer aihi = *m;\
  Integer ajlo = 1;\
  Integer ajhi = *k;\
\
  Integer bilo = 1;\
  Integer bihi = *k;\
  Integer bjlo = 1;\
  Integer bjhi = *n;\
\
  Integer cilo = 1;\
  Integer cihi = *m;\
  Integer cjlo = 1;\
  Integer cjhi = *n

#if defined(CRAY) || defined(WIN32)
void FATR GA_DGEMM(_fcd Transa, _fcd Transb, Integer *m, Integer *n, Integer *k,
             void *alpha, Integer *g_a, Integer *g_b,
             void *beta, Integer *g_c)
{
char *transa, *transb;
SET_GEMM_INDICES;
      transa = _fcdtocp(Transa);
      transb = _fcdtocp(Transb);
#else
void FATR GA_DGEMM(char *transa, char *transb, Integer *m, Integer *n, Integer *k,
             void *alpha, Integer *g_a, Integer *g_b,
             void *beta, Integer *g_c, int talen, int tblen)
{
SET_GEMM_INDICES;
#endif
 
 ga_matmul(transa, transb, alpha, beta,
	   g_a, &ailo, &aihi, &ajlo, &ajhi,
	   g_b, &bilo, &bihi, &bjlo, &bjhi,
	   g_c, &cilo, &cihi, &cjlo, &cjhi);
}

#if defined(CRAY) || defined(WIN32)
void FATR ga_sgemm_(_fcd Transa, _fcd Transb, Integer *m, Integer *n, Integer *k,
             void *alpha, Integer *g_a, Integer *g_b,
             void *beta, Integer *g_c)
{
char *transa, *transb;
SET_GEMM_INDICES;
      transa = _fcdtocp(Transa);
      transb = _fcdtocp(Transb);
#else
void FATR ga_sgemm_(char *transa, char *transb, Integer *m, Integer *n, Integer *k,
             void *alpha, Integer *g_a, Integer *g_b,
             void *beta, Integer *g_c, int talen, int tblen)
{
SET_GEMM_INDICES;
#endif


  ga_matmul (transa, transb, alpha, beta,
	     g_a, &ailo, &aihi, &ajlo, &ajhi,
	     g_b, &bilo, &bihi, &bjlo, &bjhi,
	     g_c, &cilo, &cihi, &cjlo, &cjhi);
}


#if defined(CRAY) || defined(WIN32)
void FATR ga_zgemm_(_fcd Transa, _fcd Transb, Integer *m, Integer *n, Integer *k,
             void *alpha, Integer *g_a, Integer *g_b,
             void *beta, Integer *g_c)
{
char *transa, *transb;
SET_GEMM_INDICES;
      transa = _fcdtocp(Transa);
      transb = _fcdtocp(Transb);
#else
void FATR ga_zgemm_(char *transa, char *transb, Integer *m, Integer *n, Integer *k,
             void *alpha, Integer *g_a, Integer *g_b,
             void *beta, Integer *g_c, int talen, int tblen)
{
SET_GEMM_INDICES;
#endif


  ga_matmul (transa, transb, alpha, beta,
	     g_a, &ailo, &aihi, &ajlo, &ajhi,
	     g_b, &bilo, &bihi, &bjlo, &bjhi,
	     g_c, &cilo, &cihi, &cjlo, &cjhi);
}

