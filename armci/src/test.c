/* $Id: test.c,v 1.22 2000-06-16 22:25:41 d3h325 Exp $ */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#ifdef WIN32
#  include <windows.h>
#  define sleep(x) Sleep(1000*(x))
#endif

/* ARMCI is impartial to message-passing libs - we handle them with MP macros */
#if defined(PVM)
#   include <pvm3.h>
#   ifdef CRAY
#     define MPGROUP         (char *)NULL
#     define MP_INIT(arc,argv)
#   else
#     define MPGROUP           "mp_working_group"
#     define MP_INIT(arc,argv) pvm_init(arc, argv)
#   endif
#   define MP_FINALIZE()     pvm_exit()
#   define MP_BARRIER()      pvm_barrier(MPGROUP,-1)
#   define MP_MYID(pid)      *(pid)   = pvm_getinst(MPGROUP,pvm_mytid())
#   define MP_PROCS(pproc)   *(pproc) = (int)pvm_gsize(MPGROUP)
    void pvm_init(int argc, char *argv[]);
#elif defined(TCGMSG)
#   include <sndrcv.h>
    long tcg_tag =30000;
#   define MP_BARRIER()      SYNCH_(&tcg_tag)
#   define MP_INIT(arc,argv) PBEGIN_((argc),(argv))
#   define MP_FINALIZE()     PEND_()
#   define MP_MYID(pid)      *(pid)   = (int)NODEID_()
#   define MP_PROCS(pproc)   *(pproc) = (int)NNODES_()
#else
#   include <mpi.h>
#   define MP_BARRIER()      MPI_Barrier(MPI_COMM_WORLD)
#   define MP_FINALIZE()     MPI_Finalize()
#   define MP_INIT(arc,argv) MPI_Init(&(argc),&(argv))
#   define MP_MYID(pid)      MPI_Comm_rank(MPI_COMM_WORLD, (pid))
#   define MP_PROCS(pproc)   MPI_Comm_size(MPI_COMM_WORLD, (pproc));
#endif

#include "armci.h"

#define DIM1 5
#define DIM2 3
#ifdef __sun
/* Solaris has shared memory shortages in the default system configuration */
# define DIM3 6
# define DIM4 5
# define DIM5 4
#else
# define DIM3 8
# define DIM4 9
# define DIM5 7
#endif
#define DIM6 3
#define DIM7 2


#define OFF 1
#define EDIM1 (DIM1+OFF)
#define EDIM2 (DIM2+OFF)
#define EDIM3 (DIM3+OFF)
#define EDIM4 (DIM4+OFF)
#define EDIM5 (DIM5+OFF)
#define EDIM6 (DIM6+OFF)
#define EDIM7 (DIM7+OFF)

#define DIMS 4
#define MAXDIMS 7
#define MAX_DIM_VAL 50 
#define LOOP 200

#define BASE 100.
#define MAXPROC 128
#define TIMES 100

#ifdef CRAY
# define ELEMS 800
#else
# define ELEMS 200
#endif


/***************************** macros ************************/
#define COPY(src, dst, bytes) memcpy((dst),(src),(bytes))
#define MAX(a,b) (((a) >= (b)) ? (a) : (b))
#define MIN(a,b) (((a) <= (b)) ? (a) : (b))
#define ABS(a) (((a) <0) ? -(a) : (a))

/***************************** global data *******************/
int me, nproc;
void* work[MAXPROC]; /* work array for propagating addresses */



#ifdef PVM
void pvm_init(int argc, char *argv[])
{
    int mytid, mygid, ctid[MAXPROC];
    int np, i;

    mytid = pvm_mytid();
    if((argc != 2) && (argc != 1)) goto usage;
    if(argc == 1) np = 1;
    if(argc == 2)
        if((np = atoi(argv[1])) < 1) goto usage;
    if(np > MAXPROC) goto usage;

    mygid = pvm_joingroup(MPGROUP);

    if(np > 1)
        if (mygid == 0) 
            i = pvm_spawn(argv[0], argv+1, 0, "", np-1, ctid);

    while(pvm_gsize(MPGROUP) < np) sleep(1);

    /* sync */
    pvm_barrier(MPGROUP, np);
    
    printf("PVM initialization done!\n");
    
    return;

usage:
    fprintf(stderr, "usage: %s <nproc>\n", argv[0]);
    pvm_exit();
    exit(-1);
}
#endif

/*\ generate random range for a section of multidimensional array 
\*/
void get_range(int ndim, int dims[], int lo[], int hi[]) 
{
	int dim;
	for(dim=0; dim <ndim;dim++){
		int toss1, toss2;
		toss1 = rand()%dims[dim];
		toss2 = rand()%dims[dim];
		if(toss1<toss2){
			lo[dim]=toss1;
			hi[dim]=toss2;
		}else {
  			hi[dim]=toss1;
			lo[dim]=toss2;
		}
    }
}



/*\ generates a new random range similar to the input range for an array with specified dimensions
\*/
void new_range(int ndim, int dims[], int lo[], int hi[],int new_lo[], int new_hi[])
{
	int dim;
	for(dim=0; dim <ndim;dim++){
		int toss, range;
		int diff = hi[dim] -lo[dim]+1;
		assert(diff <= dims[dim]);
                range = dims[dim]-diff;
                toss = (range > 0)? rand()%range : lo[dim];
		new_lo[dim] = toss;
		new_hi[dim] = toss + diff -1;
		assert(new_hi[dim] < dims[dim]);
		assert(diff == (new_hi[dim] -new_lo[dim]+1));
	}
}





/*\ print range of ndim dimensional array with two strings before and after
\*/
void print_range(char *pre,int ndim, int lo[], int hi[], char* post)
{
	int i;

	printf("%s[",pre);
	for(i=0;i<ndim;i++){
		printf("%d:%d",lo[i],hi[i]);
		if(i==ndim-1)printf("] %s",post);
		else printf(",");
	}
}

/*\ print subscript of ndim dimensional array with two strings before and after
\*/
void print_subscript(char *pre,int ndim, int subscript[], char* post)
{
	int i;

	printf("%s [",pre);
	for(i=0;i<ndim;i++){
		printf("%d",subscript[i]);
		if(i==ndim-1)printf("] %s",post);
		else printf(",");
	}
}


/*\ print a section of a 2-D array of doubles
\*/
void print_2D_double(double *a, int ld, int *lo, int *hi)
{
int i,j;
     for(i=lo[0];i<=hi[0];i++){
       for(j=lo[1];j<=hi[1];j++) printf("%13f ",a[ld*j+i]);
       printf("\n");
     }
}
          

/*\ initialize array: a[i,j,k,..]=i+100*j+10000*k+ ... 
\*/
void init(double *a, int ndim, int elems, int dims[])
{
  int idx[MAXDIMS];
  int i,dim;

 	for(i=0; i<elems; i++){
		int Index = i;
		double field, val;
        
		for(dim = 0; dim < ndim; dim++){
			idx[dim] = Index%dims[dim];
			Index /= dims[dim];
		}
		
        field=1.; val=0.;
		for(dim=0; dim< ndim;dim++){
			val += field*idx[dim];
			field *= BASE;
		}
		a[i] = val;
		/* printf("(%d,%d,%d)=%6.0f",idx[0],idx[1],idx[2],val); */
	}
}


/*\ compute Index from subscript
 *  assume that first subscript component changes first
\*/
int Index(int ndim, int subscript[], int dims[])
{
	int idx = 0, i, factor=1;
	for(i=0;i<ndim;i++){
		idx += subscript[i]*factor;
		factor *= dims[i];
	}
	return idx;
}


void update_subscript(int ndim, int subscript[], int lo[], int hi[], int dims[])
{
	int i;
	for(i=0;i<ndim;i++){
		if(subscript[i] < hi[i]) { subscript[i]++; return; }
		subscript[i] = lo[i];
	}
}

	

void compare_patches(double eps, int ndim, double *patch1, int lo1[], int hi1[], int dims1[],
					           double *patch2, int lo2[], int hi2[], int dims2[])
{
	int i,j, elems=1;	
	int subscr1[MAXDIMS], subscr2[MAXDIMS];
        double diff,max;

	for(i=0;i<ndim;i++){   /* count # of elements & verify consistency of both patches */
		int diff = hi1[i]-lo1[i];
		assert(diff == (hi2[i]-lo2[i]));
		assert(diff < dims1[i]);
		assert(diff < dims2[i]);
		elems *= diff+1;
		subscr1[i]= lo1[i];
		subscr2[i]=lo2[i];
	}

	
	/* compare element values in both patches */ 
	for(j=0; j< elems; j++){ 
		int idx1, idx2, offset1, offset2;
		
		idx1 = Index(ndim, subscr1, dims1);	 /* calculate element Index from a subscript */
		idx2 = Index(ndim, subscr2, dims2);

		if(j==0){
			offset1 =idx1;
			offset2 =idx2;
		}
		idx1 -= offset1;
		idx2 -= offset2;
		

                diff = patch1[idx1] - patch2[idx2];
                max  = MAX(ABS(patch1[idx1]),ABS(patch2[idx2]));
                if(max == 0. || max <eps) max = 1.; 

		if(eps < ABS(diff)/max){
			char msg[48];
			sprintf(msg,"(proc=%d):%lf",me,patch1[idx1]);
			print_subscript("ERROR: a",ndim,subscr1,msg);
			sprintf(msg,"%lf\n",patch2[idx2]);
			print_subscript(" b",ndim,subscr2,msg);
                        fflush(stdout);
                        sleep(1);
                        ARMCI_Error("Bailing out",0);
		}

		{ /* update subscript for the patches */
		   update_subscript(ndim, subscr1, lo1,hi1, dims1);
		   update_subscript(ndim, subscr2, lo2,hi2, dims2);
		}
	}
				 
	

	/* make sure we reached upper limit */
	/*for(i=0;i<ndim;i++){ 
		assert(subscr1[i]==hi1[i]);
		assert(subscr2[i]==hi2[i]);
	}*/ 
}


void scale_patch(double alpha, int ndim, double *patch1, int lo1[], int hi1[], int dims1[])
{
	int i,j, elems=1;	
	int subscr1[MAXDIMS];

	for(i=0;i<ndim;i++){   /* count # of elements in patch */
		int diff = hi1[i]-lo1[i];
		assert(diff < dims1[i]);
		elems *= diff+1;
		subscr1[i]= lo1[i];
	}

	/* scale element values in both patches */ 
	for(j=0; j< elems; j++){ 
		int idx1, offset1;
		
		idx1 = Index(ndim, subscr1, dims1);	 /* calculate element Index from a subscript */

		if(j==0){
			offset1 =idx1;
		}
		idx1 -= offset1;

		patch1[idx1] *= alpha;
		update_subscript(ndim, subscr1, lo1,hi1, dims1);
	}	
}


void create_array(void *a[], int elem_size, int ndim, int dims[])
{
     int bytes=elem_size, i, rc;

     assert(ndim<=MAXDIMS);
     for(i=0;i<ndim;i++)bytes*=dims[i];

     rc = ARMCI_Malloc(a, bytes);
     assert(rc==0);
     
     assert(a[me]);
     
}

void destroy_array(void *ptr[])
{
    MP_BARRIER();

    assert(!ARMCI_Free(ptr[me]));
}


int loA[MAXDIMS], hiA[MAXDIMS];
int dimsA[MAXDIMS]={DIM1,DIM2,DIM3,DIM4,DIM5,DIM6, DIM7};
int loB[MAXDIMS], hiB[MAXDIMS];
int dimsB[MAXDIMS]={EDIM1,EDIM2,EDIM3,EDIM4,EDIM5,EDIM6,EDIM7};
int count[MAXDIMS];
int strideA[MAXDIMS], strideB[MAXDIMS];
int loC[MAXDIMS], hiC[MAXDIMS];
int idx[MAXDIMS]={0,0,0,0,0,0,0};

    
void test_dim(int ndim)
{
	int dim,elems;
	int i,j, proc;
	/* double a[DIM4][DIM3][DIM2][DIM1], b[EDIM4][EDIM3][EDIM2][EDIM1];*/
        void *b[MAXPROC];
        void *a, *c;

	elems = 1;   
        strideA[0]=sizeof(double); 
        strideB[0]=sizeof(double);
	for(i=0;i<ndim;i++){
		strideA[i] *= dimsA[i];
		strideB[i] *= dimsB[i];
                if(i<ndim-1){
                     strideA[i+1] = strideA[i];
                     strideB[i+1] = strideB[i];
                }
		elems *= dimsA[i];
	}

        /* create shared and local arrays */
        create_array(b, sizeof(double),ndim,dimsB);
        a = malloc(sizeof(double)*elems);
        assert(a);
        c = malloc(sizeof(double)*elems);
        assert(c);

	init(a, ndim, elems, dimsA);
	
	if(me==0){
            printf("--------array[%d",dimsA[0]);
	    for(dim=1;dim<ndim;dim++)printf(",%d",dimsA[dim]);
	    printf("]--------\n");
        }
        sleep(1);

        ARMCI_AllFence();
        MP_BARRIER();
	for(i=0;i<LOOP;i++){
	    int idx1, idx2, idx3;
	    get_range(ndim, dimsA, loA, hiA);
	    new_range(ndim, dimsB, loA, hiA, loB, hiB);
	    new_range(ndim, dimsA, loA, hiA, loC, hiC);

            proc=nproc-1-me;

            if(me==0){
	       print_range("local",ndim,loA, hiA,"-> ");
	       print_range("remote",ndim,loB, hiB,"-> ");
	       print_range("local",ndim,loC, hiC,"\n");
            }

	    idx1 = Index(ndim, loA, dimsA);
	    idx2 = Index(ndim, loB, dimsB);
	    idx3 = Index(ndim, loC, dimsA);

	    for(j=0;j<ndim;j++)count[j]=hiA[j]-loA[j]+1;

	    count[0]   *= sizeof(double); /* convert range to bytes at stride level zero */

            (void)ARMCI_PutS((double*)a + idx1, strideA, (double*)b[proc] + idx2, strideB, count, ndim-1, proc);

/*            sleep(1);*/

/*            printf("%d: a=(%x,%f) b=(%x,%f)\n",me,idx1 + (double*)a,*(idx1 + (double*)a),idx2 + (double*)b,*(idx2 + (double*)b));*/
/*            fflush(stdout);*/
/*            sleep(1);*/

            /* note that we do not need ARMCI_Fence here since
             * consectutive operations targeting the same process are ordered */
	    (void)ARMCI_GetS((double*)b[proc] + idx2, strideB, (double*)c + idx3, strideA,  count, ndim-1, proc);
            
            compare_patches(0., ndim, (double*)a+idx1, loA, hiA, dimsA, (double*)c+idx3, loC, hiC, dimsA);

    
        }

        free(c);
        destroy_array(b);
        free(a);
}



void GetPermutedProcList(int* ProcList)
{
    int i, iswap, temp;

    if(nproc > MAXPROC) ARMCI_Error("permute_proc: nproc to big ", nproc);

    /* initialize list */
    for(i=0; i< nproc; i++) ProcList[i]=i;
    if(nproc ==1) return;

    /* every process generates different random sequence */
    (void)srand((unsigned)me);

    /* list permutation generated by random swapping */
    for(i=0; i< nproc; i++){
      iswap = (int)(rand() % nproc);
      temp = ProcList[iswap];
      ProcList[iswap] = ProcList[i];
      ProcList[i] = temp;
    }
}



/*\ Atomic Accumulate test:  remote += alpha*local
 *  Every process/or has its patch of array b updated TIMES*NPROC times.
 *  The sequence of updates is random: everybody uses a randomly permuted list
 *  and accumulate is non-collective (of-course)
\*/
void test_acc(int ndim)
{
	int dim,elems;
	int i, proc;
        void *b[MAXPROC];
        void *a, *c;
        double alpha=0.1, scale;
	int idx1, idx2;
        int *proclist = (int*)work;

        elems = 1;   
        strideA[0]=sizeof(double); 
        strideB[0]=sizeof(double);
	for(i=0;i<ndim;i++){
		strideA[i] *= dimsA[i];
		strideB[i] *= dimsB[i];
                if(i<ndim-1){
                     strideA[i+1] = strideA[i];
                     strideB[i+1] = strideB[i];
                }
		elems *= dimsA[i];

                /* set up patch coordinates: same on every processor */
                loA[i]=0;
                hiA[i]=loA[i]+1;
                loB[i]=dimsB[i]-2;
                hiB[i]=loB[i]+1;
                count[i]=hiA[i]-loA[i]+1;
	}

        /* create shared and local arrays */
        create_array(b, sizeof(double),ndim,dimsB);
        a = malloc(sizeof(double)*elems);
        assert(a);
        c = malloc(sizeof(double)*elems);
        assert(c);

	init(a, ndim, elems, dimsA);
	
	if(me==0){
            printf("--------array[%d",dimsA[0]);
	    for(dim=1;dim<ndim;dim++)printf(",%d",dimsA[dim]);
	    printf("]--------\n");
        }

        GetPermutedProcList(proclist);

 	idx1 = Index(ndim, loA, dimsA);
	idx2 = Index(ndim, loB, dimsB);
	count[0]   *= sizeof(double); /* convert range to bytes at stride level zero */
        
        /* initialize all elements of array b to zero */
	elems = 1;
        for(i=0;i<ndim;i++)elems *= dimsB[i];
        for(i=0;i<elems;i++)((double*)b[me])[i]=0.;

        sleep(1);

        if(me==0){
               print_range("patch",ndim,loA, hiA," -> ");
               print_range("patch",ndim,loB, hiB,"\n");
               fflush(stdout);
        }

        ARMCI_AllFence();
        MP_BARRIER();
        for(i=0;i<TIMES*nproc;i++){ 

            proc=proclist[i%nproc];
            (void)ARMCI_AccS(ARMCI_ACC_DBL,&alpha,(double*)a + idx1, strideA, (double*)b[proc] + idx2, strideB, count, ndim-1, proc);
        }

/*	sleep(9);*/
        ARMCI_AllFence();
        MP_BARRIER();

        /* copy my patch into local array c */
	(void)ARMCI_GetS((double*)b[me] + idx2, strideB, (double*)c + idx1, strideA,  count, ndim-1, me);

        scale = alpha*TIMES*nproc; 

        scale_patch(scale, ndim, (double*)a+idx1, loA, hiA, dimsA);
        
        compare_patches(.0001, ndim, (double*)a+idx1, loA, hiA, dimsA, (double*)c+idx1, loA, hiA, dimsA);
        MP_BARRIER();

        if(0==me){
            printf(" OK\n\n");
            fflush(stdout);
        }

        free(c);
        destroy_array(b);
        free(a);
}


/*************************** vector interface *********************************\
 * tests vector interface for transfers of triangular sections of a 2-D array *
 ******************************************************************************/
void test_vector()
{
	int dim,elems,ndim,cols,rows,mrc;
	int i, proc, loop;
        int rc;
        int idx1, idx3;
        void *b[MAXPROC];
        void *a, *c;
        armci_giov_t dsc[MAX_DIM_VAL];
        void *psrc[MAX_DIM_VAL];
        void *pdst[MAX_DIM_VAL];

	elems = 1;   
        ndim  = 2;
	for(i=0;i<ndim;i++){
                dimsA[i]=MAX_DIM_VAL;
                dimsB[i]=MAX_DIM_VAL+1;
		elems *= dimsA[i];
	}

        /* create shared and local arrays */
        create_array(b, sizeof(double),ndim,dimsB);
        a = malloc(sizeof(double)*elems);
        assert(a);
        c = malloc(sizeof(double)*elems);
        assert(c);

	init(a, ndim, elems, dimsA);
	
	if(me==0){
            printf("--------array[%d",dimsA[0]);
	    for(dim=1;dim<ndim;dim++)printf(",%d",dimsA[dim]);
	    printf("]--------\n");
        }
        sleep(1);

	for(loop=0;loop<LOOP;loop++){
	    get_range(ndim, dimsA, loA, hiA);
	    new_range(ndim, dimsB, loA, hiA, loB, hiB);
	    new_range(ndim, dimsA, loA, hiA, loC, hiC);

            proc=nproc-1-me;

            if(me==0){
	       print_range("local",ndim,loA, hiA,"-> ");
	       print_range("remote",ndim,loB, hiB,"-> ");
	       print_range("local",ndim,loC, hiC,"\n");
            }

/*            printf("array at source\n");*/
/*            print_2D_double((double *)a, dimsA[0], loA, hiA);*/

            cols =  hiA[1]-loA[1]+1; 
            rows =  hiA[0]-loA[0]+1; 
            mrc =MIN(cols,rows);

            /* generate a data descriptor for a lower-triangular patch */
            for(i=0; i < mrc; i++){
               int ij[2];
               int idx;

               ij[0] = loA[0]+i;
               ij[1] = loA[1]+i;
               idx = Index(ndim, ij, dimsA);
               psrc[i]= (double*)a + idx;

               ij[0] = loB[0]+i;
               ij[1] = loB[1]+i;
               idx = Index(ndim, ij, dimsB);
               pdst[i]= (double*)b[proc] + idx;

               dsc[i].bytes = (rows-i)*sizeof(double);
               dsc[i].src_ptr_array = &psrc[i];
               dsc[i].dst_ptr_array = &pdst[i];

               /* assume each element different in size (not true in rectangular patches) */ 
               dsc[i].ptr_array_len = 1; 
            }

            if(rc=ARMCI_PutV(dsc, mrc, proc))ARMCI_Error("putv failed ",rc);

/*            printf("array at destination\n");*/
/*            print_2D_double((double *)b[proc], dimsB[0], loB, hiB);*/

            /* generate a data descriptor for the upper-triangular patch */
            /* there is one less element since diagonal is excluded      */
            for(i=1; i < cols; i++){
               int ij[2];

               ij[0] = loA[0];
               ij[1] = loA[1]+i;
               psrc[i-1]= (double*)a + Index(ndim, ij, dimsA);

               ij[0] = loB[0];
               ij[1] = loB[1]+i;
               pdst[i-1]= (double*)b[proc] + Index(ndim, ij, dimsB);

               mrc = MIN(i,rows);
               dsc[i-1].bytes = mrc*sizeof(double);
               dsc[i-1].src_ptr_array = &psrc[i-1];
               dsc[i-1].dst_ptr_array = &pdst[i-1];

               /* assume each element different in size (not true in rectangular patches) */ 
               dsc[i-1].ptr_array_len = 1; 
            }

            if(cols-1)if(rc=ARMCI_PutV(dsc, cols-1, proc))ARMCI_Error("putv(2) failed ",rc);

            /* we get back entire rectangular patch */
            for(i=0; i < cols; i++){
               int ij[2];
               ij[0] = loB[0];
               ij[1] = loB[1]+i;
               psrc[i]= (double*)b[proc] + Index(ndim, ij, dimsB);

               ij[0] = loC[0];
               ij[1] = loC[1]+i;
               pdst[i]= (double*)c + Index(ndim, ij, dimsA);
            }

            dsc[0].bytes = rows*sizeof(double);
            dsc[0].src_ptr_array = psrc;
            dsc[0].dst_ptr_array = pdst;
            dsc[0].ptr_array_len = cols; 

            /* note that we do not need ARMCI_Fence here since
             * consecutive operations targeting the same process are ordered */
            if(rc=ARMCI_GetV(dsc, 1, proc))ARMCI_Error("getv failed ",rc);
            
	    idx1 = Index(ndim, loA, dimsA);
	    idx3 = Index(ndim, loC, dimsA);
            compare_patches(0., ndim, (double*)a+idx1, loA, hiA, dimsA, (double*)c+idx3, loC, hiC, dimsA);
    
        }

        free(c);
        destroy_array(b);
        free(a);
}


/*\ Atomic Accumulate test for vector API:  remote += alpha*local
 *  Every process/or has its patch of array b updated TIMES*NPROC times.
 *  The sequence of updates is random: everybody uses a randomly permuted list
 *  and accumulate is non-collective (of-course)
\*/
void test_vector_acc()
{
	int dim,elems,bytes;
	int i, j, proc, rc, one=1;
        void *b[MAXPROC];
        void *psrc[ELEMS/2], *pdst[ELEMS/2];
        void *a, *c;
        double alpha=0.1, scale;
        int *proclist = (int*)work;
        armci_giov_t dsc;

        elems = ELEMS;
        dim =1;
        bytes = sizeof(double)*elems;

        /* create shared and local arrays */
        create_array(b, sizeof(double),dim,&elems);
        a = malloc(bytes);
        assert(a);
        c = malloc(bytes);
        assert(c);

	init(a, dim, elems, &elems);
	
	if(me==0){
            printf("--------array[%d",elems);
	    printf("]--------\n");
            fflush(stdout);
        }

        GetPermutedProcList(proclist);
        
        /* initialize all elements of array b to zero */
        for(i=0;i<elems;i++)((double*)b[me])[i]=0.;

        sleep(1);

        dsc.bytes = sizeof(double);
        dsc.src_ptr_array = psrc;
        dsc.dst_ptr_array = pdst;
        dsc.ptr_array_len = elems/2; 


        MP_BARRIER();
        for(i=0;i<TIMES*nproc;i++){ 

/*            proc=proclist[i%nproc];*/
            proc=0;

            /* accumulate even numbered elements */
            for(j=0; j<elems/2; j++){
                psrc[j]= 2*j + (double*)a;
                pdst[j]= 2*j + (double*)b[proc];
            }
            if(rc = ARMCI_AccV(ARMCI_ACC_DBL, &alpha, &dsc, 1, proc))
                ARMCI_Error("accumlate failed",rc);
/*            for(j=0; j<elems; j++)
                printf("%d %lf %lf\n",j, *(j+ (double*)b[proc]), *(j+ (double*)a));
*/
            /* accumulate odd numbered elements */
            for(j=0; j< elems/2; j++){
                psrc[j]= 2*j+1 + (double*)a;
                pdst[j]= 2*j+1 + (double*)b[proc];
            }
            (void)ARMCI_AccV(ARMCI_ACC_DBL, &alpha, &dsc, 1, proc);

/*            for(j=0; j<elems; j++)
                printf("%d %lf %lf\n",j, *(j+ (double*)a), *(j+ (double*)b[proc]));
*/
        }

        ARMCI_AllFence();
        MP_BARRIER();

        /* copy my patch into local array c */
	assert(!ARMCI_Get((double*)b[proc], c, bytes, proc));

/*        scale = alpha*TIMES*nproc; */
        scale = alpha*TIMES*nproc*nproc; 
        scale_patch(scale, dim, a, &one, &elems, &elems);
        
        compare_patches(.0001, dim, a, &one, &elems, &elems, c, &one, &elems, &elems);
        MP_BARRIER();

        if(0==me){
            printf(" OK\n\n");
            fflush(stdout);
        }

        free(c);
        destroy_array((void**)b);
        free(a);
}



void test_fetch_add()
{
    int rc, bytes, i, val, times =0;
    int *arr[MAXPROC];

    /* shared variable is located on processor 0 */
    bytes = me == 0 ? sizeof(int) : 0;

    rc = ARMCI_Malloc((void**)arr,bytes);
    assert(rc==0);
    MP_BARRIER();

    if(me == 0) *arr[0] = 0;  /* initialization */

    MP_BARRIER();

    /* show what everybody gets */
    rc = ARMCI_Rmw(ARMCI_FETCH_AND_ADD, &val, arr[0], 1, 0);
    assert(rc==0);

    for(i = 0; i< nproc; i++){
        if(i==me){
            printf("process %d got value of %d\n",i,val);
            fflush(stdout);
        }
        MP_BARRIER();
    }

    if(me == 0){
      printf("\nIncrement the shared counter until reaches %d\n",LOOP);
      fflush(stdout);
    }
    
    MP_BARRIER();

    /* now increment the counter value until reaches LOOP */
    while(val<LOOP){
          rc = ARMCI_Rmw(ARMCI_FETCH_AND_ADD, &val, arr[0], 1, 0);
          assert(rc==0);
          times++;
    }

    for(i = 0; i< nproc; i++){
        if(i==me){
            printf("process %d incremented the counter %d times value=%d\n",i,times,val);
            fflush(stdout);
        }
        MP_BARRIER();
    }


    if(me == 0) *arr[0] = 0;  /* set it back to 0 */
    if(me == 0){
       printf("\nNow everybody increments the counter %d times\n",LOOP); 
       fflush(stdout);
    }

    ARMCI_AllFence();
    MP_BARRIER();

    for(i = 0; i< LOOP; i++){
          rc = ARMCI_Rmw(ARMCI_FETCH_AND_ADD, &val, arr[0], 1, 0);
          assert(rc==0);
    }

    ARMCI_AllFence();
    MP_BARRIER();

    if(me == 0){
       printf("The final value is %d, should be %d.\n\n",*arr[0],LOOP*nproc); 
       fflush(stdout);
       if( *arr[0] != LOOP*nproc) ARMCI_Error("failed ...",*arr[0]);
    }

    ARMCI_Free(arr[me]);
}


#define LOCKED -1
void test_swap()
{
    int rc, bytes, i, val, whatever=-8999;
    int *arr[MAXPROC];

    /* shared variable is located on processor 0 */
    bytes = me == 0 ? sizeof(int) : 0;

    rc = ARMCI_Malloc((void**)arr,bytes);
    assert(rc==0);
    MP_BARRIER();

    if(me == 0) *arr[0] = 0;  /* initialization */

    MP_BARRIER();
    for(i = 0; i< LOOP; i++){
          val = LOCKED;
          do{
            rc = ARMCI_Rmw(ARMCI_SWAP, &val, arr[0], whatever, 0);
            assert(rc==0);
          }while (val == LOCKED); 
          val++;
          rc = ARMCI_Rmw(ARMCI_SWAP, &val, arr[0], whatever, 0);
          assert(rc==0);
    }


    ARMCI_AllFence();
    MP_BARRIER();

    if(me == 0){
       printf("The final value is %d, should be %d.\n\n",*arr[0],LOOP*nproc); 
       fflush(stdout);
       if( *arr[0] != LOOP*nproc) ARMCI_Error("failed ...",*arr[0]);
    }

    ARMCI_Free(arr[me]);
}




void test_memlock()
{
        int dim,elems,bytes;
        int i, j,k, proc;
        double *b[MAXPROC];
        double *a, *c;
        int *proclist = (int*)work;
                void *pstart, *pend;
                int first, last;
                void armci_lockmem(void*, void*, int);
                void armci_unlockmem(void);

        elems = ELEMS;
        dim =1;

                bytes = elems*sizeof(double);

        /* create shared and local arrays */
        create_array((void**)b, sizeof(double),dim,&elems);
        a = (double*)malloc(bytes);
        assert(a);
        c = (double*)malloc(bytes);
        assert(c);
        
        /* initialize all elements of array b to zero */
        for(i=0;i<elems;i++)b[me][i]=-1.;

        sleep(1);

        proc=0;
                for(i=0;i<ELEMS/5;i++)a[i]=me;

        MP_BARRIER();
        for(j=0;j<10*TIMES;j++){ 
         for(i=0;i<TIMES*nproc;i++){ 
            first = rand()%(ELEMS/2);
                    last = first+ELEMS/5 -1;
                        pstart = b[proc]+first;
                        pend = b[proc]+last+1;
                        elems = last -first +1;
                        bytes = sizeof(double)*elems;

            armci_lockmem(pstart,pend,proc);
            assert(!ARMCI_Put(a, pstart, bytes, proc));
            assert(!ARMCI_Get(pstart, c, bytes, proc));
            assert(!ARMCI_Get(pstart, c, bytes, proc));
            armci_unlockmem();
            for(k=0;k<elems;k++)if(a[k]!=c[k]){
                printf("%d: error patch (%d:%d) elem=%d val=%lf\n",me,first,last,k,c[k]);
                fflush(stdout);
                ARMCI_Error("failed is ",(int)c[k]);
            }

          }
          if(0==me)fprintf(stderr,"done %d\n",j);
                }

        MP_BARRIER();


        if(0==me){
            printf(" OK\n\n");
            fflush(stdout);
        }

        free(c);
        destroy_array((void**)b);
        free(a);
}




/* we need to rename main if linking with frt compiler */
#ifdef FUJITSU_FRT
#define main MAIN__
#endif


int main(int argc, char* argv[])
{
    int ndim;

    MP_INIT(argc, argv);
    MP_PROCS(&nproc);
    MP_MYID(&me);

/*    printf("nproc = %d, me = %d\n", nproc, me);*/
    
    if(nproc>MAXPROC && me==0)
       ARMCI_Error("Test works for up to %d processors\n",MAXPROC);

    if(me==0){
       printf("ARMCI test program (%d processes)\n",nproc); 
       fflush(stdout);
       sleep(1);
    }
    
    ARMCI_Init();

/*
       if(me==1)armci_die("process 1 committing suicide",1);
*/
        if(me==0){
           printf("\nTesting strided gets and puts\n");
           printf("(Only std output for process 0 is printed)\n\n"); 
           fflush(stdout);
           sleep(1);
        }

        for(ndim=1; ndim<= MAXDIMS; ndim++) test_dim(ndim);
        ARMCI_AllFence();
        MP_BARRIER();

        if(me==0){
           printf("\nTesting atomic accumulate\n");
           fflush(stdout);
           sleep(1);
        }
        for(ndim=1; ndim<= MAXDIMS; ndim++) test_acc(ndim); 
        ARMCI_AllFence();
        MP_BARRIER();

        if(me==0){
           printf("\nTesting Vector Interface using triangular patches of a 2-D array\n\n");
           fflush(stdout);
           sleep(1);
        }

        test_vector();
        ARMCI_AllFence();
        MP_BARRIER();

        if(me==0){
           printf("\nTesting Accumulate with Vector Interface\n\n");
           fflush(stdout);
           sleep(1);
        }
        test_vector_acc();

        ARMCI_AllFence();
        MP_BARRIER();

        if(me==0){
           printf("\nTesting atomic fetch&add\n");
           printf("(Std Output for all processes is printed)\n\n"); 
           fflush(stdout);
           sleep(1);
        }
        MP_BARRIER();

        test_fetch_add();


        ARMCI_AllFence();
        MP_BARRIER();

        if(me==0){
           printf("\nTesting atomic swap\n");
           fflush(stdout);
        }
        test_swap();

        MP_BARRIER();
        /*test_memlock();*/

        MP_BARRIER();
	if(me==0){printf("All tests passed\n"); fflush(stdout);}
    sleep(2);

    MP_BARRIER();
    ARMCI_Finalize();
    MP_FINALIZE();
    return(0);
}

