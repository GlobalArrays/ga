/* $Id: message.c,v 1.17 2000-06-13 23:18:59 d3h325 Exp $ */
#if defined(PVM)
#   include <pvm3.h>
#elif defined(TCGMSG)
#   include <sndrcv.h>
#else
#   ifndef MPI
#      define MPI
#   endif
#   include <mpi.h>
#endif
#include "message.h"
#include "armcip.h"
#include "copy.h"

#ifdef CRAY
char *mp_group_name = (char *)NULL;
#else
char *mp_group_name = "mp_working_group";
#endif

#define BUF_SIZE  2048
#define INFO_BUF_SIZE  (BUF_SIZE*sizeof(BUF_SIZE) - sizeof(double))
static double work[BUF_SIZE];
static long *lwork = (long*)work;
static int *iwork = (int*)work;


void armci_msg_barrier()
{
#  ifdef MPI
     MPI_Barrier(MPI_COMM_WORLD);
#  elif defined(PVM)
     pvm_barrier(mp_group_name, armci_nproc);
#  else
     long tag=ARMCI_TAG;
     SYNCH_(&tag);
#  endif
}


void armci_exchange_address(void *ptr_ar[], int n)
{
  armci_msg_lgop((long*)ptr_ar, n, "+");
}


int armci_msg_me()
{
#  ifdef MPI
     int me;
     MPI_Comm_rank(MPI_COMM_WORLD, &me);
     return(me);
#  elif defined(PVM)
       return(pvm_getinst(mp_group_name,pvm_mytid()));
#  else
     return (int)NODEID_();
#  endif
}


int armci_msg_nproc()
{
#  ifdef MPI
     int nproc;
     MPI_Comm_size(MPI_COMM_WORLD, &nproc);
     return nproc;
#  elif defined(PVM)
     return(pvm_gsize(mp_group_name));
#  else
     return (int)NNODES_();
#  endif
}

void armci_msg_abort(int code)
{
#  ifdef MPI
     MPI_Abort(MPI_COMM_WORLD,code);
#  elif defined(PVM)
     char error_msg[25];
     sprintf(error_msg, "ARMCI aborting [%d]", code);
     pvm_halt();
#  else
     Error("ARMCI aborting",(long)code);
#  endif
}


static void armci_msg_bintree(int scope, int* Root, int *Up, int *Left, int *Right)
{
int root, up, left, right, index, nproc;
    if(scope == SCOPE_NODE){
       root  = armci_clus_info[armci_clus_me].master;
       nproc = armci_clus_info[armci_clus_me].nslave;
       index = armci_me - root;
       up    = (index-1)/2 + root; if( up < root) up = -1;
       left  = 2*index + 1 + root; if(left >= root+nproc) left = -1;
       right = 2*index + 2 + root; if(right >= root+nproc)right = -1;
    }else if(scope ==SCOPE_MASTERS){
       root  = armci_clus_info[0].master;
       nproc = armci_nclus;
       if(armci_me != armci_master){up = -1; left = -1; right = -1; }
       else{
               index = armci_clus_me - root;
               up    = (index-1)/2 + root;
               up = ( up < root)? -1: armci_clus_info[up].master;
               left  = 2*index + 1 + root;
                       left = ( left >= root+nproc)? -1: armci_clus_info[left].master;
               right = 2*index + 2 + root;
                       right = ( right >= root+nproc)? -1: armci_clus_info[right].master;
       }
    }else{
       root  = 0;
       nproc = armci_nproc;
       index = armci_me - root;
       up    = (index-1)/2 + root; if( up < root) up = -1;
       left  = 2*index + 1 + root; if(left >= root+nproc) left = -1;
       right = 2*index + 2 + root; if(right >= root+nproc)right = -1;
    }

    *Up = up;
    *Left = left;
    *Right = right;
    *Root = root;
}


/*\ root broadcasts to everyone else
\*/
void armci_msg_bcast_scope(int scope, void *buf, int len, int root)
{
    int up, left, right, Root;

    if(!buf)armci_die("armci_msg_bcast: NULL pointer", len);
    
    armci_msg_bintree(scope, &Root, &up, &left, &right);

    if(root !=Root){
        if(armci_me == root) armci_msg_snd(ARMCI_TAG, buf,len, Root);
        if(armci_me ==Root) armci_msg_rcv(ARMCI_TAG, buf, len, NULL, root);
    }
    
    /* printf("%d: scope=%d left=%d right=%d up=%d\n",armci_me, scope, left, right, up);*/

    if(armci_me != Root) armci_msg_rcv(ARMCI_TAG, buf, len, NULL, up);
    if (left > -1)  armci_msg_snd(ARMCI_TAG, buf, len, left);
    if (right > -1) armci_msg_snd(ARMCI_TAG, buf, len, right);
}


#ifndef armci_msg_bcast
/*\ SMP-aware global broadcast routine
\*/
void armci_msg_bcast(void *buf, int len, int root)
{
    /* inter-node operation between masters */
    if(armci_nclus>1)armci_msg_bcast_scope(SCOPE_MASTERS, buf, len, root); 

    /* intra-node operation */
    armci_msg_bcast_scope(SCOPE_NODE, buf, len, armci_master);
}
#endif


void armci_msg_brdcst(void* buffer, int len, int root)
{
   if(!buffer)armci_die("armci_msg_brdcast: NULL pointer", len);

#  ifdef MPI
      MPI_Bcast(buffer, len, MPI_CHAR, root, MPI_COMM_WORLD);
#  elif defined(PVM)
      armci_msg_bcast(buffer, len, root);
#  else
   {
      long ttag=ARMCI_TAG, llen=len, rroot=root;
      BRDCST_(&ttag, buffer, &llen, &rroot);
   }
#  endif
}


void armci_msg_snd(int tag, void* buffer, int len, int to)
{
#  ifdef MPI
      MPI_Send(buffer, len, MPI_CHAR, to, tag, MPI_COMM_WORLD);
#  elif defined(PVM)
      pvm_psend(pvm_gettid(mp_group_name, to), tag, buffer, len, PVM_BYTE);
#  else
      long ttag=tag, llen=len, tto=to, block=1;
      SND_(&ttag, buffer, &llen, &tto, &block);
#  endif
}


/*\ receive message of specified tag from proc and get its len if msglen!=NULL 
\*/
void armci_msg_rcv(int tag, void* buffer, int buflen, int *msglen, int from)
{
#  ifdef MPI
      MPI_Status status;
      MPI_Recv(buffer, buflen, MPI_CHAR, from, tag, MPI_COMM_WORLD, &status);
      if(msglen) MPI_Get_count(&status, MPI_CHAR, msglen);
#  elif defined(PVM)
      int src, rtag,mlen;
      pvm_precv(pvm_gettid(mp_group_name, from), tag, buffer, buflen, PVM_BYTE,
                &src, &rtag, &mlen);
      if(msglen)*msglen=mlen;
#  else
      long ttag=tag, llen=buflen, mlen, ffrom=from, sender, block=1;
      RCV_(&ttag, buffer, &llen, &mlen, &ffrom, &sender, &block);
      if(msglen)*msglen = (int)mlen;
#  endif
}


int armci_msg_rcvany(int tag, void* buffer, int buflen, int *msglen)
{
#if defined(MPI)
      int ierr;
      MPI_Status status;

      ierr = MPI_Recv(buffer, buflen, MPI_CHAR, MPI_ANY_SOURCE, tag,
             MPI_COMM_WORLD, &status);
      if(ierr != MPI_SUCCESS) armci_die("armci_msg_rcvany: Recv failed ", tag);

      if(msglen)if(MPI_SUCCESS!=MPI_Get_count(&status, MPI_CHAR, msglen))
                       armci_die("armci_msg_rcvany: count failed ", tag);
      return (int)status.MPI_SOURCE;
#  elif defined(PVM)
      int src, rtag,mlen;
      pvm_precv(-1, tag, buffer, buflen, PVM_BYTE, &src, &rtag, &mlen);
      if(msglen)*msglen=mlen;
      return(pvm_getinst(mp_group_name,src));
#  else
      long ttag=tag, llen=buflen, mlen, ffrom=-1, sender, block=1;
      RCV_(&ttag, buffer, &llen, &mlen, &ffrom, &sender, &block);
      if(msglen)*msglen = (int)mlen;
      return (int)sender;
#  endif
}


/*\ cluster master broadcasts to everyone else in the same cluster
\*/
void armci_msg_clus_brdcst(void *buf, int len)
{
int root, up, left, right;
int tag=ARMCI_TAG, lenmes;

    armci_msg_bintree(SCOPE_NODE, &root, &up, &left, &right);
    if(armci_me != root) armci_msg_rcv(tag, buf, len, &lenmes, up);
    if (left > -1)  armci_msg_snd(tag, buf, len, left);
    if (right > -1) armci_msg_snd(tag, buf, len, right);
}


/*\ reduce operation for long 
\*/
static void ldoop(int n, char *op, long *x, long* work)
{
  if (strncmp(op,"+",1) == 0)
    while(n--)
      *x++ += *work++;
  else if (strncmp(op,"*",1) == 0)
    while(n--)
      *x++ *= *work++;
  else if (strncmp(op,"max",3) == 0)
    while(n--) {
      *x = MAX(*x, *work);
      x++; work++;
    }
  else if (strncmp(op,"min",3) == 0)
    while(n--) {
      *x = MIN(*x, *work);
      x++; work++;
    }
  else if (strncmp(op,"absmax",6) == 0)
    while(n--) {
      register long x1 = ABS(*x), x2 = ABS(*work);
      *x = MAX(x1, x2);
      x++; work++;
    }
  else if (strncmp(op,"absmin",6) == 0)
    while(n--) {
      register long x1 = ABS(*x), x2 = ABS(*work);
      *x = MIN(x1, x2);
      x++; work++;
    }
  else if (strncmp(op,"or",2) == 0)
    while(n--) {
      *x |= *work;
      x++; work++;
    }
  else
    armci_die("ldoop: unknown operation requested", n);
}

static void idoop(int n, char *op, int *x, int* work)
{
  if (strncmp(op,"+",1) == 0)
    while(n--)
      *x++ += *work++;
  else if (strncmp(op,"*",1) == 0)
    while(n--)
      *x++ *= *work++;
  else if (strncmp(op,"max",3) == 0)
    while(n--) {
      *x = MAX(*x, *work);
      x++; work++;
    }
  else if (strncmp(op,"min",3) == 0)
    while(n--) {
      *x = MIN(*x, *work);
      x++; work++;
    }
  else if (strncmp(op,"absmax",6) == 0)
    while(n--) {
      register int x1 = ABS(*x), x2 = ABS(*work);
      *x = MAX(x1, x2);
      x++; work++;
    }
  else if (strncmp(op,"absmin",6) == 0)
    while(n--) {
      register int x1 = ABS(*x), x2 = ABS(*work);
      *x = MIN(x1, x2);
      x++; work++;
    }
  else if (strncmp(op,"or",2) == 0)
    while(n--) {
      *x |= *work;
      x++; work++;
    }
  else
    armci_die("idoop: unknown operation requested", n);
}


static void ddoop(int n, char* op, double* x, double* work)
{
  if (strncmp(op,"+",1) == 0)
    while(n--)
      *x++ += *work++;
  else if (strncmp(op,"*",1) == 0)
    while(n--)
      *x++ *= *work++;
  else if (strncmp(op,"max",3) == 0)
    while(n--) {
      *x = MAX(*x, *work);
      x++; work++;
    }
  else if (strncmp(op,"min",3) == 0)
    while(n--) {
      *x = MIN(*x, *work);
      x++; work++;
    }
  else if (strncmp(op,"absmax",6) == 0)
    while(n--) {
      register double x1 = ABS(*x), x2 = ABS(*work);
      *x = MAX(x1, x2);
      x++; work++;
    }
  else if (strncmp(op,"absmin",6) == 0)
    while(n--) {
      register double x1 = ABS(*x), x2 = ABS(*work);
      *x = MIN(x1, x2);
      x++; work++;
    }
  else
    armci_die("ddoop: unknown operation requested", n);
}



/*\ combine array of longs/ints accross all processes
\*/
void armci_msg_gop(int scope, void *x, int n, char* op, int type)
{
int root, up, left, right, size;
int tag=ARMCI_TAG;
int ndo, len, lenmes, orign =n, ratio;
void *origx =x;


    if(!x)armci_die("armci_msg_gop: NULL pointer", n);

    armci_msg_bintree(scope, &root, &up, &left, &right);

    if(type==ARMCI_INT) size = sizeof(int);
	else if(type==ARMCI_LONG) size = sizeof(long);
    else size = sizeof(double);

    ratio = sizeof(double)/size;
    
    while ((ndo = (n<=BUF_SIZE*ratio) ? n : BUF_SIZE*ratio)) {
         len = lenmes = ndo*size;

         if (left > -1) {
           armci_msg_rcv(tag, lwork, len, &lenmes, left);
           if(type==ARMCI_INT) idoop(ndo, op, (int*)x, iwork);
           else if(type==ARMCI_LONG) ldoop(ndo, op, (long*)x, lwork);
           else ddoop(ndo, op, (double*)x, work);
         }

         if (right > -1) {
           armci_msg_rcv(tag, lwork, len, &lenmes, right);
           if(type==ARMCI_INT) idoop(ndo, op, (int*)x, iwork);
           else if(type==ARMCI_LONG) ldoop(ndo, op, (long*)x, lwork);
           else ddoop(ndo, op, (double*)x, work);
         }
         if (armci_me != root) armci_msg_snd(tag, x, len, up);

         n -=ndo;
         x = len + (char*)x;
     }

     /* Now, root broadcasts the result down the binary tree */
     len = orign*size;

     if(scope == SCOPE_NODE) armci_msg_clus_brdcst(origx, len );
     else armci_msg_brdcst(origx, len,0 );
}


void armci_msg_reduce_scope(int scope, void *x, int n, char* op, int type)
{
int root, up, left, right, size;
int tag=ARMCI_TAG;
int ndo, len, lenmes, ratio;


    if(!x)armci_die("armci_msg_gop: NULL pointer", n);

    armci_msg_bintree(scope, &root, &up, &left, &right);

    if(type==ARMCI_INT) size = sizeof(int);
        else if(type==ARMCI_LONG) size = sizeof(long);
    else size = sizeof(double);

    ratio = sizeof(double)/size;
   
    while ((ndo = (n<=BUF_SIZE*ratio) ? n : BUF_SIZE*ratio)) {
         len = lenmes = ndo*size;

         if (left > -1) {
           armci_msg_rcv(tag, lwork, len, &lenmes, left);
           if(type==ARMCI_INT) idoop(ndo, op, (int*)x, iwork);
           else if(type==ARMCI_LONG) ldoop(ndo, op, (long*)x, lwork);
           else ddoop(ndo, op, (double*)x, work);
         }

         if (right > -1) {
           armci_msg_rcv(tag, lwork, len, &lenmes, right);
           if(type==ARMCI_INT) idoop(ndo, op, (int*)x, iwork);
           else if(type==ARMCI_LONG) ldoop(ndo, op, (long*)x, lwork);
           else ddoop(ndo, op, (double*)x, work);
         }
         if (armci_me != root) armci_msg_snd(tag, x, len, up);

         n -=ndo;
         x = len + (char*)x;
     }
}


static void armci_msg_reduce(void *x, int n, char* op, int type)
{
    /* inter-node operation between masters */
    if(armci_nclus>1)armci_msg_reduce_scope(SCOPE_MASTERS, x, n, op, type);

    /* intra-node operation */
    armci_msg_reduce_scope(SCOPE_NODE, x, n, op, type);
}


static void armci_msg_gop2(void *x, int n, char* op, int type)
{
int size, root=0;

     if(type==ARMCI_INT) size = sizeof(int);
        else if(type==ARMCI_LONG) size = sizeof(long);
     else size = sizeof(double);

     armci_msg_reduce(x, n, op, type);
     armci_msg_bcast(x, size, root);
}


static void armci_sel(int type, char *op, void *x, void* work, int n)
{
int selected=0;
  switch (type) {
  case ARMCI_INT:
     if(strncmp(op,"min",3) == 0){ 
        if(*(int*)x > *(int*)work) selected=1;
     }else
        if(*(int*)x < *(int*)work) selected=1;
     break;
  case ARMCI_LONG:
     if(strncmp(op,"min",3) == 0){ 
        if(*(long*)x > *(long*)work) selected=1;
     }else
        if(*(long*)x < *(long*)work) selected=1;
     break;
  default:
     if(strncmp(op,"min",3) == 0){
        if(*(double*)x > *(double*)work) selected=1;
     }else
        if(*(double*)x < *(double*)work) selected=1;
  }
  if(selected)armci_copy(work,x, n);
}
   
 

/*\ global for  op with extra info 
\*/
void armci_msg_sel_scope(int scope, void *x, int n, char* op, int type, int contribute)
{
int root, up, left, right;
int tag=ARMCI_TAG;
int len, lenmes, min;

    min = (strncmp(op,"min",3) == 0);
    if(!min && (strncmp(op,"max",3) != 0))
            armci_die("armci_msg_gop_info: operation not supported ", 0);

    if(!x)armci_die("armci_msg_gop_info: NULL pointer", n);

    if(n>INFO_BUF_SIZE)armci_die("armci_msg_gop_info: info too large",n);

    len = lenmes = n;

    armci_msg_bintree(scope, &root, &up, &left, &right);

    if (left > -1) {

        /* receive into work if contributing otherwise into x */
        if(contribute)armci_msg_rcv(tag, work, len, &lenmes, left);
        else armci_msg_rcv(tag, x, len, &lenmes, left);

        if(lenmes){
           if(contribute) armci_sel(type, op, x, work, n);
           else contribute =1; /* now we got data to pass */ 
        }
    }

    if (right > -1) {
        /* receive into work if contributing otherwise into x */
        if(contribute) armci_msg_rcv(tag, work, len, &lenmes, right);
        else armci_msg_rcv(tag, x, len, &lenmes, right);

        if(lenmes){
           if(contribute) armci_sel(type, op, x, work, n);
           else contribute =1; /* now we got data to pass */ 
        }
    }

    if (armci_me != root)
       if(contribute) armci_msg_snd(tag, x, len, up);
       else armci_msg_snd(tag, x, 0, up); /* send 0 bytes */

    /* Now, root broadcasts the result down the binary tree */
    armci_msg_bcast_scope(scope, x, n, root);
}



/*\ combine array of longs/ints/doubles accross all processes
\*/
#if 1
void armci_msg_igop(int *x, int n, char* op)
{ armci_msg_gop(SCOPE_ALL,x, n, op, ARMCI_INT); }

void armci_msg_lgop(long *x, int n, char* op)
{ armci_msg_gop(SCOPE_ALL,x, n, op, ARMCI_LONG); }

void armci_msg_dgop(double *x, int n, char* op)
{ armci_msg_gop(SCOPE_ALL,x, n, op, ARMCI_DOUBLE); }
#else
void armci_msg_igop(int *x, int n, char* op) { armci_msg_gop2(x, n, op, ARMCI_INT); }
void armci_msg_lgop(long *x, int n, char* op) { armci_msg_gop2(x, n, op, ARMCI_LONG); }
void armci_msg_dgop(double *x, int n, char* op) { armci_msg_gop2(x, n, op, ARMCI_DOUBLE); }
#endif


/*\ add array of longs/ints within the same cluster 
\*/
void armci_msg_clus_igop(int *x, int n, char* op)
{ armci_msg_gop(SCOPE_NODE,x, n, op, ARMCI_INT); }

void armci_msg_clus_lgop(long *x, int n, char* op)
{ armci_msg_gop(SCOPE_NODE,x, n, op, ARMCI_INT); }

void armci_msg_clus_dgop(double *x, int n, char* op)
{ armci_msg_gop(SCOPE_NODE,x, n, op, ARMCI_DOUBLE); }


#ifdef PVM
/* set the group name if using PVM */
void ARMCI_PVM_Init(char *mpgroup)
{
#ifdef CRAY
    mp_group_name = (char *)NULL;
#else
    if(mpgroup != NULL) {
/*        free(mp_group_name); */
        mp_group_name = (char *)malloc(25 * sizeof(char));
        strcpy(mp_group_name, mpgroup);
    }
#endif
}
#endif
