/* $Id: message.c,v 1.6 1999-10-14 00:18:51 d3h325 Exp $ */
#include "message.h"
#include "armcip.h"

#ifdef CRAY
char *mp_group_name = (char *)NULL;
#else
char *mp_group_name = "mp_working_group";
#endif

static double work[BUF_SIZE];
static long *lwork = (long*)work;

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
  armci_msg_igop((long*)ptr_ar, n, "+", 1);
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

/*\ root broadcasts to everyone else
\*/
void armci_msg_bcast(void *buf, int len, int root)
{
    int up, left, right, index, Root=0;
    int tag=ARMCI_TAG, lenmes;
    
    if(root !=Root){
        int msglen;
        
        if(armci_me == root) armci_msg_snd(ARMCI_TAG, buf,len, Root);
        if(armci_me ==Root) armci_msg_rcv(ARMCI_TAG, buf, len, &msglen, root);
    }
    
    index = armci_me - Root;
    up    = (index-1)/2 + Root; if( up < Root) up = -1;
    left  = 2*index + 1 + Root; if(left >= Root+armci_nproc) left = -1;
    right = 2*index + 2 + Root; if(right >= Root+armci_nproc)right = -1;
    
    if(armci_me != Root) armci_msg_rcv(tag, buf, len, &lenmes, up);
    if (left > -1)  armci_msg_snd(tag, buf, len, left);
    if (right > -1) armci_msg_snd(tag, buf, len, right);
}

void armci_msg_brdcst(void* buffer, int len, int root)
{
#  ifdef MPI
      MPI_Bcast(buffer, len, MPI_CHAR, root, MPI_COMM_WORLD);
#  elif defined(PVM)
      armci_msg_bcast(buffer, len, root);
#  else
      long ttag=ARMCI_TAG, llen=len, rroot=root;
      BRDCST_(&ttag, buffer, &llen, &rroot);
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


void armci_msg_rcv(int tag, void* buffer, int buflen, int *msglen, int from)
{
#  ifdef MPI
      MPI_Status status;
      MPI_Recv(buffer, buflen, MPI_CHAR, from, tag, MPI_COMM_WORLD, &status);
      MPI_Get_count(&status, MPI_CHAR, msglen);
#  elif defined(PVM)
      int src, rtag;
      pvm_precv(pvm_gettid(mp_group_name, from), tag, buffer, buflen, PVM_BYTE,
                &src, &rtag, msglen);
#  else
      long ttag=tag, llen=buflen, mmsglen, ffrom=from, sender, block=1;
      RCV_(&ttag, buffer, &llen, &mmsglen, &ffrom, &sender, &block);
      *msglen = (int)mmsglen;
#  endif
}


/*\ cluster master broadcasts to everyone else in the same cluster
\*/
void armci_msg_clus_brdcst(void *buf, int len)
{
int root, up, left, right, index, nproc;
int tag=ARMCI_TAG, lenmes;

    root  = armci_clus_info[armci_clus_me].master;
    nproc = armci_clus_info[armci_clus_me].nslave;
    index = armci_me - root;
    up    = (index-1)/2 + root; if( up < root) up = -1;
    left  = 2*index + 1 + root; if(left >= root+nproc) left = -1;
    right = 2*index + 2 + root; if(right >= root+nproc)right = -1;

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
    armci_die("ldoop: unknown operation requested", n);
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


/*\ add array of longs/ints within the same cluster 
\*/
void armci_msg_clus_igop(long *x, int n, char* op, int longint)
{
int root, up, left, right, index, nproc, size=sizeof(long);
int tag=ARMCI_TAG;
int ndo, len, lenmes, orign =n, bufsize;
long *origx =x;  

    root  = armci_clus_info[armci_clus_me].master;
    nproc = armci_clus_info[armci_clus_me].nslave;
    index = armci_me - root;
    up    = (index-1)/2 + root; if( up < root) up = -1;
    left  = 2*index + 1 + root; if(left >= root+nproc) left = -1;
    right = 2*index + 2 + root; if(right >= root+nproc)right = -1;

    bufsize = BUF_SIZE;
    if(!longint) size =sizeof(int);
    bufsize *= sizeof(double)/size;

    while ((ndo = (n<=bufsize) ? n : bufsize)) {
         len = lenmes = ndo*size;

         if (left > -1) {
           armci_msg_rcv(tag, lwork, len, &lenmes, left);
           if(longint)ldoop(ndo, op, x, lwork);
           else idoop(ndo, op, (int*)x, (int*)lwork);
         }
         if (right > -1) {
           armci_msg_rcv(tag, lwork, len, &lenmes, right);
           if(longint)ldoop(ndo, op, x, lwork);
           else idoop(ndo, op, (int*)x, (int*)lwork);
         }
         if (armci_me != root) armci_msg_snd(tag, x, len, up);

       n -=ndo;
       x +=ndo;
     }

     /* Now, root broadcasts the result down the binary tree */
     len = orign*size;
     armci_msg_clus_brdcst(origx, len );
}



/*\ combine array of longs/ints accross all processes
\*/
void armci_msg_igop(long *x, int n, char* op, int longint)
{
int root, up, left, right, index, nproc,size=sizeof(long);
int tag=ARMCI_TAG;
int ndo, len, lenmes, orign =n, bufsize;
long *origx =x;

    root  = 0;
    nproc = armci_nproc;
    index = armci_me - root;
    up    = (index-1)/2 + root; if( up < root) up = -1;
    left  = 2*index + 1 + root; if(left >= root+nproc) left = -1;
    right = 2*index + 2 + root; if(right >= root+nproc)right = -1;

    bufsize = BUF_SIZE;
    if(!longint) size =sizeof(int);
    bufsize *= sizeof(double)/size;

    while ((ndo = (n<=BUF_SIZE) ? n : BUF_SIZE)) {
         len = lenmes = ndo*size;

         if (left > -1) {
           armci_msg_rcv(tag, lwork, len, &lenmes, left);
           if(longint)ldoop(ndo, op, x, lwork);
           else idoop(ndo, op, (int*)x, (int*)lwork);
         }
         if (right > -1) {
           armci_msg_rcv(tag, lwork, len, &lenmes, right);
           if(longint)ldoop(ndo, op, x, lwork);
           else idoop(ndo, op, (int*)x, (int*)lwork);
         }
         if (armci_me != root) armci_msg_snd(tag, x, len, up);

         n -=ndo;
         x +=ndo;
     }

     /* Now, root broadcasts the result down the binary tree */
     len = orign*size;
     armci_msg_brdcst(origx, len,0 );
}


/*\ add array of doubles accross all processes
\*/
void armci_msg_dgop(double *x, int n, char* op)
{
int root, up, left, right, index, nproc,size=sizeof(double);
int tag=ARMCI_TAG;
int ndo, len, lenmes, orign =n;
double *origx =x;

    root  = 0;
    nproc = armci_nproc;
    index = armci_me - root;
    up    = (index-1)/2 + root; if( up < root) up = -1;
    left  = 2*index + 1 + root; if(left >= root+nproc) left = -1;
    right = 2*index + 2 + root; if(right >= root+nproc)right = -1;

    while ((ndo = (n<=BUF_SIZE) ? n : BUF_SIZE)) {
         len = lenmes = ndo*size;

         if (left > -1) {
           armci_msg_rcv(tag, work, len, &lenmes, left);
           ddoop(ndo, op, x, work);
         }
         if (right > -1) {
           armci_msg_rcv(tag, work, len, &lenmes, right);
           ddoop(ndo, op, x, work);
         }
         if (armci_me != root) armci_msg_snd(tag, x, len, up);

         n -=ndo;
         x +=ndo;
     }

     /* Now, root broadcasts the result down the binary tree */
     len = orign*size;
     armci_msg_brdcst(origx, len,0 );
}

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
