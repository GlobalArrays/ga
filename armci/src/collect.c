#include <stdio.h>

#include <mpi.h>





#define MIN(a,b) (((a) <= (b)) ? (a) : (b))

#define ABS(a)   (((a) >= 0)   ? (a) : (-(a)))





#define Error(str,code){\
        fprintf(stderr,"%s",(str)); MPI_Abort(MPI_COMM_WORLD,(int)code);\
 }



int me, nproc;







/* size of internal buffer for global ops */

#define DGOP_BUF_SIZE 65536

#define IGOP_BUF_SIZE (sizeof(double)/sizeof(int))*DGOP_BUF_SIZE



static double gop_work[DGOP_BUF_SIZE];              /* global ops buffer */







/*\ global operations -- integer version

\*/

void IGOP_(ptype, x, pn, op)

     int  *x;

     int  *ptype, *pn;

     char *op;

{

int *work   = (int *) gop_work;

int nleft  = *pn;

int buflen = MIN(nleft,IGOP_BUF_SIZE); /* Try to get even sized buffers */

int nbuf   = (nleft-1) / buflen + 1;

int n;



  buflen = (nleft-1) / nbuf + 1;



  if (strncmp(op,"abs",3) == 0) {

    n = *pn;

    while(n--) x[n] = ABS(x[n]);

  }



  while (nleft) {

    int root = 0;

    int ierr  ;

    int ndo = MIN(nleft, buflen);



    if (strncmp(op,"+",1) == 0)

      ierr   = MPI_Reduce(x, work, ndo, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);

    else if (strncmp(op,"*",1) == 0)

      ierr   = MPI_Reduce(x, work, ndo, MPI_INT, MPI_PROD, root, MPI_COMM_WORLD);

    else if (strncmp(op,"max",3) == 0 || strncmp(op,"absmax",6) == 0)

      ierr   = MPI_Reduce(x, work, ndo, MPI_INT, MPI_MAX, root, MPI_COMM_WORLD);

    else if (strncmp(op,"min",3) == 0 || strncmp(op,"absmin",6) == 0)

      ierr   = MPI_Reduce(x, work, ndo, MPI_INT, MPI_MIN, root, MPI_COMM_WORLD);

    else if (strncmp(op,"or",2) == 0)

      ierr   = MPI_Reduce(x, work, ndo, MPI_INT, MPI_BOR, root, MPI_COMM_WORLD);

    else

      Error("IGOP: unknown operation requested", (int) *pn);



    ierr   = MPI_Bcast(work, ndo, MPI_INT, root, MPI_COMM_WORLD);



    n = ndo;

    while(n--) x[n] = work[n];



    nleft -= ndo; x+= ndo;

  }

}



/*\ global operations -- double version

\*/

void DGOP_(ptype, x, pn, op)

     double  *x;

     int     *ptype, *pn;

     char    *op;

{

double *work=  gop_work;

int nleft  = *pn;

int buflen = MIN(nleft,DGOP_BUF_SIZE); /* Try to get even sized buffers */

int nbuf   = (nleft-1) / buflen + 1;

int n;



  buflen = (nleft-1) / nbuf + 1;



  if (strncmp(op,"abs",3) == 0) {

    n = *pn;

    while(n--) x[n] = ABS(x[n]);

  }



  while (nleft) {

    int root = 0;

    int ierr  ;

    int ndo = MIN(nleft, buflen);



    if (strncmp(op,"+",1) == 0)

      ierr   = MPI_Reduce(x, work, ndo, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);

    else if (strncmp(op,"*",1) == 0)

      ierr   = MPI_Reduce(x, work, ndo, MPI_DOUBLE, MPI_PROD, root, MPI_COMM_WORLD);

    else if (strncmp(op,"max",3) == 0 || strncmp(op,"absmax",6) == 0)

      ierr   = MPI_Reduce(x, work, ndo, MPI_DOUBLE, MPI_MAX, root, MPI_COMM_WORLD);

    else if (strncmp(op,"min",3) == 0 || strncmp(op,"absmin",6) == 0)

      ierr   = MPI_Reduce(x, work, ndo, MPI_DOUBLE, MPI_MIN, root, MPI_COMM_WORLD);

    else

      Error("DGOP: unknown operation requested", (int) *pn);



    ierr   = MPI_Bcast(work, ndo, MPI_DOUBLE, root, MPI_COMM_WORLD);



    n = ndo;

    while(n--) x[n] = work[n];



    nleft -= ndo; x+= ndo;

  }

}







void BRDCST_(type, buf, lenbuf, originator)

     int  *type;

     char *buf;

     int  *lenbuf;

     int  *originator;

{

int count = *lenbuf, root = *originator;



     MPI_Bcast(buf, count, MPI_CHAR, root,  MPI_COMM_WORLD);

}









static void TestGlobals()

{

#define MAXLENG 256*1024

  double *dtest;

  int *itest;

  int len;

  int from=nproc-1;

  int itype=3, dtype=4;



  if (me == 0) {

    (void) printf("Global test ... test brodcast, igop and dgop\n----------\n\n"

);

    (void) fflush(stdout);

  }



  if (!(dtest = (double *) malloc((unsigned) (MAXLENG*sizeof(double)))))

    Error("TestGlobals: failed to allocated dtest", (int) MAXLENG);

  if (!(itest = (int *) malloc((unsigned) (MAXLENG*sizeof(int)))))

    Error("TestGlobals: failed to allocated itest", (int) MAXLENG);



  for (len=1; len<MAXLENG; len*=2) {

    int ilen = len*sizeof(int);

    int dlen = len*sizeof(double);

    int i;



    if (me == 0) {

      printf("Test length = %d ... ", len);

      fflush(stdout);

    }



    /* Test broadcast */



    if (me == (nproc-1)) {

      for (i=0; i<len; i++) {

        itest[i] = i;

        dtest[i] = (double) itest[i];

      }

    }

    else {

      for (i=0; i<len; i++) {

        itest[i] = 0;

        dtest[i] = 0.0;

      }

    }

    BRDCST_(&itype, (char *) itest, &ilen, &from);

    BRDCST_(&dtype, (char *) dtest, &dlen, &from);



    for (i=0; i<len; i++)

      if (itest[i] != i || dtest[i] != (double) i)

        Error("TestGlobal: broadcast failed", (int) i);



    if (me == 0) {

      printf("broadcast OK ...");

      fflush(stdout);

    }



    /* Test global sum */



    for (i=0; i<len; i++) {

      itest[i] = i*me;

      dtest[i] = (double) itest[i];

    }



    IGOP_(&itype, itest, &len, "+");

    DGOP_(&dtype, dtest, &len, "+");



    for (i=0; i<len; i++) {

      int iresult = i*nproc*(nproc-1)/2;

      if (itest[i] != iresult || dtest[i] != (double) iresult)

        Error("TestGlobals: global sum failed", (int) i);

    }



    if (me == 0) {

      printf("global sums OK\n");

      fflush(stdout);

    }



  }

     

  free((char *) itest);

  free((char *) dtest);

}

 



int main(argc, argv)

    int argc;

    char **argv;

{

      MPI_Init(&argc, &argv);

      MPI_Comm_size(MPI_COMM_WORLD, &nproc);

      MPI_Comm_rank(MPI_COMM_WORLD, &me);

      if(me==0)printf("Testing Global Operations (%d MPI processes)\n",nproc);

      TestGlobals();

      MPI_Finalize();

      return 0;

}



