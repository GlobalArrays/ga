/**
 * GA_Lu_solve_seq.c: Implemented with CLINPACK routines. Uses LINPACK
 * routines if GA_C_CORE is defined, else uses scalapack.
 */

#include "global.h"
#include "globalp.h"
#include "macdecls.h"
#include "math.h"
#ifdef GA_USE_VAMPIR
#  include "ga_vampir.h"
#endif

#ifdef CRAY
#  include <fortran.h>
#  define DGETRF SGETRF
#  define DGETRS SGETRS
#elif defined(WIN32)
extern void FATR DGETRF(Integer *, Integer *, void *, Integer *, void *, 
			Integer *);
extern void FATR DGETRS(char *, int, Integer *, Integer *, void *, 
			Integer *, void *, void *, Integer *, Integer *);
#elif defined(F2C2__)
#  define DGETRF dgetrf__
#  define DGETRS dgetrs__
#elif defined(HITACHI)
#  define dgetrf_ DGETRF
#  define dgetrs_ DGETRS
#else
#  define DGETRF dgetrf_
#  define DGETRS dgetrs_
#endif

#if defined(CRAY) || defined(WIN32)
#   define cptofcd(fcd)  _cptofcd((fcd),1)
#else
#      define cptofcd(fcd) (fcd)
#endif



#define REAL double
#define ZERO 0.0e0
#define ONE 1.0e0


#define ROLL

/** WHY ??? */
#ifdef ROLL
#define ROLLING "Rolled "
#endif
#ifdef UNROLL
#define ROLLING "Unrolled "
#endif

/*----------------------*/ 
void LP_daxpy(n,da,dx,incx,dy,incy)
/*
     constant times a vector plus a vector.
     jack dongarra, linpack, 3/11/78.
*/
REAL dx[],dy[],da;
int incx,incy,n;
{
	int i,ix,iy;

	if(n <= 0) return;
	if (da == ZERO) return;

	if(incx != 1 || incy != 1) {

		/* code for unequal increments or equal increments
		   not equal to 1 					*/

		ix = 0;
		iy = 0;
		if(incx < 0) ix = (-n+1)*incx;
		if(incy < 0)iy = (-n+1)*incy;
		for (i = 0;i < n; i++) {
			dy[iy] = dy[iy] + da*dx[ix];
			ix = ix + incx;
			iy = iy + incy;
		}
      		return;
	}

	/* code for both increments equal to 1 */

#ifdef ROLL
	for (i = 0;i < n; i++) {
		dy[i] = dy[i] + da*dx[i];
	}
#endif
#ifdef UNROLL

	m = n % 4;
	if ( m != 0) {
		for (i = 0; i < m; i++) 
			dy[i] = dy[i] + da*dx[i];
		if (n < 4) return;
	}
	for (i = m; i < n; i = i + 4) {
		dy[i] = dy[i] + da*dx[i];
		dy[i+1] = dy[i+1] + da*dx[i+1];
		dy[i+2] = dy[i+2] + da*dx[i+2];
		dy[i+3] = dy[i+3] + da*dx[i+3];
	}
#endif
}
   
/*----------------------*/ 

REAL LP_ddot(n,dx,incx,dy,incy)
/*
     forms the dot product of two vectors.
     jack dongarra, linpack, 3/11/78.
*/
REAL dx[],dy[];

int incx,incy,n;
{
	REAL dtemp;
	int i,ix,iy;

	dtemp = ZERO;

	if(n <= 0) return(ZERO);

	if(incx != 1 || incy != 1) {

		/* code for unequal increments or equal increments
		   not equal to 1					*/

		ix = 0;
		iy = 0;
		if (incx < 0) ix = (-n+1)*incx;
		if (incy < 0) iy = (-n+1)*incy;
		for (i = 0;i < n; i++) {
			dtemp = dtemp + dx[ix]*dy[iy];
			ix = ix + incx;
			iy = iy + incy;
		}
		return(dtemp);
	}

	/* code for both increments equal to 1 */

#ifdef ROLL
	for (i=0;i < n; i++)
		dtemp = dtemp + dx[i]*dy[i];
	return(dtemp);
#endif
#ifdef UNROLL

	m = n % 5;
	if (m != 0) {
		for (i = 0; i < m; i++)
			dtemp = dtemp + dx[i]*dy[i];
		if (n < 5) return(dtemp);
	}
	for (i = m; i < n; i = i + 5) {
		dtemp = dtemp + dx[i]*dy[i] +
		dx[i+1]*dy[i+1] + dx[i+2]*dy[i+2] +
		dx[i+3]*dy[i+3] + dx[i+4]*dy[i+4];
	}
	return(dtemp);
#endif
}

/*----------------------*/ 
void LP_dscal(n,da,dx,incx)

/*     scales a vector by a constant.
      jack dongarra, linpack, 3/11/78.
*/
REAL da,dx[];
int n, incx;
{
	int i,nincx;

	if(n <= 0)return;
	if(incx != 1) {

		/* code for increment not equal to 1 */

		nincx = n*incx;
		for (i = 0; i < nincx; i = i + incx)
			dx[i] = da*dx[i];
		return;
	}

	/* code for increment equal to 1 */

#ifdef ROLL
	for (i = 0; i < n; i++)
		dx[i] = da*dx[i];
#endif
#ifdef UNROLL

	m = n % 5;
	if (m != 0) {
		for (i = 0; i < m; i++)
			dx[i] = da*dx[i];
		if (n < 5) return;
	}
	for (i = m; i < n; i = i + 5){
		dx[i] = da*dx[i];
		dx[i+1] = da*dx[i+1];
		dx[i+2] = da*dx[i+2];
		dx[i+3] = da*dx[i+3];
		dx[i+4] = da*dx[i+4];
	}
#endif

}

/*----------------------*/ 
int LP_idamax(n,dx,incx)

/*
     finds the index of element having max. absolute value.
     jack dongarra, linpack, 3/11/78.
*/

REAL dx[];
int incx,n;
{
	REAL dmax;
	int i, ix, itemp;

	if( n < 1 ) return(-1);
	if(n ==1 ) return(0);
	if(incx != 1) {

		/* code for increment not equal to 1 */

		ix = 1;
		dmax = fabs((double)dx[0]);
		ix = ix + incx;
		for (i = 1; i < n; i++) {
			if(fabs((double)dx[ix]) > dmax)  {
				itemp = i;
				dmax = fabs((double)dx[ix]);
			}
			ix = ix + incx;
		}
	}
	else {

		/* code for increment equal to 1 */

		itemp = 0;
		dmax = fabs((double)dx[0]);
		for (i = 1; i < n; i++) {
			if(fabs((double)dx[i]) > dmax) {
				itemp = i;
				dmax = fabs((double)dx[i]);
			}
		}
	}
	return (itemp);
}



/*----------------------*/ 
void LP_dgefa(a,lda,n,ipvt,info)
REAL a[];
int lda,n,ipvt[],*info;

/* We would like to declare a[][lda], but c does not allow it.  In this
function, references to a[i][j] are written a[lda*i+j].  */
/*
     LP_dgefa factors a double precision matrix by gaussian elimination.

     LP_dgefa is usually called by dgeco, but it can be called
     directly with a saving in time if  rcond  is not needed.
     (time for dgeco) = (1 + 9/n)*(time for LP_dgefa) .

     on entry

        a       REAL precision[n][lda]
                the matrix to be factored.

        lda     integer
                the leading dimension of the array  a .

        n       integer
                the order of the matrix  a .

     on return

        a       an upper triangular matrix and the multipliers
                which were used to obtain it.
                the factorization can be written  a = l*u  where
                l  is a product of permutation and unit lower
                triangular matrices and  u  is upper triangular.

        ipvt    integer[n]
                an integer vector of pivot indices.

        info    integer
                = 0  normal value.
                = k  if  u[k][k] .eq. 0.0 .  this is not an error
                     condition for this subroutine, but it does
                     indicate that LP_dgesl or dgedi will divide by zero
                     if called.  use  rcond  in dgeco for a reliable
                     indication of singularity.

     linpack. this version dated 08/14/78 .
     cleve moler, university of new mexico, argonne national lab.

     functions

     blas LP_daxpy,LP_dscal,LP_idamax
*/

{
/*     internal variables	*/

  REAL t;
  int LP_idamax(),j,k,kp1,l,nm1;


/*     gaussian elimination with partial pivoting	*/
	*info = 0;
	nm1 = n - 1;
	if (nm1 >=  0) {
		for (k = 0; k < nm1; k++) {
			kp1 = k + 1;

          		/* find l = pivot index	*/

			l = LP_idamax(n-k,&a[lda*k+k],1) + k;
			ipvt[k] = l;

			/* zero pivot implies this column already 
			   triangularized */
			if (a[lda*k+l] != ZERO) {

				/* interchange if necessary */

				if (l != k) {
					t = a[lda*k+l];
					a[lda*k+l] = a[lda*k+k];
					a[lda*k+k] = t; 
				}

				/* compute multipliers */

				t = -ONE/a[lda*k+k];
				LP_dscal(n-(k+1),t,&a[lda*k+k+1],1);

				/* row elimination with column indexing */

				for (j = kp1; j < n; j++) {
					t = a[lda*j+l];
					if (l != k) {
						a[lda*j+l] = a[lda*j+k];
						a[lda*j+k] = t;
					}
					LP_daxpy(n-(k+1),t,&a[lda*k+k+1],1,
					      &a[lda*j+k+1],1);
  				} 
  			}
			else { 
            			*info = k;
			}
		} 
	}
	ipvt[n-1] = n-1;
	if (a[lda*(n-1)+(n-1)] == ZERO) *info = n-1;
}

/*----------------------*/ 

void LP_dgesl(a,lda,n,ipvt,b,job)
int lda,n,ipvt[],job;
REAL a[],b[];

/* We would like to declare a[][lda], but c does not allow it.  In this
function, references to a[i][j] are written a[lda*i+j].  */

/*
     LP_dgesl solves the double precision system
     a * x = b  or  trans(a) * x = b
     using the factors computed by dgeco or LP_dgefa.

     on entry

        a       double precision[n][lda]
                the output from dgeco or LP_dgefa.

        lda     integer
                the leading dimension of the array  a .

        n       integer
                the order of the matrix  a .

        ipvt    integer[n]
                the pivot vector from dgeco or LP_dgefa.

        b       double precision[n]
                the right hand side vector.

        job     integer
                = 0         to solve  a*x = b ,
                = nonzero   to solve  trans(a)*x = b  where
                            trans(a)  is the transpose.

    on return

        b       the solution vector  x .

     error condition

        a division by zero will occur if the input factor contains a
        zero on the diagonal.  technically this indicates singularity
        but it is often caused by improper arguments or improper
        setting of lda .  it will not occur if the subroutines are
        called correctly and if dgeco has set rcond .gt. 0.0
        or LP_dgefa has set info .eq. 0 .

     to compute  inverse(a) * c  where  c  is a matrix
     with  p  columns
           dgeco(a,lda,n,ipvt,rcond,z)
           if (!rcond is too small){
           	for (j=0,j<p,j++)
              		LP_dgesl(a,lda,n,ipvt,c[j][0],0);
	   }

     linpack. this version dated 08/14/78 .
     cleve moler, university of new mexico, argonne national lab.

     functions

     blas LP_daxpy,LP_ddot
*/
{
/*     internal variables	*/

	REAL LP_ddot(),t;
	int k,kb,l,nm1;

	nm1 = n - 1;
	if (job == 0) {

		/* job = 0 , solve  a * x = b
		   first solve  l*y = b    	*/

		if (nm1 >= 1) {
			for (k = 0; k < nm1; k++) {
				l = ipvt[k];
				t = b[l];
				if (l != k){ 
					b[l] = b[k];
					b[k] = t;
				}	
				LP_daxpy(n-(k+1),t,&a[lda*k+k+1],1,&b[k+1],1);
			}
		} 

		/* now solve  u*x = y */

		for (kb = 0; kb < n; kb++) {
		    k = n - (kb + 1);
		    b[k] = b[k]/a[lda*k+k];
		    t = -b[k];
		    LP_daxpy(k,t,&a[lda*k+0],1,&b[0],1);
		}
	}
	else { 

		/* job = nonzero, solve  trans(a) * x = b
		   first solve  trans(u)*y = b 			*/

		for (k = 0; k < n; k++) {
			t = LP_ddot(k,&a[lda*k+0],1,&b[0],1);
			b[k] = (b[k] - t)/a[lda*k+k];
		}

		/* now solve trans(l)*x = y	*/

		if (nm1 >= 1) {
			for (kb = 1; kb < nm1; kb++) {
				k = n - (kb+1);
				b[k] = b[k] + LP_ddot(n-(k+1),&a[lda*k+k+1],1,&b[k+1],1);
				l = ipvt[k];
				if (l != k) {
					t = b[l];
					b[l] = b[k];
					b[k] = t;
				}
			}
		}
	}
}




/**
 * solve the set of linear equations 
 *
 *     AX = B
 *
 * with possibly multiple rhs stored as columns of matrix B
 * the matrix A is not destroyed
 */
void ga_lu_solve_seq(char *trans, Integer *g_a, Integer *g_b) {

  logical oactive;  /* true iff this process participates */
  Integer dimA1, dimA2, typeA;
  Integer dimB1, dimB2, typeB;
  Integer me;
  Integer info;

  /** check environment */
#ifdef GA_USE_VAMPIR
  vampir_begin(GA_LU_SOLVE_SEQ,__FILE__,__LINE__);
#endif
  me     = ga_nodeid_();
  
  /** check GA info for input arrays */
  ga_check_handle(g_a, "ga_lu_solve: a");
  ga_check_handle(g_b, "ga_lu_solve: b");
  ga_inquire(g_a, &typeA, &dimA1, &dimA2);
  ga_inquire(g_b, &typeB, &dimB1, &dimB2);
  
  GA_PUSH_NAME("ga_lu_solve_seq");

  if (dimA1 != dimA2) 
    ga_error("ga_lu_solve: g_a must be square matrix ", 1);
  else if(dimA1 != dimB1) 
    ga_error("ga_lu_solve: dims of A and B do not match ", 1);
  else if(typeA != C_DBL || typeB != C_DBL) 
    ga_error("ga_lu_solve: wrong type(s) of A and/or B ", 1);
  
  ga_sync_();
  oactive = (me == 0);

  if (oactive) {
    DoublePrecision *adra, *adrb, *adri;
    Integer one=1; 

    /** allocate a,b, and work and ipiv arrays */
    adra = (DoublePrecision*) ga_malloc(dimA1*dimA2, C_DBL, "a");
    adrb = (DoublePrecision*) ga_malloc(dimB1*dimB2, C_DBL, "b");
    adri = (DoublePrecision*) ga_malloc(MIN(dimA1,dimA2), C_DBL, "ipiv");

    /** Fill local arrays from global arrays */   
    ga_get_(g_a, &one, &dimA1, &one, &dimA2, adra, &dimA1);
    ga_get_(g_b, &one, &dimB1, &one, &dimB2, adrb, &dimB1);
    
    /** LU factorization */
#ifdef GA_C_CORE
    {  int info_t;
       LP_dgefa(adra, dimA1, dimA2, adri, &info_t);
       info = info_t;
    }
#else
    DGETRF(&dimA1, &dimA2, adra, &dimA1, adri, &info);
#endif

    /** SOLVE */
    if(info == 0) {
#ifdef GA_C_CORE
      DoublePrecision *p_b;
      Integer i;
      int job;
      if(*trans == 't' || *trans == 'T') job = 1; 
      for(i=0; i<dimB2; i++) {
	p_b = adrb + i*dimB1;
	LP_dgesl(adra, dimA1, dimA2, adri, p_b, job);
      }
#else
      DGETRS(cptofcd(trans), &dimA1, &dimB2, adra, &dimA1, 
	     adri, adrb, &dimB1, &info);
#endif

      if(info == 0) 
	ga_put_(g_b, &one, &dimB1, &one, &dimB2, adrb, &dimB1);
      else
	ga_error(" ga_lu_solve: LP_dgesl failed ", -info);
      
    }
    else
      ga_error(" ga_lu_solve: LP_dgefa failed ", -info);
    
    /** deallocate work arrays */
    ga_free(adri);
    ga_free(adrb);
    ga_free(adra);
  }

  ga_sync_();
#ifdef GA_USE_VAMPIR
  vampir_end(GA_LU_SOLVE_SEQ,__FILE__,__LINE__);
#endif
  
  GA_POP_NAME;
}

void FATR ga_lu_solve_seq_(trans, g_a, g_b) 
     Integer *g_a, *g_b;
#if defined(CRAY) || defined(WIN32)
     _fcd    trans;
{ ga_lu_solve_seq(_fcdtocp(trans), g_a, g_b); }
#else
     char* trans;
{ ga_lu_solve_seq(trans, g_a, g_b); }
#endif
