*
*     Routines to be used on systems where Fortran compiler
*     does a better job than C compiler.
*
*     WARNING: This file must be compiled WITH optimization under AIX.
*              IBM fortran compilers generate bad code with -g option. 
*              UROLL_LOOPS settings are irrelevant to this problem. 
*

#if   defined(IBM) || defined(SP) || defined(LAPI)
#define UROLL_LOOPS 1
#endif

      subroutine d_accumulate(alpha, rows, cols, A, ald, B, bld)
      integer rows, cols
      integer c, r, ald, bld
      double precision A(ald,*), B(bld,*), alpha
#ifdef UROLL_LOOPS 
      integer r1
      doubleprecision d1, d2, d3, d4
      do c = 1, cols
      r1 = iand(max0(rows,0),3)
      do r = 1, r1
         a(r,c) = a(r,c) + alpha*b(r,c)
      end do
      do r = r1 + 1, rows, 4
         d1 = a(r,c) + alpha*b(r,c)
         d2 = a(r+1,c) + alpha*b(r+1,c)
         d3 = a(r+2,c) + alpha*b(r+2,c)
         d4 = a(r+3,c) + alpha*b(r+3,c)
         a(r,c) = d1
         a(r+1,c) = d2
         a(r+2,c) = d3
         a(r+3,c) = d4
      enddo
      enddo
#else
      do c = 1, cols
         do r = 1, rows
            A(r,c) = A(r,c)+ alpha*B(r,c)
         enddo
      enddo
#endif
      end

      subroutine z_accumulate(alpha, rows, cols, A, ald, B, bld)
      integer rows, cols
      integer c, r, ald, bld
      double complex A(ald,*), B(bld,*), alpha
#ifdef UROLL_LOOPS
      integer r1
      complex*16 x1, x2, x3, x4
      do c = 1, cols
      r1 = iand(max0(rows,0),3)
      do r = 1, r1
         a(r,c) = a(r,c) + alpha*b(r,c)
      end do
      do r = r1 + 1, rows, 4
         x1 = a(r,c) + alpha*b(r,c)
         x2 = a(r+1,c) + alpha*b(r+1,c)
         x3 = a(r+2,c) + alpha*b(r+2,c)
         x4 = a(r+3,c) + alpha*b(r+3,c)
         a(r,c) = x1
         a(r+1,c) = x2
         a(r+2,c) = x3
         a(r+3,c) = x4
      enddo
      enddo
#else
      do c = 1, cols
         do r = 1, rows
            A(r,c) = A(r,c)+ alpha*B(r,c)
         enddo
      enddo
#endif
      end


      subroutine i_accumulate(alpha, rows, cols, A, ald, B, bld)
      integer rows, cols
      integer c, r, ald, bld
      integer A(ald,*), B(bld,*), alpha
#ifdef UROLL_LOOPS
      integer r1, j2, j3, j4, j5
      do c = 1, cols
      r1 = iand(max0(rows,0),3)
      do r = 1, r1
         a(r,c) = a(r,c) + alpha*b(r,c)
      end do
      do r = r1 + 1, rows, 4
         j2 = a(r,c) + alpha*b(r,c)
         j3 = a(r+1,c) + alpha*b(r+1,c)
         j4 = a(r+2,c) + alpha*b(r+2,c)
         j5 = a(r+3,c) + alpha*b(r+3,c)
         a(r,c) = j2
         a(r+1,c) = j3
         a(r+2,c) = j4
         a(r+3,c) = j5
      enddo
      enddo
#else
      do c = 1, cols
         do r = 1, rows
            A(r,c) = A(r,c)+ alpha*B(r,c)
         enddo
      enddo
#endif
      end


      subroutine dcopy2d(rows, cols, A, ald, B, bld)
      integer rows, cols
      integer c, r, ald, bld
      double precision A(ald,*), B(bld,*)
#ifdef UROLL_LOOPS
      integer r1
      doubleprecision d1, d2, d3, d4
      do c = 1, cols
      r1 = iand(max0(rows,0),3)
      do r = 1, r1
         b(r,c) = a(r,c)
      end do
      do r = r1 + 1, rows, 4
         d1 = a(r,c)
         d2 = a(r+1,c)
         d3 = a(r+2,c)
         d4 = a(r+3,c)
         b(r,c) = d1
         b(r+1,c) = d2
         b(r+2,c) = d3
         b(r+3,c) = d4
      enddo
      enddo
#else
      do c = 1, cols
         do r = 1, rows
            B(r,c) = A(r,c)
         enddo
      enddo
#endif
      end


      subroutine icopy2d(rows, cols, A, ald, B, bld)
      integer rows, cols
      integer c, r, ald, bld
      integer A(ald,*), B(bld,*)
#ifdef UROLL_LOOPS
      integer r1, j2, j3, j4, j5
      do c = 1, cols
      r1 = iand(max0(rows,0),3)
      do r = 1, r1
         b(r,c) = a(r,c)
      end do
      do r = r1 + 1, rows, 4
         j2 = a(r,c)
         j3 = a(r+1,c)
         j4 = a(r+2,c)
         j5 = a(r+3,c)
         b(r,c) = j2
         b(r+1,c) = j3
         b(r+2,c) = j4
         b(r+3,c) = j5
      enddo
      enddo
#else
      do c = 1, cols
         do r = 1, rows
            B(r,c) = A(r,c)
         enddo
      enddo
#endif
      end

