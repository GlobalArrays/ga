*
*     Routines to be used on systems where Fortran compiler
*     does a better job than C compiler.
*
      subroutine accumulatef(alpha, rows, cols, A, ald, B, bld)
      integer rows, cols
      integer c, r, ald, bld
      double precision A(ald,*), B(bld,*), alpha
      do c = 1, cols
         do r = 1, rows
            A(r,c) = A(r,c)+ alpha*B(r,c)
         enddo
      enddo
      end


      subroutine dcopy2d(rows, cols, A, ald, B, bld)
      integer rows, cols
      integer c, r, ald, bld
      double precision A(ald,*), B(bld,*)
      do c = 1, cols
         do r = 1, rows
            B(r,c) = A(r,c)
         enddo
      enddo
      end


      subroutine icopy2d(rows, cols, A, ald, B, bld)
      integer rows, cols
      integer c, r, ald, bld
      integer A(ald,*), B(bld,*)
      do c = 1, cols
         do r = 1, rows
            B(r,c) = A(r,c)
         enddo
      enddo
      end

