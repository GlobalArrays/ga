      subroutine dcopy2d_n(rows, cols, A, ald, B, bld)
      integer rows, cols
      integer c, r, ald, bld
      double precision A(ald,*), B(bld,*)

      do c = 1, cols
         do r = 1, rows
            B(r,c) = A(r,c)
         end do
      end do
      end

      

      subroutine dcopy2d_u(rows, cols, A, ald, B, bld)
      integer rows, cols
      integer c, r, ald, bld
      double precision A(ald,*), B(bld,*)
      integer r1
      double precision d1, d2, d3, d4
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
      end
