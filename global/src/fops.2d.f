*
*     Routines to be used on systems where Fortran compiler
*     does a better job than C compiler.
*
      subroutine d_accumulate(alpha, rows, cols, A, ald, B, bld)
      integer rows, cols
      integer c, r, ald, bld
      double precision A(ald,*), B(bld,*), alpha
      do c = 1, cols
         do r = 1, rows
            A(r,c) = A(r,c)+ alpha*B(r,c)
         enddo
      enddo
      end

      subroutine z_accumulate(alpha, rows, cols, A, ald, B, bld)
      integer rows, cols
      integer c, r, ald, bld
      double complex A(ald,*), B(bld,*), alpha
      do c = 1, cols
         do r = 1, rows
            A(r,c) = A(r,c)+ alpha*B(r,c)
         enddo
      enddo
      end


      subroutine i_accumulate(alpha, rows, cols, A, ald, B, bld)
      integer rows, cols
      integer c, r, ald, bld
      integer A(ald,*), B(bld,*), alpha
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

      double complex function ga_zdot(g_a,g_b)
      implicit none
      integer g_a, g_b
      external gai_dot
      ga_zdot = (0.,0.)
      call gai_dot(g_a,g_b,ga_zdot)
      end


      double complex function ga_zdot_patch(
     $            g_a, t_a, ailo, aihi, ajlo, ajhi,
     $            g_b, t_b, bilo, bihi, bjlo, bjhi)
      implicit none
      integer  g_a, g_b, ailo, aihi, ajlo, ajhi
      integer  bilo, bihi, bjlo, bjhi
      character*1 t_a, t_b
      external gai_dot_patch
      ga_zdot_patch = (0.,0.)
      call gai_dot_patch(
     $            g_a, t_a, ailo, aihi, ajlo, ajhi,
     $            g_b, t_b, bilo, bihi, bjlo, bjhi, ga_zdot_patch)
      end


      double precision function ga_ddot_patch(
     $            g_a, t_a, ailo, aihi, ajlo, ajhi,
     $            g_b, t_b, bilo, bihi, bjlo, bjhi)
      implicit none
      integer  g_a, g_b, ailo, aihi, ajlo, ajhi
      integer  bilo, bihi, bjlo, bjhi
      character*1 t_a, t_b
      external gai_dot_patch
      ga_ddot_patch = 0.
      call gai_dot_patch(
     $            g_a, t_a, ailo, aihi, ajlo, ajhi,
     $            g_b, t_b, bilo, bihi, bjlo, bjhi, ga_ddot_patch)
      end


      subroutine ga_dadd(alpha, g_a, beta, g_b, g_c)
      integer g_a, g_b, g_c
      double precision alpha, beta
      external ga_add 
      call ga_add(alpha, g_a, beta, g_b, g_c)
      end

      

      subroutine ga_dadd_patch(alpha, g_a, ailo, aihi, ajlo, ajhi,
     $                  beta,  g_b, bilo, bihi, bjlo, bjhi,
     $                         g_c, cilo, cihi, cjlo, cjhi)

      integer g_a, g_b, g_c           
      double precision alpha, beta   
      integer  ailo, aihi, ajlo, ajhi
      integer  bilo, bihi, bjlo, bjhi
      integer  cilo, cihi, cjlo, cjhi
      external ga_add_patch 
      call ga_add_patch(alpha, g_a, ailo, aihi, ajlo, ajhi,
     $                  beta,  g_b, bilo, bihi, bjlo, bjhi, 
     $                         g_c, cilo, cihi, cjlo, cjhi)
      end

      subroutine ga_dscal(g_a, s)
      integer g_a                      
      double precision s                
      external ga_scale
      call  ga_scale(g_a, s)
      end

      subroutine ga_dscal_patch(g_a, ailo, aihi, ajlo, ajhi, s)
      integer g_a  
      double precision s
      integer  ailo, aihi, ajlo, ajhi
      external ga_scale_patch
      call ga_scale_patch(g_a, ailo, aihi, ajlo, ajhi, s)
      end

      subroutine ga_dfill_patch(g_a, ailo, aihi, ajlo, ajhi, s)
      integer g_a
      double precision s 
      integer  ailo, aihi, ajlo, ajhi
      external ga_fill_patch
      call ga_fill_patch(g_a, ailo, aihi, ajlo, ajhi, s)
      end

      subroutine ga_ifill_patch(g_a, ailo, aihi, ajlo, ajhi, s)
      integer g_a
      integer s 
      integer  ailo, aihi, ajlo, ajhi
      external ga_fill_patch
      call ga_fill_patch(g_a, ailo, aihi, ajlo, ajhi, s)
      end

