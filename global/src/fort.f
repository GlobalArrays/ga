      double complex function ga_zdot(g_a,g_b)
      implicit none
      integer g_a, g_b
      ga_zdot = (0.,0.)
      call gai_zdot(g_a,g_b,ga_zdot)
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

