      program main
      implicit none
#include "global.fh"
#include "mafdecls.fh"
      integer nproc,me 
      integer g_a,g_b,g_c
      logical status 
      integer n,i,j
      parameter (n=4)
      integer lo(2),hi(2)
      integer dims(2)
      double precision buf(n,n)
      integer proc_group(0:100),proclist(100),inode,nprocs,nnodes

      call pbeginf

      call ga_initialize()
      nproc = ga_nnodes()
      me = ga_nodeid()

      status = ma_init(MT_DBL, 100000, 100000) 
      if (.not. status) call ga_error('ma_init failed',-1)
      
      inode = ga_cluster_nodeid()
      nnodes = ga_cluster_nprocs(inode)

      do i = 0, nnodes-1
        nprocs = ga_cluster_nprocs(i)
c       if (i.eq.inode) then
        do j = 0, nprocs - 1
          proclist(j+1) = ga_cluster_procid(i,j)
        end do
        proc_group(i) = ga_pgroup_create(proclist, nprocs)
        write(6,*) me,'Created groups'
        call ga_sync()
      enddo
      do i = 0, nnodes-1
        call ga_sync()
        if (i.eq.inode) then
          call ga_pgroup_set_default(proc_group(inode))
          status = ga_create(MT_DBL, n, n, 'a', 1, 1, g_a) 
          if (.not. status) call ga_error('ga_create failed',-1)
          write(6,*) me,'Created array'
          call ga_print_distribution(g_a) 
        endif
        call ga_sync()
      enddo

      call ga_terminate()

      call pend

      end
