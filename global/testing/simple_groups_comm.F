#if HAVE_CONFIG_H
#   include "config.fh"
#endif
      program main
      implicit none
#include "mpif.h"
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
      integer proc_group(0:100),proclist(100),inode,nprocs
      integer ierr,comm,sbuf(1),rbuf(1)
      call ga_initialize()
      nproc = ga_nnodes()
      me = ga_nodeid()

      status = ma_init(MT_DBL, 100000, 100000) 
      if (.not. status) call ga_error('ma_init failed',-1)
      status = ma_set_auto_verify(.true.)
      status = ma_set_hard_fail(.true.)
      status = ma_set_error_print(.true.)
      
      inode = ga_cluster_nodeid()
      if (me.eq.0) then
          write(0,'(A,I4,A,I4,A)') 'there are ', ga_cluster_nnodes(),
     ,          ' nodes, node 0 has ', ga_cluster_nprocs(0), ' procs'
          call ffflush(6)
      endif
      call ga_sync()
      do i=0,ga_cluster_nnodes()-1
         do j=0,ga_cluster_nprocs(i)-1
            proclist(j+1)=ga_cluster_procid(i,j)
         enddo
         proc_group(i)=ga_pgroup_create(proclist,ga_cluster_nprocs(i))
      enddo
      call ga_sync()
      do i = 0, ga_cluster_nnodes()-1
        if (i.eq.inode) then
          write(0,'(I4,A,I4)') me,' joining group', proc_group(inode)
          call ga_pgroup_set_default(proc_group(inode))
          status = ga_create(MT_DBL, n, n, 'a', 1, 1, g_a) 
          if (.not. status) call ga_error('ga_create failed',-1)
          write(0,'(I4,A,I4,A,I4)') me,' Created array of  group ',
     ,         proc_group(inode), ' as proc no. ',ga_nodeid()
          call ga_print_distribution(g_a) 
          call ga_mpi_comm_pgroup_default(comm)
          if (comm.ne.mpi_comm_null) then
            sbuf(1) = ga_nodeid()
            call mpi_allreduce(sbuf, rbuf, 1, mpi_integer,
     ,                         mpi_max, comm, ierr)
            write(0,'(I4,A,I4)') me, ' max nodeid is ', rbuf(1)
            if ((rbuf(1)+1).ne.ga_cluster_nprocs(i)) then
              call ga_error('mpi_allreduce failed',1)
            endif
          else
            write(0,'(A)') 'mpi_comm was null!'
          endif
          call ga_pgroup_set_default(ga_pgroup_get_world())
        endif
        call ffflush(6)
        call ga_sync()
      enddo

      call ga_terminate()

      call mpi_finalize(ierr)

      end
