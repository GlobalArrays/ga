#include <mpi.h>

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>

/* number of processors in a progress rank block. The last processor in the block
 * is the progress rank. This block must evenly divide the number of processors
 * on an SMP node */
#define NPROC_BLOCK 7
#define SEGMENT_SIZE 4194304
#define NLOOP 10000

#define TEST_TAG 27624

int main(int argc, char **argv)
{
  int rank, nprocs, nvprocs;
  int vrank;
  int ierr;
  int iloop;
  int i;
  MPI_Comm comm = MPI_COMM_WORLD;
  long *tmpids, *hostids;
  long my_hostid;
  int smp_size;
  int *pr_world;
  char *name, *names;
  void **ptrs;
  void *my_shm_buf;
  int *my_loc_buf;
  int dest, vdest;
  MPI_Status status;
  MPI_Request request;
  int loopinc = NLOOP/100;
  int tok, ok;
  int nints, nproc_blocks, my_proc_block;
  int *offset;

  MPI_Init(&argc, &argv);
  ierr = MPI_Comm_size(comm,&nprocs);
  ierr = MPI_Comm_rank(comm,&rank);
  if (nprocs%NPROC_BLOCK != 0) { 
    printf("p[%d] Number of processors not a multiple of block size");
    MPI_Abort(comm,-1);
  }
  nvprocs = (nprocs*(NPROC_BLOCK-1))/NPROC_BLOCK;
  vrank = ((rank-rank%NPROC_BLOCK)/NPROC_BLOCK)*(NPROC_BLOCK-1)+rank%NPROC_BLOCK;
  if (rank == 0) {
    printf("Number of processors in test: %d\n",nprocs);
    printf("Number of visible processors in test: %d\n",nvprocs);
    printf("Size of process block: %d\n",NPROC_BLOCK);
  }
  nproc_blocks = nprocs/NPROC_BLOCK;
  my_proc_block = (rank-rank%NPROC_BLOCK)/NPROC_BLOCK;

  /* Find host IDs */
  tmpids = (long*)malloc(nprocs*sizeof(long));
  hostids = (long*)malloc(nprocs*sizeof(long));
  for (i=0; i<nprocs; i++) tmpids[i] = (long)0;
  my_hostid = gethostid();
  tmpids[rank] = my_hostid;
  ierr = MPI_Allreduce(tmpids,hostids,nprocs,MPI_LONG,MPI_SUM,comm);
  free(tmpids);
  /* Find out how many processors have my host ID */
  smp_size = 0;
  for (i=0; i<nprocs; i++) {
    if (my_hostid == hostids[i]) smp_size++;
  }
  if (smp_size%NPROC_BLOCK != 0) {
    printf("p[%d] SMP size %d is not evenly divided by NPROC_BLOCK %d\n",
        rank,smp_size,NPROC_BLOCK);
    MPI_Abort(comm,-1);
  }
  if (rank == 0) {
    printf("Number of progress ranks per SMP node: %d\n",smp_size/NPROC_BLOCK);
  }
  /* Find progress ranks for all processors */
  pr_world = (int*)malloc(nprocs*sizeof(int));
  for (i=0; i<nprocs; i++) {
    if ((i+1)%NPROC_BLOCK != 0) {
      pr_world[i] = i-i%NPROC_BLOCK+NPROC_BLOCK-1;
    } else {
      pr_world[i] = -1;
    }
  }

  /* set up offsets for each rank within a processor block */
  offset = (int*)malloc((NPROC_BLOCK-1)*sizeof(int));
  iloop = 0;
  i=0;
  while (iloop < NPROC_BLOCK-1) {
    /* find offset that will send data to another SMP node */
    int nblk_per_node = smp_size/NPROC_BLOCK;
    if ((i*nblk_per_node)%nproc_blocks != 0) {
      offset[iloop] = (i*nblk_per_node*NPROC_BLOCK)%nprocs;
      iloop++;
    }
    i++;
  }
  /* create unique name for shared memory segment (this is not used on
   * progress ranks) */
  name = (char*)malloc(31*sizeof(char));
  snprintf(name,31,"/mpi%010u%010u%06u",getuid(),getpid(),rank);
  /* share name with all processors */
  names = (char*)malloc(31*nprocs*sizeof(char));
  memcpy(&names[rank*31],name,31);
  ierr = MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, names,
      31, MPI_BYTE, comm);
  /* allocate shared memory segment on all non-pr processors */
  ptrs = (void**)malloc(nprocs*sizeof(void*));
  if (pr_world[rank] != -1) {
    int fd = 0;
    int retval = 0;
    /* Create shared memory segment */
    fd = shm_open(name,O_CREAT|O_EXCL|O_RDWR, S_IRUSR|S_IWUSR);
    if (-1 == fd && EEXIST == errno) {
      retval = shm_unlink(name);
      if (-1 == retval) {
        perror("shm_open: shm_unlink failed");
        MPI_Abort(comm,-1);
      }
    }
    /* try a second time */
    if (-1 == fd) {
      fd = shm_open(name,O_CREAT|O_EXCL|O_RDWR, S_IRUSR|S_IWUSR);
    }
    /* still unsuccessful. Report erro */
    if (-1 == fd) {
      if (errno == EMFILE) {
        printf("The per process limit on the number of open file"
            " descriptors has been reached\n");
      } else if ( errno == ENFILE) {
        printf("The system-wide limit on the total number of open files"
            " has been reached\n");
      }
      perror("shm_open: shm_unlink failed");
      MPI_Abort(comm,-1);
    }
    retval = ftruncate(fd,SEGMENT_SIZE);
    if (-1 == retval) {
      if (errno == EFAULT) {
        printf("File descriptor points outside the processes allocated"
            " address space\n");
      }
      perror("ftruncate failed");
      MPI_Abort(comm,-1);
    }
    my_shm_buf = mmap(NULL,SEGMENT_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (MAP_FAILED == my_shm_buf) {
      if (errno == EBADF) {
        printf("File descriptor used in mmap is bad\n");
      } else if (errno == ENFILE) {
        printf("The system-wid limit on the total number of open files"
            " has been reached\n");
      } else if (errno == ENODEV) {
        printf("The system does not support memory mapping\n");
      } else if (errno == ENOMEM) {
        printf("The processes maximum number of mappings has been exceeded\n");
      }
      perror("mmap failed");
      MPI_Abort(comm,-1);
    }
    /* close file descriptor */
    retval = close(fd);
    if (-1 == retval) {
      perror("close for shared memory failed");
      MPI_Abort(comm,-1);
    }
    /* call barrier so progress ranks can't access memory before it exists */
    MPI_Barrier(comm);
  } else {
    int fd = 0;
    int retval = 0;
    /* wait for all shared memory segments to be created */
    MPI_Barrier(comm);
    /* Open shared memory segments to all processors managed by this progress
     * rank */
    for (i=rank-1; i>=rank-NPROC_BLOCK+1;i--) {
      fd = shm_open(&names[i*31], O_RDWR, S_IRUSR|S_IWUSR);
      if (-1 == fd) {
        if (errno == EMFILE) {
          printf("The per process limit on the number of open file"
              " descriptors has been reached\n");
        } else if (errno == ENFILE) {
          printf("The system-wide limit on the total number of open files"
              " has been reached\n");
        }
        printf("p[%d] Failed to open (%s)\n",rank,&names[i*31]);
        perror("shm_attach: shm_open failed");
        MPI_Abort(comm,-1);
      }
      ptrs[i] = mmap(NULL,SEGMENT_SIZE,PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
      if (MAP_FAILED == ptrs[i]) {
        if (errno == EBADF) {
          printf("File descriptor used in mmap is bad\n");
        } else if (errno == ENFILE) {
          printf("The system-wide limit on the total number of open files"
              " has been reached\n");
        } else if (errno == ENODEV) {
          printf("The system does not support memory mapping\n");
        } else if (errno == ENOMEM) {
          printf("The processes maximum number of mappings has been exceeded\n");
        }
        perror("shm_attach: mmap failed");
        MPI_Abort(comm,-1);
      }
      retval = close(fd);
      if (-1 == retval) {
        perror("shm_attach: close failed");
        MPI_Abort(comm,-1);
      }
    }
  }

  my_loc_buf = (int*)malloc(SEGMENT_SIZE);
  for (iloop=0; iloop < NLOOP; iloop++) {
    if (iloop%loopinc == 0 && rank == 0) {
      printf("Evaluating loop %d\n",iloop);
    }
    /* Initialize data and send it to progress rank (if not a progress rank) */
    if (pr_world[rank] != -1) {
      int numi = SEGMENT_SIZE/sizeof(int);
      int inc;
      int dest_pr;
      MPI_Request request_d, request_h;
      int ierr;
      /* header contains 3 entries: source rank, destination rank, message length */
      int header[3];
      /* find destination process */
      inc = rank%NPROC_BLOCK;
      dest = (rank+offset[inc])%nprocs;
      dest_pr = pr_world[dest];
      vdest = ((dest-inc)/NPROC_BLOCK)*(NPROC_BLOCK-1)+inc;
      if (iloop == 0) {
        printf("Process %d sending to process %d\n",rank,dest_pr);
      }
      header[0] = rank;
      header[1] = dest;
      header[2] = SEGMENT_SIZE;
      for (i=0; i<numi; i++) {
        my_loc_buf[i] = i+vdest*numi;
      }
      /* Send header */
      MPI_Isend(header,3,MPI_INT,pr_world[dest],TEST_TAG,comm,&request_h);
      /* Send data payload */
      MPI_Isend(my_loc_buf,SEGMENT_SIZE,MPI_CHAR,pr_world[dest],TEST_TAG,
          comm,&request_d);
      MPI_Wait(&request_h, &status);
      MPI_Wait(&request_d, &status);
    } else {
      int loop_cnt = 0;
      /* expecting NPROC_BLOCK-1 messages from other processors */
      while (loop_cnt < NPROC_BLOCK-1) {
        int header[3];
        int src, dest, nsize;
        int retval;
        int recv_count = 0;
        retval = MPI_Recv(header,3,MPI_INT,MPI_ANY_SOURCE,TEST_TAG,comm,&status);
        src = header[0];
        dest = header[1];
        nsize = header[2];
        retval = MPI_Recv(ptrs[dest], nsize, MPI_CHAR, src, TEST_TAG, comm, &status);
        retval = MPI_Get_count(&status, MPI_CHAR, &recv_count);
        if (recv_count != nsize) {
          printf("recv_count not equal to nsize\n");
          MPI_Abort(comm,-1);
        }
        loop_cnt++;
      }
    }
  }
  MPI_Barrier(comm);
  /* Check data in shared buffers to see if it is correct */
  nints = SEGMENT_SIZE/sizeof(int);
  tok = 1;
  if (pr_world[rank] != -1) {
    for (i=0; i<nints; i++) {
      if (((int*)my_shm_buf)[i]  != i+vrank*nints) {
        tok = 0;
      }
    }
  }
  ierr = MPI_Allreduce(&tok,&ok,1,MPI_INT,MPI_SUM,comm);
  if (rank == 0) {
    if (ok) {
      printf("SUCCESS: Correct data found in destination\n");
    } else {
      printf("FAILURE: Incorrect data found in destination\n");
    }
  }

  /* Clean up shared memory segments */
  if (pr_world[rank] != -1) {
    int retval = 0;
    retval = munmap(my_shm_buf, SEGMENT_SIZE);
    if (-1 == retval) {
      perror("munmap fails");
      MPI_Abort(comm,-1);
    }
    retval = shm_unlink(&names[rank*31]);
    if (-1 == retval) {
      perror("shm_unlink fails");
      MPI_Abort(comm,-1);
    }
  } else {
    for (i=rank-1; i>=rank-NPROC_BLOCK+1;i--) {
      int retval = 0;
      retval = munmap(ptrs[i], SEGMENT_SIZE);
      if (-1 == retval) {
        perror("munmap fails");
        MPI_Abort(comm,-1);
      }
    }
  }
  free(my_loc_buf);
  free(ptrs);
  free(name);
  free(names);
  free(hostids);
  free(pr_world);
   
  MPI_Finalize();
  return 0;
}
