/* #include "file.modes.h" */
#include "elio.h"
#include "eaf.h"


#define IO_TEST_SZ (1024*1024/32)
#define IO_NUM_FILES 4
#define MAX_ITER 4
#define MAP(x) (IO_TEST_SZ*x)

main()
{
  char buf[IO_TEST_SZ], buf2[IO_TEST_SZ];
  char fname[120];
  int  fnum, sz, iter;
  int  i;
  Fd_t *fd[IO_NUM_FILES];
  
/*  EAF_Init(); */
  
  for(fnum = 0; fnum < IO_NUM_FILES; fnum++)
    {
#if 1
      sprintf(fname,"/tmp/output.%1d", fnum); 
#else
      sprintf(fname, "/piofs/mogill/output.%1d", fnum);
#endif

      fd[fnum] = EAF_OpenPersistC(fname, ELIO_RW);

      for(iter = 0; iter < MAX_ITER; iter++)
	{
	  for(i = 0; i < IO_TEST_SZ; i++)
	    buf[i] = 'a' + fnum + (iter*5);
	  
	  if((sz = EAF_WriteC(fd[fnum], MAP(iter), buf, IO_TEST_SZ)) != IO_TEST_SZ)
	    {
	      fprintf(stderr, "Only able to write %d of %d into file %d, iter %d\n",
		      sz, IO_TEST_SZ, fnum, iter);
	      exit(1);
	    };
	};
    };
  for(fnum = 0; fnum < IO_NUM_FILES; fnum++)
    {
      sprintf(fname,"ouput.%1d", fnum);
      EAF_CloseC(fd[fnum]);
    };

    
   
  
  for(fnum = 0; fnum < IO_NUM_FILES; fnum++)
    {
#if 1
      sprintf(fname,"/tmp/output.%1d", fnum); 
#else
      sprintf(fname, "/piofs/mogill/output.%1d", fnum);
#endif

      fd[fnum] = EAF_OpenScratchC(fname, ELIO_RW);

      for(iter = 0; iter < MAX_ITER; iter++)
	{
	  for(i = 0; i < IO_TEST_SZ; i++)
	    buf[i] = 'a' + fnum + (iter*5);
	  
	  if((sz = EAF_ReadC(fd[fnum], MAP(iter), buf2, IO_TEST_SZ)) != IO_TEST_SZ)
	    {
	      fprintf(stderr, "Only able to read %d of %d into file %d, iter %d\n",
		      sz, IO_TEST_SZ, fnum, iter);
	      exit(1);
	    };
	  
	  for(i = 0; i < IO_TEST_SZ; i++)
	    if(buf[i] != buf2[i])
	      {
		fprintf(stderr, "File %d,  iter %d,  offset %d  -- No read match !%c! !%c!\n",
			fnum, iter, i, buf[i], buf2[i]);
		exit(1);
	      };
	};
    };
  for(fnum = 0; fnum < IO_NUM_FILES; fnum++)
    EAF_CloseC(fd[fnum]);

}

