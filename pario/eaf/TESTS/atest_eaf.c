/* #include "file.modes.h" */
#include "elio.h"
#include "eaf.h"


#define IO_TEST_SZ (1024*1024/4)
#define IO_NUM_FILES 4
#define MAX_ITER 4
#define MAP(x) (IO_TEST_SZ*x)


void test(id, l_fnum, l_iter)
    io_request_t id[IO_NUM_FILES][MAX_ITER];
    int l_fnum, l_iter;
{
  int fnum, iter;
  int stat;
  
  for(fnum = 0; fnum <= l_fnum; fnum++)
    for(iter = 0; iter <= l_iter; iter++)
      {
/*	fprintf(stderr, "Probing  id[%d][%d]=%d\n", fnum, iter, id[fnum][iter]); */
	if(EAF_ProbeC(&id[fnum][iter], &stat))
	  {
	    fprintf(stderr, "Bad probe of write fnum=%d   iter %d\n", fnum, iter);
	    exit(1);
	  };
	if(id[fnum][iter] == ELIO_DONE)
	  fprintf(stderr, "Write finished:   fnum=%d  iter=%d  id=%d\n", fnum, iter, id[fnum][iter]);
	else
	  fprintf(stderr, "Pending: fnum=%d, iter=%d  id=%d  stat=%d",fnum, iter, id[fnum][iter], stat);
      };
}


main()
{
  char buf[MAX_ITER][IO_TEST_SZ], buf2[MAX_ITER][IO_TEST_SZ];
  char fname[120];
  int  fnum, sz, iter;
  int  i;
  Fd_t fd[IO_NUM_FILES];
  io_request_t id[IO_NUM_FILES][MAX_ITER];
  int  stat;
  
/*  EAF_Init(); */
  
  for(fnum = 0; fnum < IO_NUM_FILES; fnum++)
    {
#if 1
      sprintf(fname,"/tmp/output.%1d", fnum); 
#else
      sprintf(fname, "/pfs-temp/mogill/output.%1d", fnum);
#endif

      fd[fnum] = EAF_OpenPersistC(fname, ELIO_RW);

      for(iter = 0; iter < MAX_ITER; iter++)
	{
	  for(i = 0; i < IO_TEST_SZ; i++)
	    buf[iter][i] = 'a' + fnum + (iter*5);
	  
	  if((stat=EAF_AWriteC(fd[fnum], MAP(iter), &buf[iter][0], IO_TEST_SZ, &(id[fnum][iter]))) !=0 )
	    {
	      fprintf(stderr, "Bad asynch write of %d bytes, fnum=%d   iter %d    stat=%d\n",
		      IO_TEST_SZ, fnum, iter, stat);
	      exit(1);
	    };
	  fprintf(stderr, "Issued EAF_AWrite  fnum=%d  iter=%d  id=%d\n", fnum, iter, id[fnum][iter]);
#if 1
	  if(fnum%2 == 0 && iter%2 == 0) test(id, fnum, iter);
#else
	  printf("Waiting for EAF_AWrite  fnum=%d  iter=%d  id=%d\n", fnum, iter, id[fnum][iter]);
	  EAF_WaitC(&id[fnum][iter]);
#endif
	};
    };


  
  for(fnum = 0; fnum < IO_NUM_FILES; fnum++)
    {
      for(iter = 0; iter < MAX_ITER; iter++)
	{
	  fprintf(stderr, "Waiting to close  fnum=%d,  iter=%d  id=%d\n", fnum, iter, id[fnum][iter]);
	  EAF_WaitC(&id[fnum][iter]);
	};
      EAF_CloseC(fd[fnum]);
    };

/* . .      . .      . .      . .      . .      . .      . .      . .     */
   
  
  for(fnum = 0; fnum < IO_NUM_FILES; fnum++)
    {
#if 1
      sprintf(fname,"/tmp/output.%1d", fnum); 
#else
      sprintf(fname, "/pfs-temp/mogill/output.%1d", fnum);
#endif

      fd[fnum] = EAF_OpenScratchC(fname, ELIO_RW);

      for(iter = 0; iter < MAX_ITER; iter++)
	{
	  for(i = 0; i < IO_TEST_SZ; i++)
	    buf[iter][i] = 'a' + fnum + (iter*5);

	  if((stat=EAF_AReadC(fd[fnum], MAP(iter), &buf2[iter][0], IO_TEST_SZ, &id[fnum][iter])) != 0)
	    {
	      fprintf(stderr, "Only able to read %d of %d into file %d, iter %d\n",
		      sz, IO_TEST_SZ, fnum, iter);
	      exit(1);
	    };
	  fprintf(stderr, "Issued ARead- ");
#if 1
	  if(fnum%2 == 0 && iter%2 == 0) test(id, fnum, iter);
#else	  
	  do
	    {
	      if( EAF_ProbeC(&id[fnum][iter], &stat) != ELIO_OK)
		ELIO_ABORT("Bad read probe.\n",0);
	      fprintf(stderr,"%d.",id[fnum][iter]);
	      usleep(500000);
	    } while(id[fnum][iter] != ELIO_DONE);
#endif
	  fprintf(stderr, "Finished EAF_ARead()  id=%d  fnum=%d,  iter=%d\n",
		  id[fnum][iter], fnum, iter);
	  
	  if(id[fnum][iter] == ELIO_DONE)
	    for(i = 0; i < IO_TEST_SZ; i++)
	      if(buf[iter][i] != buf2[iter][i])
		{
		  fprintf(stderr, "File %d,  iter %d,  offset %d  -- No read match !%c! !%c!\n",
			  fnum, iter, i, buf[iter][i], buf2[iter][i]);
		  exit(1);
		};
	};
    };
  for(fnum = 0; fnum < IO_NUM_FILES; fnum++)
    EAF_CloseC(fd[fnum]);
  
}



