#include <stdio.h>
#include "tcgmsg.h"
#include "sndrcv.h"

extern void msg_snd(long, char *, long, long);
extern void msg_rcv(long, char *, long, long *, long);

void Ring(void)
{
  long msg[80], in[80];
  long me    = NODEID_();
  long nproc = NNODES_();
  long left  = (me+1) % nproc;
  long right = (me+nproc-1) % nproc;
  long type  = 2;
  long nloop = 10;
  long len;

  if (me == 0) {
    sprintf((char *) msg, "from=%ld to=%ld loop=%ld", me, right, nloop);
    len = strlen(msg)+1;
    msg_snd(type, (char *) msg, len, right);
  }

  while (nloop--) {

    msg_rcv(type, (char *) in, (long) sizeof(in), &len, left);
    printf("me=%ld left=%ld len=%ld msg={%s}\n", me, left, len, (char *) in);
    (void) fflush(stdout);

    if (!((nloop==0) && (me==0))) {
      (void) sprintf((char *) msg, "from=%ld to=%ld loop=%ld", me, right, nloop);
      len = strlen(msg)+1;
      msg_snd(type, (char *) msg, len, right);
    }
  }
}


int main(int argc, char **argv)
{
  PBEGIN_(argc, argv);

  Ring();

  PEND_();

  return 0;
}

