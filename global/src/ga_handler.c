#if defined(NX) || defined(SP1)
extern void ga_server_( long *);	/* FORTRAN routine */

#define GA_TYPE_REQ 32759L	/* MUST match globalp.h */

void ga_init_handler_(double*, long*);



#ifdef NX 
extern long masktrap(long);

void ga_handler(long type, long count, long node, long pid)
{
  long oldmask;
  oldmask = masktrap(1);

  ga_server_(&node);

  ga_reinit_handler_();

  oldmask = masktrap(oldmask);
}

static long  htype=GA_TYPE_REQ;

void ga_init_handler_(buffer, lenbuf)         /* Also called from FORTRAN */
double *buffer;
long   *lenbuf;
{
  hrecv(htype, (char *) buffer, *lenbuf, ga_handler);
}



#elif defined(SP1)


/******************** SP1 interrupt receive stuff *************/

extern long   lockrnc(long*,long *);
extern long   rcvncall(char*,long *, long*,long*,long*,void*());
extern long   mpc_wait(long*,long *);
static long   requesting_node;
static long dontcare, allmsg, nulltask,allgrp; /*values for EUI/EUIH wildcards*/
#include <stdio.h>

/*\ gets values of EUI wildcards
\*/
void wildcards()
{
long buf[4], qtype, nelem, status;
        qtype = 3; nelem = 4;
        status = mpc_task_query(buf,nelem,qtype);
        if(status==-1) Error("wildcards: mpc_task_query error", -1L);

        dontcare = buf[0];
        allmsg   = buf[1];
        nulltask = buf[2];
        allgrp   = buf[3];
}



static void ga_handler(long *pid)
{
long msglen;

  mpc_wait(pid, &msglen);

  /* fprintf(stderr,"in handler: msg from %d\n",requesting_node); */
  ga_server_(&requesting_node);
  ga_reinit_handler_();  /* ask fortran to reinitialize handler */ 
                         /* -- we don't know the buffer address */
  /* fprintf(stderr,"leaving handler\n"); */
}



static long  htype=GA_TYPE_REQ, msgid; 

void ga_init_handler_(buffer, lenbuf)         /* Also called from FORTRAN */
double *buffer;
long   *lenbuf;
{
static long status; 

  wildcards();

  requesting_node = dontcare;

  status=rcvncall(buffer, lenbuf, &requesting_node,
                     &htype, &msgid, ga_handler);
}


void fake_work_() /* something to call while syncing */
{}

#endif


#else

This file should only be linked in under NX or EUIH (SP1)

#endif
 
