/*$Id: ga_handler.c,v 1.4 1995-02-02 23:13:17 d3g681 Exp $*/
#if defined(NX) || defined(SP1)

#include "global.h"
#include "globalp.h"
#include "message.h"
#include "interrupt.h"
#include <stdio.h>


long htype = GA_TYPE_REQ;
void ga_init_handler(char*, long);


#ifdef NX 

void ga_handler(long type, long count, long node, long pid)
{
  long oldmask;
  ga_mask(1L, &oldmask);

  ga_SERVER(node);

  ga_init_handler((char*) MessageRcv, (long)TOT_MSG_SIZE );

  ga_mask(oldmask, &oldmask);
}


void ga_init_handler(char *buffer, long lenbuf) /*Also called in ga_initialize*/
{
  hrecv(htype, buffer, lenbuf, ga_handler); 
}



#elif defined(SP1)
/******************** SP interrupt receive stuff *************/

static long  requesting_node;
static long  msgid; 
static long  have_wildcards=0; 
static long  dontcare, allmsg, nulltask,allgrp; /*values of MPL/EUIH wildcards*/


/*\ gets values of EUI wildcards
\*/
void wildcards()
{
long buf[4], qtype, nelem, status;
     qtype = 3; nelem = 4;
     status = mpc_task_query(buf,nelem,qtype);
     if(status==-1) ga_error("wildcards: mpc_task_query error", -1L);

     dontcare = buf[0];
     allmsg   = buf[1];
     nulltask = buf[2];
     allgrp   = buf[3];
     have_wildcards=1; 
}



static void ga_handler(long *pid)
{
long msglen;

  mpc_wait(pid, &msglen);

  /* fprintf(stderr,"in handler: msg from %d\n",requesting_node); */
  ga_SERVER(requesting_node);
  ga_init_handler(MessageRcv, TOT_MSG_SIZE );
  /* fprintf(stderr,"leaving handler\n"); */
}




void ga_init_handler(buffer, lenbuf)   /* Also called in ga_initialize */
char *buffer;
long lenbuf;
{
static long status; 

  if( ! have_wildcards) wildcards();

  requesting_node = dontcare;

  status=mp_rcvncall(buffer, &lenbuf, &requesting_node,
                     &htype, &msgid, ga_handler);
}


void fake_work_() /* something to call while syncing */
{}

#endif


#else

This file should only be linked in under NX or EUIH/MPL

#endif
 
