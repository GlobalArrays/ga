#include "global.h"
#include "globalp.h"
#include "message.h"
#include "macommon.h"
#include <stdio.h>

#define MAX_LOCKS 32768
#define SPINMAX 1000
#define DEBUG 0

double _dummy_work_=0.;
static Integer g_mutexes;
static Integer num_mutexes=0, *next; 
static Integer one=1, two=2;
static Integer Whandle, Nhandle;

typedef struct {
       Integer lock;
       Integer turn;
} waiting_list_t;

static waiting_list_t* blocked;


#define ERROR(str,code) ga_error(str, (Integer)code)

#if defined(NX) || defined(SP1) || defined(SP)
#   define SERVER_LOCK
#   define ASYN_COMMS
#endif



logical ga_create_mutexes_(Integer *num)
{
Integer type=MT_F_INT, nproc = ga_nnodes_(), indx;

	if (*num <= 0 || *num > 32768) return(FALSE);
        if(num_mutexes) ERROR("mutexes already created",num_mutexes);

        num_mutexes= (int)*num;

        if(nproc == 1) return(TRUE);

        if(!ga_create(&type, num, &two, "GA mutexes", &one,&two, &g_mutexes))
                                                                return(FALSE);

#       if defined(SERVER_LOCK)
          if(!MA_alloc_get(MT_F_INT,2*nproc,"GA lock wait list",&Whandle,&indx))
                 ERROR("ga_create_mutexes:error allocating memory for lock",0); 
          MA_get_pointer(Whandle, &blocked);

#       endif
        /* one extra element to make indexing consistent with GA */
        if(!MA_alloc_get(MT_F_INT, *num+1, "GA lock next", &Nhandle,&indx))
                 ERROR("ga_create_mutexes:error allocating memory for lock",1);
        MA_get_pointer(Nhandle, &next);

        ga_zero_(&g_mutexes);

        return(TRUE);
}
        

logical ga_destroy_mutexes_()
{
     if(num_mutexes==0)ERROR("ga_destroy_mutexes: mutexes not created",0);
     num_mutexes=0;
     if(ga_nnodes_() == 1) return(TRUE);

     MA_free_heap(Nhandle);
#    if defined(SERVER_LOCK)
        if(blocked)   MA_free_heap(Whandle); 
#    endif

     return(ga_destroy_(&g_mutexes));
}


static void ga_generic_lock(Integer id)
{
Integer i, myturn, factor=0;

        /* register in the queue */
        next[id] = ga_read_inc_(&g_mutexes,&id,&one, &one);

        _dummy_work_ = 0.;
        do {

           ga_get_(&g_mutexes, &id, &id, &two, &two, &myturn, &one);
          
           /* linear backoff before retrying  */
           for(i=0; i<  SPINMAX * factor; i++) _dummy_work_ += 1.;

           factor += 1;

        }while (myturn != next[id]);
}


static void ga_generic_unlock(Integer id)
{
       next[id]++;
       ga_put_(&g_mutexes, &id, &id, &two, &two, next+id, &one); 
}


/*\  Acquire mutex "id" for "node"
 *   -must be executed in hrecv/AM handler thread
 *   -application thread must use generic_lock routine
\*/
void ga_server_lock(Integer handle, Integer id, Integer node)
{
Integer myturn, turn, me = ga_nodeid_();
        

        myturn = ga_read_inc_local(handle, id, one, one, me);
        ga_get_local(handle, id, id, two, two, &turn, 0, 1, me);

        if(turn != myturn){
           blocked[node].lock = id;
           blocked[node].turn = myturn;
           if(DEBUG) fprintf(stderr,"server=%d node=%d wait for lock (%d)\n",
                                                         me, node, myturn);
        } else {
           /* send ticket to requesting node */
           ga_msg_snd(GA_TYPE_LCK, &myturn, sizeof(Integer), node);  
           if(DEBUG)fprintf(stderr,"server=%d node=%d got lock\n",me, node);
        } 
}
           
        
/*\  Release mutex "id" held by "node"
 *   called from hrecv/AM handler AND application thread
\*/
void ga_server_unlock(Integer handle, Integer id, Integer node, Integer server_next)
{
Integer i, proc=-1, me = ga_nodeid_();
int ack;

        server_next++;
        ga_put_local(handle, id, id, two, two, &server_next, 0, one, me);
        
        /* search for the node next in queue for this lock */
        for(i=0; i< ga_nnodes_(); i++){
           if(DEBUG)fprintf(stderr,"server=%d node=%d list=(%d,%d)\n",
                             me, i, blocked[i].lock, blocked[i].turn);
           if((blocked[i].lock == id) && (blocked[i].turn == server_next)){
              proc = i;
              break;
           }
        }
        if(proc != -1)
           if(proc ==me)ERROR("server_unlock: cannot unlock self",0);
           else ga_msg_snd(GA_TYPE_LCK, &server_next, sizeof(Integer), proc);  

        if(DEBUG)fprintf(stderr,"server=%d node=%d unlock next=%d go=%d\n",
                                             me, node, server_next, proc);
}
            
     
/*\ lock for machines with interrupt-driven comms
 *  the algorithm similar to generic one, however remote nodes do not spin wait
\*/
static void dst_lock(Integer id)
{
Integer owner, owner2, server;

      /* check if elements (id,1) and (id,2) reside on the same process */ 
      if(!ga_locate_(&g_mutexes, &id, &one, &owner))
                     ERROR("ga_lock:locate fail",1);
      if(!ga_locate_(&g_mutexes, &id, &two, &owner2))
                     ERROR("ga_lock:locate fail",2);
      if(owner != owner2) ERROR("ga_lock:error in mutex array distribution",0);

      server = DataServer(owner);

      if(server == ga_nodeid_()){
         ga_generic_lock(id);
      }else{
         Integer len, exp_len=sizeof(Integer), from;

#        if defined(SERVER_LOCK) && defined(ASYN_COMMS)
            msgid_t msgid;
            msgid = ga_msg_ircv(GA_TYPE_LCK,  next+id, exp_len, server);
            ga_snd_req(g_mutexes, id,0,0,0, 0, 0, GA_OP_LCK, owner, server);
            ga_msg_wait(msgid, &from, &len);
#        else
            ga_snd_req(g_mutexes, id,0,0,0, 0, 0, GA_OP_LCK, owner, server);
            ga_msg_rcv(GA_TYPE_LCK, next+id, exp_len, &len, server, &from);
#        endif
      }
}



/*\ unlock for machines with interrupt-driven comms
\*/
static void dst_unlock(Integer id)
{
Integer owner, server;

      if(!ga_locate_(&g_mutexes, &id, &one, &owner)) 
                                  ERROR("ga_lock:locate failed",0);      
      server = DataServer(owner);
      if(server == ga_nodeid_()){
         ga_server_unlock(g_mutexes, id, server, next[id]);
      }else ga_snd_req(g_mutexes, id,next[id],0,0, 0,0,GA_OP_UNL, owner,server);
}


void ga_lock_(Integer *id)        
{

        if(DEBUG)fprintf(stderr,"%d enter lock\n",ga_nodeid_());

        if(!num_mutexes) ERROR("ga_lock: need to create mutexes first",0);

        if(*id >= num_mutexes)ERROR("ga_lock: need more mutexes", num_mutexes); 

        if(ga_nnodes_() == 1) return;

#       if defined(SERVER_LOCK)
           dst_lock(*id+1);
#       else
           ga_generic_lock(*id+1);
#       endif

        if(DEBUG)fprintf(stderr,"%d leave lock\n",ga_nodeid_());
}



void ga_unlock_(Integer *id)
{
        if(DEBUG)fprintf(stderr,"%d enter unlock\n",ga_nodeid_());

        if(!num_mutexes) ERROR("ga_unlock: need to create mutexes first",0);

        if(*id >= num_mutexes) ERROR("ga_lock: need more mutexes", num_mutexes);

        if(ga_nnodes_() == 1) return;

#       if defined(SERVER_LOCK)
           dst_unlock(*id+1);
#       else
           ga_generic_unlock(*id+1);
#       endif

        if(DEBUG)fprintf(stderr,"%d leave unlock\n",ga_nodeid_());
}
