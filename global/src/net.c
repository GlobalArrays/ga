#include <stdio.h>
#include "global.h"
#include "globalp.h"
#include "message.h"
#include "macdecls.h"

extern Integer  NumSndReq; /* request counter */


#include "mem.ops.h"
#include "../../server/p2p.h"

typedef struct{
               int type;
               int from;
               int to;
               int len;
}server_header_t; 
server_header_t msg_header;
static int    got_header  =0;
static int    local_done  =0;
static int    others_done =0;

static map[MAX_NPROC][5];

#define     GA_TYPE_FWD 32700 + 30
#define     GA_TYPE_END 32700 + 31

#define DEBUG 0
#define DEBUG0 0

static logical patch_intersect(ilo, ihi, jlo, jhi, ilop, ihip, jlop, jhip)
     Integer *ilo, *ihi, *jlo, *jhi;
     Integer *ilop, *ihip, *jlop, *jhip;
{
     /* check consistency of patch coordinates */
     if( *ihi < *ilo || *jhi < *jlo)     return FALSE; /* inconsistent */
     if( *ihip < *ilop || *jhip < *jlop) return FALSE; /* inconsistent */

     /* find the intersection and update (ilop: ihip, jlop: jhip) */
     if( *ihi < *ilop || *ihip < *ilo) return FALSE; /* don't intersect */
     if( *jhi < *jlop || *jhip < *jlo) return FALSE; /* don't intersect */
     *ilop = MAX(*ilo,*ilop);
     *ihip = MIN(*ihi,*ihip);
     *jlop = MAX(*jlo,*jlop);
     *jhip = MIN(*jhi,*jhip);

     return TRUE;
}


void ga_sock_snd(type, buffer, bytes, to)
     Integer type, bytes, to;
     Void    *buffer;
{
server_header_t send_header;

     if(DEBUG)fprintf(stderr,"node %d sending sock msg type=%d bytes%d\n",
              ga_nodeid_(), type, bytes);

        send_header.type = (int)type;
        send_header.len  = (int)bytes;
        send_header.to   = (int)to;
        send_header.from = (int)ga_msg_nodeid_();

        send_to_server(&send_header, sizeof(send_header));
        send_to_server(buffer, (int)bytes);
}


void ga_sock_rcv(type, buffer, buflen, msglen, from, whofrom)
     Integer type, buflen, *msglen, from, *whofrom;
     Void    *buffer;
{

     if(DEBUG)fprintf(stderr,"node %d receiving sock msg type=%d buflen=%d\n",
              ga_nodeid_(), type, buflen);

     if(got_header){
          if(msg_header.type != type)
              ga_error("ga_sock_rcv: server: wrong type",msg_header.type);
          if(msg_header.len > buflen)
              ga_error("ga_sock_rcv:overflowing buffer",msg_header.len);

          /* get the message body from server socket */
          recv_from_server(buffer, msglen);
          if(*msglen  != msg_header.len)
              ga_error("ga_sock_rcv: inconsistent length header",*msglen);
          *whofrom = msg_header.from;

          got_header = 0;
      }else
          ga_error("ga_sock_rcv: no header",0);
}


void ga_forward(g_src, g_dst, ilo, ihi, jlo, jhi, buf, offset, ld, proc)
   Integer g_src, g_dst, ilo, ihi, jlo, jhi, ld, offset, proc;
   Void *buf;
{
char     *ptr_src, *ptr_dst;
Integer  type, rows, cols, msglen, dim1, dim2;
Integer  len, to;

   if(DEBUG)fprintf(stderr,"%d forwarding [%d:%d, %d:%d]\n", ga_nodeid_(), 
                    ilo, ihi, jlo, jhi);

   ga_inquire_(&g_src, &type, &dim1, &dim2);
   if(type != MT_F_DBL) ga_error("ga_forward: only double type supported",type);
   rows = ihi - ilo +1;
   cols = jhi - jlo +1;

   /* Copy patch [ilo:ihi, jlo:jhi] into MessageBuffer */
   ptr_dst = (char*)MessageSnd->buffer;
   ptr_src = (char *)buf  + GAsizeofM(type)* offset;
   Copy2D(type, &rows, &cols, ptr_src, &ld, ptr_dst, &rows); 

   msglen = rows*cols*GAsizeofM(type);
   *(DoublePrecision*)(MessageSnd->buffer+ msglen) = 1.;
   msglen += GAsizeofM(type);

   MessageSnd->g_a    = g_dst;
   MessageSnd->ilo    = ilo;
   MessageSnd->ihi    = ihi;
   MessageSnd->jlo    = jlo;
   MessageSnd->jhi    = jhi;
   MessageSnd->to     = proc;
   MessageSnd->from   = ga_nodeid_();
   MessageSnd->type   = type;
   MessageSnd->req_tag    = 77;
   MessageSnd->operation  = GA_OP_ACC;

   len = msglen + MSG_HEADER_SIZE;
   to  = ga_nnodes_()-1; /* highest numbered node will forward to remote nodes*/

   if(ga_nodeid_()== to)
      ga_sock_snd(GA_TYPE_FWD, (char*)MessageSnd, len, to);
   else
      ga_msg_snd(GA_TYPE_FWD, (char*)MessageSnd, len, to);
   
}


Integer ga_sock_probe(type, from)
     Integer type, from;
{
     if(from != -1)ga_error("ga_sock_probe: only from=-1 works now",from);

     /* check if msg header was read before */
     if(got_header){

         if(msg_header.type == type) return 1; /* msg type available */ 
         else return(0);
     }

     /*check if there is a message available */
     if(poll_server()){ 

           int msglen;
    
           /* read the message header */
           recv_from_server(&msg_header,&msglen);
           if(msglen != sizeof(msg_header))
                        ga_error("ga_sock_probe: error in header",msglen); 
           got_header=1;

           if(msg_header.type == type) return 1; /* msg type available */
           else return(0);
     }
     return(0);
}


void print_array(a,ld,r,c)
DoublePrecision *a;
Integer r,c,ld;
{
Integer i,j;
       fprintf(stderr,"array %dx%d\n",r,c);
       for(i=0; i<r;i++){
          for(j=0; j<c;j++)
             fprintf(stderr,"%5.1f ",a[j*ld+i]); 
          fprintf(stderr,"\n"); 
       }
}
          

/*\ probe for local/network messages and service them
\*/
void ga_receive_and_service()
{
char *curbuf= (char*)MessageRcv;
Integer ack, buflen=TOT_MSG_SIZE, msglen, whofrom, nproc=ga_nnodes_();
Integer me = ga_nodeid_();


      if(ga_sock_probe(GA_TYPE_FWD,-1)){
         Integer ilo, ihi, jlo, jhi, g_a, ld, nowners, to;
         DoublePrecision alpha=1.;
         ga_sock_rcv(GA_TYPE_FWD, curbuf, buflen, &msglen,(Integer)-1,&whofrom);
         g_a = ((struct message_struct*)curbuf)->g_a;
         ilo = ((struct message_struct*)curbuf)->ilo;
         ihi = ((struct message_struct*)curbuf)->ihi;
         jlo = ((struct message_struct*)curbuf)->jlo;
         jhi = ((struct message_struct*)curbuf)->jhi;
         ld  = ihi - ilo + 1;
         if(DEBUG)fprintf(stderr,"%d got remote data [%d:%d, %d:%d]\n",me, 
                          ilo, ihi, jlo, jhi);
         if(DEBUG)
                  print_array((DoublePrecision*)((struct message_struct*)curbuf)
                               ->buffer, ld, ld, jhi-jlo+1);

#        ifndef SHMEM

           if(!ga_locate_region_(&g_a, &ilo, &ihi, &jlo, &jhi, map, &nowners ))
               ga_error("ga_receive_and_service: request invalid",g_a);

           to = map[0][4];

           /* if data goes to one processor only there is no need to copy */
           if(DEBUG)fprintf(stderr,"processor owners=%d\n",nowners);
           if(nowners==1 && (to != me)){

              if(DEBUG)fprintf(stderr,"forwarding req to=%d\n",to);
              ((struct message_struct*)curbuf)->to = to;
              ga_msg_snd(GA_TYPE_REQ, curbuf, msglen, to);

              NumSndReq++; /*  increment request counter */

           }else

#        endif
              ga_acc_(&g_a, &ilo, &ihi, &jlo, &jhi, 
                     (DoublePrecision*)((struct message_struct*)curbuf)->buffer,
                     &ld, &alpha);

         if(DEBUG)fprintf(stderr,"ga_receive_and_service:%d ga_acc done\n",me);
      }


      if(ga_sock_probe(GA_TYPE_END,-1)){
         ga_sock_rcv(GA_TYPE_END, (char*)&ack, 0, &msglen,(Integer)-1,&whofrom);
         others_done=1;
         if(DEBUG)fprintf(stderr,"%d got sock msg type %d\n",me, GA_TYPE_END);
      }



      if(ga_msg_probe(GA_TYPE_FWD,-1)){
         ga_msg_rcv(GA_TYPE_FWD, curbuf, buflen, &msglen, (Integer)-1,&whofrom);
         ga_sock_snd(GA_TYPE_FWD, curbuf, msglen, 0);
         if(DEBUG)fprintf(stderr,"%d got msg type %d\n",me, GA_TYPE_FWD);
      }


      if(ga_msg_probe(GA_TYPE_END,-1)){
         ga_msg_rcv(GA_TYPE_END, (char*)&ack, 0, &msglen,(Integer)-1,&whofrom);
         local_done++;
         if(local_done == nproc) ga_sock_snd(GA_TYPE_END, (char*)&ack, 0, 0);
         if(DEBUG)fprintf(stderr,"%d got msg type %d done=%d\n",me, GA_TYPE_END,
local_done);
      }


}


void ga_snd_to_server(g_src, g_dst, ilo, ihi, jlo, jhi)
   Integer  *g_src, *g_dst, *ilo, *ihi, *jlo, *jhi;
{
Integer proc, idx, ld, type, dim1, dim2;
Integer ilop, ihip, jlop, jhip, offset;
char    *buf;

      proc = ga_nodeid_();
      ga_inquire_(g_src, &type, &dim1, &dim2);
      ga_distribution_(g_src, &proc,  &ilop, &ihip, &jlop, &jhip);

      if(patch_intersect(ilo, ihi, jlo, jhi, &ilop, &ihip, &jlop, &jhip)){

            Integer ilo_chunk, ihi_chunk, jlo_chunk, jhi_chunk;
            Integer TmpSize = MSG_BUF_SIZE/GAsizeofM(type) -GAsizeofM(type);
            Integer ilimit  = MIN(TmpSize, ihip-ilop+1);
            Integer jlimit  = MIN(TmpSize/ilimit, jhip-jlop+1);

        if(DEBUG)fprintf(stderr,"%d my contrib [%d:%d, %d:%d]\n", ga_nodeid_(),
                    ilop, ihip, jlop, jhip);

            for(jlo_chunk = jlop; jlo_chunk <= jhip; jlo_chunk += jlimit){
               jhi_chunk  = MIN(jhip, jlo_chunk+jlimit-1);
               for( ilo_chunk = ilop; ilo_chunk<= ihip; ilo_chunk += ilimit){

                  /* the highest node has to be polling socket while sending */
                  if(ga_nodeid_()==ga_nnodes_()-1)
                                   ga_receive_and_service();

                  ihi_chunk = MIN(ihip, ilo_chunk+ilimit-1);
                  ga_access_(g_src, &ilo_chunk,&ihi_chunk,&jlo_chunk,&jhi_chunk,
                             &idx, &ld);
                  idx --; /* f2c index translation */

                  buf =(type==MT_F_DBL)?(char*)(DBL_MB+idx):(char*)(INT_MB+idx);
                  offset = 0;

                  ga_forward(*g_src, *g_dst, ilo_chunk, ihi_chunk,
                             jlo_chunk, jhi_chunk, buf, offset, ld, proc);

               }
            }
      }
}


Integer ga_net_nnodes_()
{
Integer i, nproc=0;

#  ifdef IWAY
     return(ga_nnodes_());
#  else
     for(i=0; i<GA_n_clus; i++){
        nproc += GA_clus_info[i].nslave;
     }
     return(nproc);
#  endif
}


Integer ga_net_nodeid_()
{
Integer i, me = ga_nodeid_();

#  ifndef IWAY
     for(i=0; i < GA_clus_id; i++){
        me += GA_clus_info[i].nslave;
     }
#  endif
   return(me);
}


/*\  merges copies of an array replicated accross the network
 *   [ilo:ihi, jlo:jhi] is the contribution of each network cluster
\*/
void ga_net_merge_(g_a, ilo, ihi, jlo, jhi)
   Integer  *g_a, *ilo, *ihi, *jlo, *jhi;
{
Integer me=ga_nodeid_(), nproc=ga_nnodes_(), ack, to;
Integer g_src, g_dst;

   if(GA_n_clus < 2) return;
   if(ClusterMode) return;  /* hierarchy of clusters is not handled now */

   if(DEBUG)
        fprintf(stderr,"ga_net_merge: nodes=%d me=%d\n", ga_net_nnodes_(), ga_net_nodeid_());

   /* temp array is to avoid overwritting the original data before it is sent */
   if(! ga_duplicate(g_a, &g_src, "ga_merge_copy"))
      ga_error("ga_net_merge: no memory for temporary array",*g_a);
   ga_copy_(g_a, &g_src);
   g_dst = *g_a;

   to = nproc-1;
   ga_sync_();

   local_done =0;
   others_done =0;
   
   ga_snd_to_server(&g_src, &g_dst, ilo, ihi, jlo, jhi);

   if(me != nproc-1){
      ga_msg_snd(GA_TYPE_END, (char*)&ack, 0, to);
   } else{ 

      local_done ++;
      if(local_done == nproc) ga_sock_snd(GA_TYPE_END, (char*)&ack, 0, to);
      while(local_done < nproc || (!others_done))
                     ga_receive_and_service();
     
      if(DEBUG)
        fprintf(stderr,"SERVER:local=%d others=%d\n",local_done,others_done);
   }

   ga_destroy_(&g_src); /*dstroy temp array */
   ga_sync_();
}

