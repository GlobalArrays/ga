/* preliminary implementation on top of portals */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include "armcip.h"
#include <stdint.h>
#if 0
#include <portals3.h>
#include P3_NAL
#include <p3rt/p3rt.h>
#include <p3api/debug.h>
#include "portals.h"
#include "armcip.h"                                                                                                        
#define MAX_OUT 50
#define MAX_ENT 64
#define MAX_PREPOST 1

typedef enum op {
        ARMCI_PORTALS_PUT,
        ARMCI_PORTALS_NBPUT,
        ARMCI_PORTALS_GET, 
        ARMCI_PORTALS_NBGET, 
        ARMCI_PORTALS_ACC
} optype;

/* array of memory segments and corresponding memory descriptors */
typedef struct md_table{
        int id;
        void * start;
        void * end;
        int bytes;       
        ptl_md_t md;
} md_table_entry_t;


typedef struct desc{
       int active;
       unsigned int tag;      
       int dest_id;
       optype type;
}comp_desc; 

/* structure of computing process */

typedef struct {
  int armci_rank;  /* if different from portals_rank */      
  int rank;       /* my rank*/
  int size;       /* size of the group */
  ptl_handle_me_t me_h[64];
  ptl_handle_me_t md_h[64];

  ptl_handle_eq_t eq_h;
  ptl_handle_ni_t ni_h; 
  ptl_pt_index_t ptl;
  int outstanding_puts;
  int outstanding_gets;
  int outstanding_accs;
  void * buffers; /* ptr to head of buffer */
  int num_match_entries;
} armci_portals_proc_t;

#endif

#define SENDER_MATCHING_BITS 100
#define RECEIVER_MATCHING_BITS 100

/*global variables and data structures */
armci_portals_proc_t _armci_portals_proc_struct;
armci_portals_proc_t *portals = &_armci_portals_proc_struct;
md_table_entry_t _armci_md_table[MAX_ENT];
comp_desc _armci_portals_comp[MAX_OUT];
int ptl_initialized = 0;
int free_desc_index = 0;

void comp_desc_init()
{
  int i;
  comp_desc *c;
  for(i=0; i< MAX_OUT;i++){
      c = &(_armci_portals_comp[i]);
      c->active = 0;
  }
}

int armci_init_portals(void)
{
     
    int rank,size;    
    int num_interface;
    int rc;
    
    if (PtlInit(&num_interface) != PTL_OK) {
        fprintf(stderr, "PtlInit() failed\n");
        exit(1);
    }
  
    /* should this be PTL_PID_ANY */ 
    if (PtlNIInit(PTL_IFACE_DEFAULT, PTL_PID_ANY, NULL, NULL, &(portals->ni_h)) != PTL_OK) {
        fprintf(stderr, "PtlNIInit() failed\n");
        exit(EXIT_FAILURE);
    }

    if (PtlGetRank(&(portals->rank), &(portals->size)) != PTL_OK) {
        fprintf(stderr, "PtlGetRank() failed\n");
        exit(EXIT_FAILURE);
    }

    rc = PtlEQAlloc(portals->ni_h,64, PTL_EQ_HANDLER_NONE, &(portals->eq_h));
    if (rc != PTL_OK) {
            fprintf(stderr, "%d:PtlEQAlloc() failed: %s (%d)\n",
                            portals->rank, PtlErrorStr(rc), rc);
            exit(EXIT_FAILURE);                            
    }
    ptl_initialized = 1;
    portals->num_match_entries = 0;
    comp_desc_init();
    return 0;   
}        


void armci_fini_portals()
{
    int rc;
    PtlNIFini(portals->ni_h);
    PtlFini();
}


/*if needed later */
/*
void armci_update_descriptor()
{


}
*/



int armci_post_descriptor(ptl_md_t md)
{
  int rc;      
  ptl_match_bits_t mb;
  ptl_process_id_t src_id;
   
   mb = RECEIVER_MATCHING_BITS+portals->num_match_entries;
  
  if (PtlGetRankId(PTL_NID_ANY,&src_id) != PTL_OK){
               printf("ERROR IN CONVERTING SRC_ID\n");
               fflush(stdout);
  }
  
  if(portals->num_match_entries == 0)
    rc = PtlMEAttach(portals->ni_h,portals->ptl,src_id,mb,0,PTL_RETAIN,PTL_INS_AFTER,&portals->me_h[0]); 
  else
    rc = PtlMEInsert(portals->me_h[portals->num_match_entries],src_id,mb,0, PTL_RETAIN,PTL_INS_AFTER,&portals->me_h[(portals->num_match_entries)+1]);

  if (rc != PTL_OK) {
      fprintf(stderr, "%d:PtlMEAttach: %s\n", portals->rank, PtlErrorStr(rc));
      exit(EXIT_FAILURE); 
  }
  rc = PtlMDAttach(portals->me_h[portals->num_match_entries],md,PTL_RETAIN,&portals->md_h[portals->num_match_entries]);
  if (rc != PTL_OK) {
       fprintf(stderr, "%d:PtlMDAttach: %s\n", portals->rank, PtlErrorStr(rc)); 
       exit(EXIT_FAILURE);
  } 
 
  return mb;  
}


/* to be called from ARMCI_PORTALS_Malloc */
int armci_prepost_descriptor(void* start, long bytes)
{
  int options;
  int rc;
  ptl_md_t md;
  ptl_handle_md_t md_h;
  void * context;
  int * index;
  ptl_match_bits_t mb;

  context = NULL;
  options = PTL_MD_OP_PUT | PTL_MD_OP_GET | PTL_MD_EVENT_START_DISABLE | PTL_MD_MANAGE_REMOTE;
  md.start = start;
  md.length = bytes;
  md.threshold = PTL_MD_THRESH_INF;
  md.options = options;
  md.user_ptr = context;
  md.eq_handle = portals->eq_h;

  mb = armci_post_descriptor(md);
  _armci_md_table[portals->num_match_entries].start = start;
  _armci_md_table[portals->num_match_entries].bytes = bytes;
  _armci_md_table[portals->num_match_entries].md = md;
  _armci_md_table[portals->num_match_entries].mb = mb;
  portals->num_match_entries++;
  return rc;
}



int armci_client_complete(ptl_event_kind_t *evt,int proc_id, int nb_tag,comp_desc *cdesc,int b_tag )
{
  int rank;
  int rc;  
  ptl_event_t *ev;
  armci_ihdl_t nb_handle;
  comp_desc *temp_comp;
  int temp_tag;
  int temp_proc;;
  while(1)
  { 
        if((rc = PtlEQWait(portals->eq_h, ev)) != PTL_OK){
            fprintf(stderr, "%d:PtlEQWait(): %s\n", portals->rank, PtlErrorStr(rc));
            exit(EXIT_FAILURE);
         }
     
         if (ev->ni_fail_type != PTL_NI_OK) {
             fprintf(stderr, "%d:NI sent %s in event.\n",
                     portals->rank, PtlNIFailStr(portals->ni_h, ev->ni_fail_type));
             exit(EXIT_FAILURE);
         }

        /* handle the corresponding event */
        if (ev->type == PTL_EVENT_SEND_END){
                temp_comp = (comp_desc *)ev->md.user_ptr;
                temp_tag = temp_comp->tag;
                temp_proc = temp_comp->dest_id;
                temp_comp->active = 1;
                if ((nb_tag != 0) && (nb_tag == temp_tag))
                        break;
                else if ((b_tag == 1) && !(cdesc) && (cdesc == temp_comp) ) 
                        break;
                               
        }

        if (ev->type == PTL_EVENT_REPLY_END){
                temp_comp = (comp_desc *)ev->md.user_ptr;
                temp_tag = temp_comp->tag;
                temp_proc = temp_comp->dest_id;
                temp_comp->active = 3;
                portals->outstanding_gets--; 
                if ((nb_tag != 0) && (nb_tag == temp_tag))
                        break;
                else if ((b_tag == 1) && !(cdesc) && (cdesc == temp_comp) )
                        break;
        }
         
        if (ev->type == PTL_EVENT_ACK){
                temp_comp = (comp_desc *)ev->md.user_ptr;
                temp_proc = temp_comp->dest_id;
                temp_comp->active = 2;
                update_fence_array(temp_proc,0);              
                portals->outstanding_puts--; 
        }

        if ( !cdesc && (temp_comp == cdesc)){
             if(cdesc->active == 2 || cdesc->active == 3)
                cdesc->active = 0;
                break;        
        }
        
        
   }
  return rc; 
  
}




int armci_get_offset(ptl_md_t md, void *ptr)
{
  void * start_address;
  int offset;
  start_address = md.start;
  offset =  ptr - start_address;
  return offset;
}




int armci_portals_put(ptl_handle_md_t md_h,ptl_process_id_t dest_id,int bytes,int mb,int local_offset, int remote_offset,int ack )
{
     int rc;
     rc = PtlPutRegion(md_h, local_offset, bytes,ack,dest_id, portals->ptl, 0, mb,remote_offset, 0);
     if (rc != PTL_OK) {
             fprintf(stderr, "%d:PtlPutRegion: %s\n", portals->rank, PtlErrorStr(rc));
             exit(EXIT_FAILURE); 
     }
     return rc;
     
}



int armci_get_md(void * start, int bytes , ptl_md_t * md_ptr, ptl_match_bits_t * mb_ptr)
{
    int i;
    int rc;
    int found = 0;

    for (i=0; i<portals->num_match_entries; i++){
         md_ptr = &(_armci_md_table[i].md);
         if (start >= _armci_md_table[i].start && ((char *)start + bytes) <= (char *)_armci_md_table[i].end)
         {
                 mb_ptr = &(_armci_md_table[i].mb);        
                 found = 1;
                 break;
         }        
    }
         
    return found;
}




comp_desc * get_free_comp_desc()
{
   comp_desc * c;     
   c = &(_armci_portals_comp[free_desc_index]);
   if (c->active != 0)
   {
      armci_client_complete(NULL,c->dest_id,0,c,0); /* there should be a function for reseeting compl_desc*/
                                  
   }
   free_desc_index = ((free_desc_index++) % MAX_OUT);
   return c;

}



/* direct protocol for put */
int armci_portals_direct_send(void *src, void* dst, int bytes, int proc, int tag, NB_CMPL_T *cmpl_info)
{
   int rc;
   int local_offset, remote_offset;
   ptl_match_bits_t mb;
   ptl_md_t md, md_client;
   ptl_handle_md_t client_md_h;
   void * context;
   comp_desc *cdesc;
   ptl_process_id_t dest_id;
   int index;
  // mb = SENDER_MATCHING_BITS; 
   int ack = 1;
   int found;

   dest_id.nid = proc;
   found = armci_get_md(src, bytes, &md, &mb);
   if (!found){
           fprintf(stderr, "unable to find preposted descriptor\n");
           exit(EXIT_FAILURE);
   }
   
   local_offset = armci_get_offset(md,src);
   remote_offset = armci_get_offset(md,dst);

   cdesc = get_free_comp_desc();
   *cmpl_info = 1; /*TOED*/
   md_client.start = md.start;
   md_client.length = bytes;
   md_client.threshold = PTL_MD_THRESH_INF;
   md_client.options = PTL_MD_OP_PUT | PTL_MD_MANAGE_REMOTE  | PTL_MD_EVENT_START_DISABLE;
   if (!tag){
           cdesc->tag = tag;
           cdesc->dest_id = proc;
           cdesc->type = ARMCI_PORTALS_PUT;
           cdesc->active = 0;
   }
   else{
          cdesc->tag = 999999;
          cdesc->dest_id = proc;
          cdesc->type = ARMCI_PORTALS_NBPUT; 
          cdesc->active = 0;
   }
   md_client.user_ptr = (void *)cdesc;
   md_client.eq_handle = portals->eq_h;

   rc = PtlMDBind(portals->ni_h,md_client, PTL_RETAIN, &client_md_h);
   if (rc != PTL_OK) {
       fprintf(stderr, "%d:PtlMDBind: %s\n", portals->rank, PtlErrorStr(rc));
       exit(EXIT_FAILURE);
   }
    
   rc = armci_portals_put(client_md_h,dest_id,bytes,mb,local_offset,remote_offset,ack); 
   update_fence_array(dest_id, 1);
   if(!tag)
   {
           
        armci_client_complete(NULL,proc,0,NULL,1); /* check this later */
   }
   else
        portals->outstanding_puts++;   
   return rc;
}



int armci_portals_complete(int tag, NB_CMPL_T *cmpl_info)
{
   int rc;
   int proc;
   /*TOED*/
   armci_client_complete(NULL,proc,tag,NULL,0);
}



/* direct protocol for get */
int armci_portals_direct_get(void *src, void *dst, int bytes, int proc, int tag, NB_CMPL_T *cmpl_info)
{
   int rc;
   int local_offset, remote_offset;
   ptl_match_bits_t mb;
   ptl_md_t md, md_client;
   ptl_handle_md_t client_md_h;
   comp_desc *cdesc;
   int index;
  // mb = SENDER_MATCHING_BITS; 
   int ack = 1;
   ptl_process_id_t dest_proc;
   int found;
   found = armci_get_md(src, bytes,&md, &mb);
   if (!found){
           fprintf(stderr, "unable to find preposted descriptor for get\n");
           exit(EXIT_FAILURE);
   }
   
   local_offset = armci_get_offset(md,src);
   remote_offset = armci_get_offset(md,dst);

   cdesc = get_free_comp_desc();
   *cmpl_info = 1; /*TOED*/
       
   md_client.start = md.start;
   md_client.length = bytes;
   md_client.threshold = PTL_MD_THRESH_INF;
   md_client.options = PTL_MD_OP_GET | PTL_MD_MANAGE_REMOTE  | PTL_MD_EVENT_START_DISABLE;
   if (!tag){
           cdesc->tag = tag;
           cdesc->dest_id = proc;
           cdesc->type = ARMCI_PORTALS_NBGET;
           cdesc->active =0;
   }
   else{
          cdesc->tag = 999999;
          cdesc->dest_id = proc; 
          cdesc->type = ARMCI_PORTALS_GET;
          cdesc->active = 0;
   }
   md_client.user_ptr = cdesc;
   md_client.eq_handle = portals->eq_h;

   rc = PtlMDBind(portals->ni_h,md_client, PTL_RETAIN, &client_md_h);
   if (rc != PTL_OK) {
       fprintf(stderr, "%d:PtlMDBind: %s\n", portals->rank, PtlErrorStr(rc));
       exit(EXIT_FAILURE);
   }
   /* need to rename this */ 
   rc = armci_portals_put(client_md_h,dest_proc,bytes,mb,local_offset,remote_offset,ack); 
   if(!tag)
   {
           
        armci_client_complete(NULL,proc,0,NULL,1);
   }
   else
        portals->outstanding_gets++;   

}

