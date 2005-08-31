/* preliminary implementation on top of portals */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include "armcip.h"
#include <stdint.h>


#define SENDER_MATCHING_BITS 100
#define RECEIVER_MATCHING_BITS 100

/*global variables and data structures */
armci_portals_proc_t _armci_portals_proc_struct;
armci_portals_proc_t *portals = &_armci_portals_proc_struct;
md_table_entry_t _armci_md_table[MAX_ENT];
comp_desc _armci_portals_comp[MAX_OUT];
int ptl_initialized = 0;
int free_desc_index = 0;
FILE *utcp_lib_out;
FILE* utcp_api_out;


/* */
void print_mem_desc_table()
{
  int i;
  for (i = 0; i<portals->num_match_entries;i++)
  {
    printf ("%d: i=%d print_mem match_ent=%d,start:%p, end:%p, bytes:%d,%p-md.length=%llu, %p-md.start=%p\n",  
                    
            portals->rank,i,portals->num_match_entries,_armci_md_table[i].start,
            _armci_md_table[i].end, _armci_md_table[i].bytes,&(_armci_md_table[i].md.length),
             _armci_md_table[i].md.length,  &(_armci_md_table[i].md.start),_armci_md_table[i].md.start);
    fflush(stdout);
  }       
        
}

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
     
    int num_interface;
    int rc;
   
    if (PtlInit(&num_interface) != PTL_OK) {
        fprintf(stderr, "PtlInit() failed\n");
        exit(1);
    }
    portals->ptl = 4;

    /* should this be PTL_PID_ANY */ 
    if (PtlNIInit(PTL_IFACE_DEFAULT, PTL_PID_ANY, NULL, NULL, &(portals->ni_h)) != PTL_OK) {
        fprintf(stderr, "PtlNIInit() failed\n");
        exit(EXIT_FAILURE);
    }

    if (PtlGetRank(&(portals->rank), &(portals->size)) != PTL_OK) {
        fprintf(stderr, "PtlGetRank() failed\n");
        exit(EXIT_FAILURE);
    }

    printf("the rank is %d, size is %d\n",portals->rank,portals->size);
    fflush(stdout);

    rc = PtlEQAlloc(portals->ni_h,64, PTL_EQ_NONE, &(portals->eq_h));
    if (rc != PTL_OK) {
            fprintf(stderr, "%d:PtlEQAlloc() failed: %s (%d)\n",
                            portals->rank, PtlErrorStr(rc), rc);
            exit(EXIT_FAILURE);                            
    }
    ptl_initialized = 1;
    portals->num_match_entries = 0;
    comp_desc_init();
    utcp_lib_out = stdout;
    utcp_api_out = stdout;
    return 0;   
}



void armci_fini_portals()
{
    PtlNIFini(portals->ni_h);
    PtlFini();
}


int armci_pin_contig_hndl(void *start,int bytes, ARMCI_MEMHDL_T *reg_mem)
{
  int rc;
  void * context;
  ptl_md_t *md_ptr;
  ptl_match_bits_t *mb;
  ptl_process_id_t match_id;

  md_ptr = &reg_mem->mem_dsc;
  mb = &reg_mem->match_bits; 
  /*md_ptr = &(_armci_md_table[portals->num_match_entries].md);*/
  /*memset(&(_armci_md_table[portals->num_match_entries].md),0, sizeof(_armci_md_table[portals->num_match_entries].md));*/
  context = NULL;
  md_ptr->start = start;
  md_ptr->length = bytes;
  md_ptr->threshold = PTL_MD_THRESH_INF;
  md_ptr->options =  PTL_MD_OP_PUT | PTL_MD_OP_GET | PTL_MD_EVENT_START_DISABLE | PTL_MD_MANAGE_REMOTE;
  md_ptr->user_ptr = context;
  md_ptr->eq_handle = portals->eq_h;
  md_ptr->max_size =0;

  /*print_mem_desc_table();*/
  
  *mb = RECEIVER_MATCHING_BITS+portals->num_match_entries;
 
  match_id.nid = PTL_NID_ANY;
  match_id.pid = PTL_PID_ANY; 

  printf("about to call attach\n");
  fflush(stdout);

  if(portals->num_match_entries == 0){
    rc = PtlMEAttach(portals->ni_h,portals->ptl,match_id,*mb,0,PTL_RETAIN,PTL_INS_AFTER,
                    &(portals->me_h[portals->num_match_entries])); 
  }
  else{
    rc = PtlMEInsert(portals->me_h[(portals->num_match_entries - 1)],match_id,*mb,0, 
                    PTL_RETAIN,PTL_INS_AFTER,
                    &portals->me_h[(portals->num_match_entries)]);
  }
  
  if (rc != PTL_OK) {
      fprintf(stderr, "%d:PtlMEAttach: %s\n", portals->rank, PtlErrorStr(rc));
      exit(EXIT_FAILURE); 
  }

  printf("about to call md attach\n");
  fflush(stdout);

  if(portals->num_match_entries == 0){
    rc = PtlMDAttach(portals->me_h[portals->num_match_entries],*md_ptr,PTL_RETAIN,
                    &(portals->md_h[portals->num_match_entries]));
     if (rc != PTL_OK) {
         fprintf(stderr, "%d:PtlMDAttach: %s\n", portals->rank, PtlErrorStr(rc)); 
         exit(EXIT_FAILURE);
     }
     printf("%d: ,finished attach\n",portals->rank);
     fflush(stdout);
  }  

  else{ 
     rc = PtlMDAttach(portals->me_h[(portals->num_match_entries)],*md_ptr,
                     PTL_RETAIN,&(portals->md_h[(portals->num_match_entries)]));
     if (rc != PTL_OK) {
         fprintf(stderr, "%d:PtlMDAttach: %s\n", portals->rank, PtlErrorStr(rc)); 
         exit(EXIT_FAILURE);
     } 
  }  

  
  portals->num_match_entries++;
  print_mem_desc_table(); 
  printf("%d:in prepost i=%d %p-start=%p,  %p-length=%llu table.eq_handle = %u\n",
                portals->rank,(portals->num_match_entries - 1),
                &(_armci_md_table[portals->num_match_entries - 1].md.start),
                 _armci_md_table[portals->num_match_entries - 1].md.start,
                &(_armci_md_table[portals->num_match_entries - 1].md.length),
                 (_armci_md_table[portals->num_match_entries - 1].md.length),
                _armci_md_table[portals->num_match_entries - 1].md.eq_handle);
  fflush(stdout); 

  return rc;
}



int armci_client_complete(ptl_event_kind_t *evt,int proc_id, int nb_tag,comp_desc *cdesc,int b_tag )
{
  int rc;  
  ptl_event_t *ev = NULL;
  /*armci_ihdl_t nb_handle;*/
  comp_desc *temp_comp = NULL;
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
                armci_update_fence_array(temp_proc,0);              
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



/* XXX: need to add code in regions to exchange base ptr and size of each allocated region */
ptl_size_t armci_get_offset(ptl_md_t md, void *ptr, int proc)
{
  void * start_address;
  ptl_size_t offset;
  start_address = md.start;
  offset =  (char *)ptr - (char *)start_address;
  printf("%d: start is %p, md.start is %p , offset is %llu\n", portals->rank,
                  ptr, start_address,offset);
  fflush(stdout); 
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

int armci_portals_get(ptl_handle_md_t md_h,ptl_process_id_t dest_id,int bytes,int mb,int local_offset, int remote_offset)
{
     int rc;
     printf("%d: about to call ptl_get, dest_id is %d, md_handle is %p\n",portals->rank,dest_id, md_h);
     fflush(stdout);
     rc = PtlGetRegion(md_h, local_offset, bytes,dest_id, portals->ptl, 0, mb,remote_offset);
     if (rc != PTL_OK) {
             fprintf(stderr, "%d:PtlGetRegion: %s\n", portals->rank, PtlErrorStr(rc));
             exit(EXIT_FAILURE); 
     }
     return rc;
     
}





int armci_get_md(void * start, int bytes , ptl_md_t * md_ptr, ptl_match_bits_t * mb_ptr)
{
    int i;
    int found = 0;
    
    printf("inside armci_get_md, the value of portals->num_match_ent is %d\n", portals->num_match_entries);
    fflush(stdout);
    
    for (i=0; i<portals->num_match_entries; i++){
         md_ptr = &(_armci_md_table[i].md);
         /*md_ptr = _armci_md_table[i].md;*/
         
         printf("the value of start is %p,  bytes is %d, the value of table-start is %p,                the value of table-end is %p,md_ptr->start is %p\n",start, 
         
                         bytes,_armci_md_table[i].start, _armci_md_table[i].end, md_ptr->start);
         fflush(stdout);
         printf("%d: start: %p, tab.start: %p, start+bytes is %p, tab.end is %p\n",
                         portals->rank,start, _armci_md_table[i].start,
                         ((char *)start + bytes), (char *)_armci_md_table[i].end );
         fflush(stdout);
         if ( (start >= _armci_md_table[i].start)  && 
                         ( ((char *)start + bytes) <= (char *)_armci_md_table[i].end)  )
         {
                 mb_ptr = &(_armci_md_table[i].mb);        
                 found = 1;
                 break;
         }        
    }
    printf("%d: returning from get_md found is %d , entry is %d , md_ptr->start is %p,                 md_ptr->eq_handle %u\n",portals->rank, found, i, md_ptr->start, 
    
                    md_ptr->eq_handle);
    fflush(stdout); 
         
    return found;
}




comp_desc * get_free_comp_desc(int * comp_id)
{
   comp_desc * c;     
   c = &(_armci_portals_comp[free_desc_index]);
   if (c->active != 0)
   {
      armci_client_complete(NULL,c->dest_id,0,c,0); /* there should be a function for reseeting compl_desc*/
                                  
   }
   *comp_id = free_desc_index;
   printf("the value of comp_desc_id is %d\n",*comp_id);
   fflush(stdout);
   free_desc_index = ((free_desc_index + 1) % MAX_OUT);
   return c;

}


int armci_client_direct_send(void *src, void* dst, int bytes, int proc, NB_CMPL_T *cmpl_info, int tag, ARMCI_MEMHDL_T *lochdl, ARMCI_MEMHDL_T *remhdl )
{
    int rc, i;
    ptl_size_t offset_local, offset_remote;
    ptl_match_bits_t mb;
    ptl_md_t *md_remote,md, *md_local;
    ptl_md_t * md_ptr;
    ptl_match_bits_t * mb_ptr;
    ptl_handle_md_t *md_hdl_local;
    comp_desc *cdesc;
    ptl_process_id_t dest_proc;
    int c_info;
    int lproc,rproc;
    int ack = 1;
 
    dest_proc.nid = proc;
    dest_proc.pid = PTL_PID_ANY;
    md_local = &lochdl->mem_dsc; 
    md_hdl_local = &lochdl->mem_dsc_hndl; 
    md_remote =&remhdl->mem_dsc;
    offset_local = lochdl->offset;
    offset_remote = remhdl->offset;

    cdesc = get_free_comp_desc(&c_info);
    *cmpl_info = c_info; /*TOED*/
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

    md_local->user_ptr = (void *)cdesc;
    md_local->eq_handle = portals->eq_h;

    rc = PtlMDBind(portals->ni_h,*md_local, PTL_RETAIN, md_hdl_local);
    if (rc != PTL_OK){
       fprintf(stderr, "%d:PtlMDBind: %s\n", portals->rank, PtlErrorStr(rc));
       armci_die("ptlmdbind failed",0);
    }
    
    rc = PtlPutRegion(md_hdl_local,offset_local,bytes,ack,dest_proc,
                   portals->ptl,
                   0, mb,offset_remote, 0);
    if (rc != PTL_OK){
       fprintf(stderr, "%d:PtlPutRegion: %s\n", portals->rank,PtlErrorStr(rc));
       exit(EXIT_FAILURE); 
    }
  
    armci_update_fence_array(proc, 1);
    if(!tag){
       armci_client_complete(NULL,proc,0,NULL,1); /* check this later */
    }
    else
       portals->outstanding_puts++;   
    return rc;
}

#if 0
int armci_portals_direct_send(void *src, void* dst, int bytes, int proc, int tag, NB_CMPL_T *cmpl_info)
{
   int rc, i;
   ptl_size_t local_offset, remote_offset;
   ptl_match_bits_t mb;
   ptl_md_t md, md_client;
   ptl_md_t * md_ptr;
   ptl_match_bits_t * mb_ptr;
   ptl_handle_md_t client_md_h;
   comp_desc *cdesc;
   ptl_process_id_t dest_proc;
   int c_info;
   int lproc,rproc;
  /* mb = SENDER_MATCHING_BITS; */
   int ack = 1;
   int found = -2;

   dest_proc.nid = proc;
   dest_proc.pid = PTL_PID_ANY;

   //found = armci_get_md(src, bytes, &md, &mb);
    for (i=0; i<portals->num_match_entries; i++){
         md_ptr = &(_armci_md_table[i].md);
         //md_ptr = _armci_md_table[i].md;
         
         
         printf("%d: start: %p, tab.start: %p, start+bytes is %p, tab.end is %p\n",portals->rank,
                         src, _armci_md_table[i].start, ((char *)src + bytes), 
                         (char *)_armci_md_table[i].end );
         fflush(stdout);
         
         if ( (src >= _armci_md_table[i].start)  && ( ((char *)src + bytes) <= (char *)_armci_md_table[i].end)  )
         {
                 mb = (_armci_md_table[i].mb);        
                 found = 1;
                 break;
         }        
    }
   
   
   
   printf("%d: the val of found is %d\n",portals->rank,found);
   fflush(stdout);
   
   if (!found){
           fprintf(stderr, "unable to find preposted descriptor\n");
           exit(EXIT_FAILURE);
   }
   
   local_offset = armci_get_offset(md,src,lproc);
   remote_offset = armci_get_offset(md,dst,rproc);

   cdesc = get_free_comp_desc(&c_info);
   *cmpl_info = c_info; /*TOED*/
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
    
   //rc = armci_portals_put(client_md_h,dest_proc,bytes,mb,local_offset,remote_offset,ack); 
   rc = PtlPutRegion(client_md_h, local_offset, bytes,ack,dest_proc, portals->ptl, 
                   0, mb,remote_offset, 0);
   if (rc != PTL_OK) {
        fprintf(stderr, "%d:PtlPutRegion: %s\n", portals->rank, PtlErrorStr(rc));
        exit(EXIT_FAILURE); 
   }
  
  
   armci_update_fence_array(proc, 1);
   if(!tag)
   {
           
        armci_client_complete(NULL,proc,0,NULL,1); /* check this later */
   }
   else
        portals->outstanding_puts++;   
   return rc;
}
#endif



int armci_portals_complete(int tag, NB_CMPL_T *cmpl_info)
{
   int rc;
   int proc;
   /*TOED*/
   rc = armci_client_complete(NULL,proc,tag,NULL,0);
   return rc;
}



/* direct protocol for get */
int armci_portals_direct_get(void *src, void *dst, int bytes, int proc, int tag, NB_CMPL_T *cmpl_info)
{
   int rc, i, found = -2;
   ptl_size_t local_offset, remote_offset;
   ptl_match_bits_t mb, mb_ptr;
   ptl_md_t md, md_client;
   ptl_md_t * md_ptr;
   ptl_handle_md_t client_md_h;
   comp_desc *cdesc;
   int lproc,rproc;
   int c_info = 9990; /* need to initialize this ***/
   /*mb = SENDER_MATCHING_BITS; */
   ptl_process_id_t dest_proc;
   dest_proc.nid = proc;
   dest_proc.pid = PTL_PID_ANY;
   printf("the value of src is %p\n",src);
   fflush(stdout);
   /*found = armci_get_md(src, bytes,&md, &mb);*/
   
    for (i=0; i<portals->num_match_entries; i++){
         md_ptr = &(_armci_md_table[i].md);
         /*md_ptr = _armci_md_table[i].md;*/
         
         printf("the value of src is %p,  bytes is %d, the value of table-start is %p,                the value of table-end is %p,md_ptr->start is %p\n",src, bytes,
         
                         _armci_md_table[i].start, _armci_md_table[i].end, md_ptr->start);
         fflush(stdout);
         printf("%d: start: %p, tab.start: %p, start+bytes is %p, tab.end is %p\n",
                         portals->rank,src, _armci_md_table[i].start,
                         ((char *)src + bytes), (char *)_armci_md_table[i].end );
         fflush(stdout);
         
         if ( (src >= _armci_md_table[i].start)  && 
                         ( ((char *)src + bytes) <= (char *)_armci_md_table[i].end)  )
         {
                 mb = (_armci_md_table[i].mb);        
                 found = 1;
                 break;
         }        
    }
   
   
   if (!found){
           fprintf(stderr, "unable to find preposted descriptor for get\n");
           exit(EXIT_FAILURE);
   }
  
   
   printf("inside armci_portals_direct_get: calling local offset\n");
   fflush(stdout);
   printf("md structure : md.start is %p, md.length is %llu\n", md_ptr->start, md_ptr->length);
   fflush(stdout);
   local_offset = armci_get_offset(*md_ptr,src,lproc);
   printf("calling remote offset\n");
   fflush(stdout);

   remote_offset = armci_get_offset(*md_ptr,dst,rproc);

   cdesc = get_free_comp_desc(&c_info);
   /**cmpl_info = c_info; */
       
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
   /*rc = armci_portals_get(client_md_h,dest_proc,bytes,mb,local_offset,remote_offset); */
   
   printf("%d: about to call ptl_get, dest_id is %d, md_handle is %p\n"
                   ,portals->rank,dest_proc.nid, client_md_h);
   fflush(stdout);
   rc = PtlGetRegion(client_md_h, local_offset, bytes,dest_proc, portals->ptl, 0, 
                   mb,remote_offset);
   if (rc != PTL_OK) {
           fprintf(stderr, "%d:PtlGetRegion: %s\n", portals->rank, 
                             PtlErrorStr(rc));
             exit(EXIT_FAILURE); 
   }
     
   if(!tag)
   {
           
        armci_client_complete(NULL,proc,0,NULL,1);
   }
   else
        portals->outstanding_gets++;   

   return 1;
}

