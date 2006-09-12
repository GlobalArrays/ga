/* preliminary implementation on top of portals */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include "armcip.h"
#include <stdint.h>

#define DEBUG
#ifdef CATAMOUNT
#include "locks.h"
typedef struct {
       int off;
       int desc;
} cnos_mutex_t;

static cnos_mutex_t *_mutex_array;
#endif

/*global variables and data structures */
armci_portals_proc_t _armci_portals_proc_struct;
armci_portals_proc_t *portals = &_armci_portals_proc_struct;
md_table_entry_t _armci_md_table[MAX_ENT];
comp_desc _armci_portals_comp[MAX_OUT];
int ptl_initialized = 0;
int free_desc_index = 0;
FILE *utcp_lib_out;
FILE* utcp_api_out;



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
    int npes,i;
   
    if (PtlInit(&num_interface) != PTL_OK) {
       fprintf(stderr, "PtlInit() failed\n");
       exit(1);
    }
    portals->ptl = 44;

    if((rc=PtlNIInit(IFACE_FROM_BRIDGE_AND_NALID(PTL_DEFAULT_BRIDGE,PTL_IFACE_DEFAULT), PTL_PID_ANY, NULL, NULL, &(portals->ni_h))) != PTL_OK){
#ifdef XT3
       if(rc!=PTL_IFACE_DUP)
#endif
       {
       printf( "PtlNIInit() failed %d\n",rc);
       exit(EXIT_FAILURE);
       }
    }

#ifdef DEBUG
    PtlNIDebug(portals->ni_h,PTL_DEBUG_ALL);
#endif

    portals->size=cnos_get_n_pes_in_app();
    PtlGetId(portals->ni_h,&portals->ptl_my_procid);
    printf("%d:the rank is %d, size is %d\n",armci_me,portals->ptl_my_procid,portals->size);
    if ((npes = cnos_get_nidpid_map(&portals->ptl_pe_procid_map)) == -1) {
      printf(" LIBSMA ERROR:Getting proc id/PE map failed (npes=%d)\n", npes);
    }
    for (i = 0; i < npes; i++) {
      printf("%d:PE %d is 0x%lx/%d (nid/pid)\n", portals->ptl_my_procid,i,portals->ptl_pe_procid_map[i].nid,portals->ptl_pe_procid_map[i].pid);
    }

    /* Allocate one shared event queue for all operations 
     * TODO tune size.
     */

    rc = PtlEQAlloc(portals->ni_h,64,NULL, &(portals->eq_h));
    if (rc != PTL_OK) {
       printf(stderr, "%d:PtlEQAlloc() failed: %s (%d)\n",
                            portals->ptl_my_procid, ptl_err_str[rc], rc);
       exit(EXIT_FAILURE);                            
    }
    ptl_initialized = 1;
    portals->num_match_entries = 0;
    comp_desc_init();


#ifndef XT3
    utcp_lib_out = stdout;
    utcp_api_out = stdout;
#endif

    printf("FINISHED PORTALS INIT\n");
    fflush(stdout); 
    return 0;   
}



void armci_fini_portals()
{
    printf("ENTERING ARMCI_FINI_PORTALS\n");fflush(stdout);    
    PtlNIFini(portals->ni_h);
    PtlFini();
    printf("LEAVING ARMCI_FINI_PORTALS\n");fflush(stdout);    
    
}



void armci_serv_register_req(void *start,int bytes, ARMCI_MEMHDL_T *reg_mem)
{
    int rc;
    void * context;
    ptl_md_t *md_ptr;
    ptl_match_bits_t *mb;
    ptl_process_id_t match_id;
    ptl_handle_md_t *md_h;

#ifdef DEBUG
    printf("inside portals.c : size of mem_hndl is %d\n", sizeof(region_memhdl_t));
    printf("\n%d:armci_serv_register_req start=%p bytes=%d",armci_me,start,bytes);fflush(stdout);
#endif
    
    md_ptr = &reg_mem->mem_dsc;
    mb = &reg_mem->match_bits; 
    md_h = &reg_mem->mem_dsc_hndl;
    context = NULL;

    md_ptr->start = start;
    md_ptr->length = bytes;
    md_ptr->threshold = PTL_MD_THRESH_INF;
    md_ptr->options =  PTL_MD_OP_PUT | PTL_MD_OP_GET | PTL_MD_MANAGE_REMOTE;
    md_ptr->user_ptr = context;
    /*eq_hdl is null for the attaches done for a remote proc*/
    /*md_ptr->eq_handle = portals->eq_h;*/
    md_ptr->eq_handle = (ptl_handle_eq_t)NULL;
    md_ptr->max_size =0;
    *mb = 100;
 
    match_id.nid = PTL_NID_ANY;
    match_id.pid = PTL_PID_ANY; 

#ifdef DEBUG
    printf("about to call attach\n");
    fflush(stdout);
#endif

    rc = PtlMEAttach(portals->ni_h,portals->ptl,match_id,*mb,0,
		     PTL_RETAIN,PTL_INS_AFTER,
		     &(portals->me_h[portals->num_match_entries])); 
  
    if (rc != PTL_OK) {
      printf("%d:PtlMDAttach: %s\n", portals->ptl_my_procid, ptl_err_str[rc]);
      armci_die("portals attach error2",rc);
    }

#ifdef DEBUG
    printf("about to call md attach: the md_h is %p\n",md_h);
    fflush(stdout);
#endif
    rc = PtlMDAttach(portals->me_h[portals->num_match_entries],*md_ptr,PTL_RETAIN,md_h);
                    
    if (rc != PTL_OK) {
      printf("%d:PtlMDAttach: %s\n", portals->ptl_my_procid, ptl_err_str[rc]);
      armci_die("portals attach error1",rc);
    }
     
#ifdef DEBUG
    printf("%d: ,finished attach\n",portals->ptl_my_procid);
    fflush(stdout);
#endif
  
    portals->num_match_entries++;
 
}

int armci_pin_contig_hndl(void *start,int bytes, ARMCI_MEMHDL_T *reg_mem)
{
   int rc;
   void * context;
   ptl_md_t *md_ptr;
   ptl_match_bits_t *mb;
   ptl_process_id_t match_id;
   ptl_handle_md_t *md_h;

#ifdef DEBUG
   printf("inside portals.c : size of mem_hndl is %d\n", sizeof(region_memhdl_t));
   printf("\n%d:armci_pin_contig_hndl start=%p bytes=%d",armci_me,start,bytes);fflush(stdout);
#endif
    
   md_ptr = &reg_mem->mem_dsc;
   md_h = &reg_mem->mem_dsc_hndl;
   context = NULL;
   md_ptr->start = start;
   md_ptr->length = bytes;
   md_ptr->threshold = PTL_MD_THRESH_INF;
   md_ptr->options =  PTL_MD_OP_GET | PTL_MD_EVENT_START_DISABLE;
   /*md_ptr->options =  PTL_MD_EVENT_START_DISABLE;*/
                      
   md_ptr->user_ptr = context;
   md_ptr->eq_handle = portals->eq_h;
   md_ptr->max_size =0;
   printf("\n%d:lochdl=%p",armci_me,md_h);
 
    rc = PtlMDBind(portals->ni_h,*md_ptr, PTL_RETAIN, md_h);
    if (rc != PTL_OK){
       printf("%d:PtlMDBind: %s\n", portals->ptl_my_procid, ptl_err_str[rc]);
       armci_die("ptlmdbind failed",0);
    }
   reg_mem->mem_dsc_save=reg_mem->mem_dsc;
    return 1;

}



int armci_client_complete(ptl_event_kind_t *evt,int proc_id, int nb_tag,comp_desc *cdesc,int b_tag )
{
  int rc;  
  ptl_event_t ev_t;
  ptl_event_t *ev=&ev_t;
  /*armci_ihdl_t nb_handle;*/
  comp_desc *temp_comp = NULL;
  int temp_tag;
  int loop = 1;
  int temp_proc;

#ifdef DEBUG
  printf("entering armci_client_complete\n");
  fflush(stdout);
#endif

    while(loop){
        if(cdesc)
        {        
                if(cdesc->active == 2 || cdesc->active == 3)
                {
                        cdesc->active = 0;
                        loop = 0;
                        break;        
                }
        }   
       ev->type=0;
       if((rc = PtlEQWait(portals->eq_h, ev)) != PTL_OK){
         printf("%d:PtlEQWait(): \n", portals->ptl_my_procid,rc); 
         armci_die("EQWait problem",rc);
       }
       printf("\n%d:done waiting type=%d\n",armci_me,ev->type);
       fflush(stdout);
    
       if (ev->ni_fail_type != PTL_NI_OK) {
         printf("%d:NI sent %d in event.\n",
                         portals->ptl_my_procid,  ev->ni_fail_type); 
         armci_die("event failure problem",0);
       }

       /* handle the corresponding event */
       if (ev->type == PTL_EVENT_SEND_END){
#ifdef DEBUG
         printf ("INSIDED PTL_EVENT_SEND_END\n");
         fflush(stdout);      
#endif               
         continue;
         temp_comp = (comp_desc *)ev->md.user_ptr;
         temp_tag = temp_comp->tag;
         temp_proc = temp_comp->dest_id;
         temp_comp->active = 1;
         if ((nb_tag != 0) && (nb_tag == temp_tag))
           break;
         else if ((b_tag == 1) && !(cdesc))
         {
               loop = 0;
#ifdef DEBUG
               printf("finished receiving event poll\n");
               fflush(stdout);
#endif
               break;
                              
         }      
       }

       if (ev->type == PTL_EVENT_REPLY_END){
#ifdef DEBUG
         printf ("INSIDED PTL_EVENT_REPLY_END\n");
         fflush(stdout);  
#endif    
         temp_comp = (comp_desc *)ev->md.user_ptr;
         temp_tag = temp_comp->tag;
         temp_proc = temp_comp->dest_id;
         temp_comp->active = 3;
         if ((nb_tag != 0) && (nb_tag == temp_tag))
           break;
         else if ((b_tag == 1) && !(cdesc))
          {
                  temp_comp->active = 0;
                  loop = 0; 
                  
#ifdef DEBUG
                  printf("breaking from event poll\n");
                  fflush(stdout); 
#endif
                  
                  break;
          }
       } 
         
        if (ev->type == PTL_EVENT_ACK){
#ifdef DEBUG
                printf ("INSIDED PTL_EVENT_ACK\n");
                fflush(stdout);      
#endif
                temp_comp = (comp_desc *)ev->md.user_ptr;
                temp_proc = temp_comp->dest_id;
                temp_comp->active = 2;
                armci_update_fence_array(temp_proc,0);              
                portals->outstanding_puts--; 
                                break;
#if 0
                if(evt != NULL) 
                {
                   if(PTL_EVENT_ACK == *evt){  /* CHECK */   
                         if (cdesc == NULL)            
                                break;
                   }
                }
#endif
        }

        if ( cdesc && (temp_comp == cdesc)){
             if(cdesc->active == 2 || cdesc->active == 3)
                cdesc->active = 0;
                break;        
        }
        
        
   }
  return rc; 
  
}

comp_desc * get_free_comp_desc(int * comp_id)
{
    comp_desc * c;     
    c = &(_armci_portals_comp[free_desc_index]);
    while (c->active != 0){
       armci_client_complete(NULL,c->dest_id,0,c,1); 
                                  
    }
    *comp_id = free_desc_index;
#ifdef DEBUG
    printf("the value of comp_desc_id is %d\n",*comp_id);
    fflush(stdout);
#endif
    free_desc_index = ((free_desc_index + 1) % MAX_OUT);
    return c;
}


print_mem_desc(ptl_md_t * md)
{
  printf("%d:md : start %p : length %d\n",armci_me,md->start, md->length);
  fflush(stdout);
        
}

ARMCI_MEMHDL_T armci_ptl_local_mhdl;
ARMCI_MEMHDL_T *armci_portals_fill_local_mhdl(void *ptr,int bytes)
{
    armci_ptl_local_mhdl.mem_dsc.start = ptr;
    armci_ptl_local_mhdl.mem_dsc.length = bytes;
    armci_ptl_local_mhdl.mem_dsc.max_size =0;
    return(&armci_ptl_local_mhdl);
}

void armci_client_direct_get(int proc, void *src_buf, void *dst_buf, int bytes,
                             void** cptr,int tag,ARMCI_MEMHDL_T *lochdl,
                             ARMCI_MEMHDL_T *remhdl)
{
    int clus = armci_clus_id(proc);
    int rc, i;
    ptl_size_t offset_local = 0, offset_remote=0;
    ptl_match_bits_t mb = 100;
    ptl_md_t *md_remote,md, *md_local, *md_local_save;
    ptl_md_t * md_ptr;
    ptl_handle_md_t *md_hdl_local;
    comp_desc *cdesc;
    ptl_process_id_t dest_proc;
    int c_info;
    int lproc,rproc;


#ifdef DEBUG
    printf("ENTERING CLIENT GET: src_buf is %p\n, loc_hd is %p , rem_hndl is %p, BYTES = %d\n",src_buf,lochdl,remhdl,bytes);
    fflush(stdout);
#endif

    dest_proc.nid = portals->ptl_pe_procid_map[proc].nid;
    dest_proc.pid = portals->ptl_pe_procid_map[proc].pid;
  
    if(lochdl == NULL)lochdl=armci_portals_fill_local_mhdl(dst_buf,bytes);
    md_local = &lochdl->mem_dsc_save; 
    md_hdl_local = &lochdl->mem_dsc_hndl; 
    md_remote =&remhdl->mem_dsc;
#ifdef DEBUG
    printf("%d ,the value of local desc is %p and remote desc is %p\n"
                    ,portals->rank,md_local,md_remote);
    print_mem_desc(md_local);
    print_mem_desc(md_remote);
#endif

    
    offset_local = (char*)dst_buf - (char *)md_local->start;
    offset_remote = (char*)src_buf - (char *)md_remote->start;
   
#ifdef DEBUG
    printf (" src_buf %p, local_offset is %lu , local_len is %d\n", 
                    src_buf, offset_local,md_local->length );
    fflush(stdout);
    printf("rem_start  is %p, len : %d, rem_offset is %lu\n",
                    md_remote->start, md_remote->length, offset_remote);
    fflush(stdout);
    printf("\n%d:lochdlptr=%p\n",armci_me,md_hdl_local);
#endif
    
    cdesc = get_free_comp_desc(&c_info);
    if(tag) *((int *)cptr) = c_info; /*TOED*/
    if (!tag){
       cdesc->tag = tag;
       cdesc->dest_id = proc;
       cdesc->type = ARMCI_PORTALS_GET;
       cdesc->active = 0;
    }
    else{
       cdesc->tag = 999999;
       cdesc->dest_id = proc;
       cdesc->type = ARMCI_PORTALS_NBGET; 
       cdesc->active = 0;
    }

    md_local->user_ptr = (void *)cdesc;
    md_local->options =  PTL_MD_OP_GET | PTL_MD_EVENT_START_DISABLE;
#if 0
    do{
      rc = PtlMDUpdate(*md_hdl_local,NULL,md_local,portals->eq_h);
    } while (rc == PTL_MD_NO_UPDATE);
    if (rc != PTL_OK){
       printf("%d:PtlMDUpdate: %s\n", portals->rank, ptl_err_str[rc]);
       armci_die("ptlmdbind failed",0);
    }
#endif
    rc = PtlMDBind(portals->ni_h,*md_local, PTL_RETAIN, md_hdl_local);
    if (rc != PTL_OK){
       fprintf(stderr, "%d:PtlMDBind: %s\n", portals->rank, ptl_err_str[rc]);
       armci_die("ptlmdbind failed",0);
    }

    
   
    rc = PtlGetRegion(*md_hdl_local,offset_local,bytes,dest_proc,
                   portals->ptl,
                   0, 
                   mb,
                   offset_remote);
    if (rc != PTL_OK){
       printf("%d:PtlGetRegion: %s\n", portals->rank,ptl_err_str[rc]);
       armci_die("PtlGetRegion failed",0); 
    }
#ifdef DEBUG
    printf("returning from ptlgetregion call\n");
    fflush(stdout);
#endif
    
    if(!tag){ 
       armci_client_complete(NULL,proc,0,NULL,1); /* check this later */
    }
}


int armci_client_direct_send(int proc,void *src, void* dst, int bytes,  NB_CMPL_T *cmpl_info, int tag, ARMCI_MEMHDL_T *lochdl, ARMCI_MEMHDL_T *remhdl )
{
    int clus = armci_clus_id(proc);    
    int rc, i;
    ptl_size_t offset_local = 0, offset_remote = 0;
    ptl_match_bits_t mb = 100;
    ptl_md_t *md_remote,md, *md_local;
    ptl_md_t * md_ptr;
    ptl_match_bits_t * mb_ptr;
    ptl_handle_md_t *md_hdl_local;
    comp_desc *cdesc;
    ptl_process_id_t dest_proc;
    int c_info;
    int lproc,rproc;

    /*if (PtlGetRankId(proc, &dest_proc) != PTL_OK) {
        fprintf(stderr, "%d:PtlGetRankId() failed\n", portals->rank);
        exit(EXIT_FAILURE);
    }*/
    dest_proc.nid = portals->ptl_pe_procid_map[proc].nid;
    dest_proc.pid = portals->ptl_pe_procid_map[proc].pid;
    
    if(lochdl == NULL)lochdl=armci_portals_fill_local_mhdl(src,bytes);
    md_local = &lochdl->mem_dsc_save; 
    md_hdl_local = &lochdl->mem_dsc_hndl; 
    md_remote =&remhdl->mem_dsc;
    
    offset_local = (char *)src - (char *)md_local->start;
    offset_remote =(char *)dst - (char *)md_remote->start;

    cdesc = get_free_comp_desc(&c_info);
    
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
    md_local->options =  PTL_MD_OP_PUT | PTL_MD_EVENT_START_DISABLE;
    

    rc = PtlMDBind(portals->ni_h,*md_local, PTL_RETAIN, md_hdl_local);
    if (rc != PTL_OK){
       fprintf(stderr, "%d:PtlMDBind: %s\n", portals->rank, ptl_err_str[rc]);
       armci_die("ptlmdbind failed",0);
    }
    
    rc = PtlPutRegion(*md_hdl_local,offset_local,bytes,PTL_ACK_REQ,dest_proc,
                   portals->ptl,
                   0, mb,offset_remote, 0);
    if (rc != PTL_OK){
       /* fprintf(stderr, "%d:PtlPutRegion: %s\n", portals->rank,PtlErrorStr(rc));*/
       exit(EXIT_FAILURE); 
    }

#ifdef DEBUG
    printf("returning from ptlputregion call\n");
    fflush(stdout);
#endif
    
    armci_update_fence_array(proc, 1);
    if(!tag){
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
   rc = armci_client_complete(NULL,proc,tag,NULL,0);
   return rc;
}

void armci_network_client_deregister_memory(ARMCI_MEMHDL_T *mh)
{
}


void armci_network_server_deregister_memory(ARMCI_MEMHDL_T *mh)
{
}

#ifdef CATAMOUNT
void armcill_allocate_locks(int numlocks)
{
}
void armcill_lock(int mutex, int proc)
{
int desc,node = armci_clus_id(proc);
int off;
    off = _mutex_array[node].off + mutex*sizeof(int);
    desc = _mutex_array[node].desc;
}


/*\ unlock specified mutex on node where process proc is running
\*/
void armcill_unlock(int mutex, int proc)
{
int desc,node = armci_clus_id(proc);
int off;
    off = _mutex_array[node].off + mutex*sizeof(int);
    desc = _mutex_array[node].desc;
}
#endif

