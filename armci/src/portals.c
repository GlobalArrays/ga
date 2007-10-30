/* preliminary implementation on top of portals */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include "armcip.h"
#include <stdint.h>

/*#define DEBUG*/
#ifdef XT3
#include "locks.h"
typedef struct {
       int off;
       int desc;
} cnos_mutex_t;

static cnos_mutex_t *_mutex_array;
#endif

/*this should match regions.c*/
#define MAX_REGIONS 8

/*global variables and data structures */
armci_portals_proc_t _armci_portals_proc_struct;
armci_portals_proc_t *portals = &_armci_portals_proc_struct;
comp_desc *_region_compdesc_array[MAX_REGIONS+1];
int ptl_initialized = 0;
int free_desc_index[MAX_REGIONS+1];
FILE *utcp_lib_out;
FILE* utcp_api_out;

int armci_init_portals(void)
{
    int num_interface;
    int rc;
    int npes,i;
    comp_desc *armci_comp_desc;
   
    if (PtlInit(&num_interface) != PTL_OK) {
       fprintf(stderr, "PtlInit() failed\n");
       exit(1);
    }
    portals->ptl = 44;
    for(i=0;i<=MAX_REGIONS;i++){
      free_desc_index[i]=0;
    }

    if((rc=PtlNIInit(IFACE_FROM_BRIDGE_AND_NALID(PTL_DEFAULT_BRIDGE,PTL_IFACE_DEFAULT), PTL_PID_ANY, NULL, NULL, &(portals->ni_h))) != PTL_OK){
#ifdef XT3
       if(rc!=PTL_IFACE_DUP)
#endif
       {
       printf( "PtlNIInit() failed %d\n",rc);
       armci_die("NIInit Failed",0);
       }
    }

#ifdef DEBUG
    PtlNIDebug(portals->ni_h,PTL_DEBUG_ALL);
#endif

    portals->size=cnos_get_n_pes_in_app();
    PtlGetId(portals->ni_h,&portals->ptl_my_procid);
#ifdef DEBUG
    printf("%d:the rank is %d, size is %d\n",armci_me,portals->ptl_my_procid,portals->size);
#endif
    if ((npes = cnos_get_nidpid_map(&portals->ptl_pe_procid_map)) == -1) {
      printf(" LIBSMA ERROR:Getting proc id/PE map failed (npes=%d)\n", npes);
    }
#ifdef DEBUG
    for (i = 0; i < npes; i++) {
      printf("%d:PE %d is 0x%lx/%d (nid/pid)\n", portals->ptl_my_procid,i,portals->ptl_pe_procid_map[i].nid,portals->ptl_pe_procid_map[i].pid);
    }
#endif

    /* Allocate one shared event queue for all operations 
     * TODO tune size.
     */

    rc = PtlEQAlloc(portals->ni_h,1024,NULL, &(portals->eq_h));
    if (rc != PTL_OK) {
       printf("%d:PtlEQAlloc() failed: %s (%d)\n",
                            portals->ptl_my_procid, ptl_err_str[rc], rc);
       exit(EXIT_FAILURE);                            
    }
    ptl_initialized = 1;
    portals->num_match_entries = 0;

#ifndef XT3
    utcp_lib_out = stdout;
    utcp_api_out = stdout;
#endif

    fflush(stdout); 
    /*now prepare for use of local memory*/ 
    armci_comp_desc = (comp_desc *)malloc(sizeof(comp_desc)*MAX_OUT); 
    for(i=0; i< MAX_OUT;i++){
      ptl_md_t *md_ptr;
      ptl_handle_md_t *md_h;
      armci_comp_desc[i].active=0;
      md_ptr = &armci_comp_desc[i].mem_dsc;
      md_h = &armci_comp_desc[i].mem_dsc_hndl;
      md_ptr->eq_handle = portals->eq_h;
      md_ptr->max_size =0;
      md_ptr->threshold = 2;
      md_ptr->options =  PTL_MD_OP_GET | PTL_MD_OP_PUT | PTL_MD_EVENT_START_DISABLE;
    }
    _region_compdesc_array[MAX_REGIONS]=armci_comp_desc;
    return 0;   
}



void armci_fini_portals()
{
#ifdef DEBUG
    printf("ENTERING ARMCI_FINI_PORTALS\n");fflush(stdout);    
#endif
    PtlNIFini(portals->ni_h);
    PtlFini();
#ifdef DEBUG
    printf("LEAVING ARMCI_FINI_PORTALS\n");fflush(stdout);    
#endif
    
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
    int rc,i;
    void * context;
    ptl_md_t *md_ptr;
    ptl_match_bits_t *mb;
    ptl_process_id_t match_id;
    ptl_handle_md_t *md_h;
    comp_desc *armci_comp_desc;

#ifdef DEBUG
    printf("inside portals.c : size of mem_hndl is %d\n", sizeof(region_memhdl_t));
    printf("\n%d:armci_pin_contig_hndl start=%p bytes=%d",armci_me,start,bytes);fflush(stdout);
#endif
   /*first create comp_desc arr for this region if it is not local*/
    if(!reg_mem->islocal){
      armci_comp_desc = (comp_desc *)malloc(sizeof(comp_desc)*MAX_OUT); 
      for(i=0; i< MAX_OUT;i++){
        armci_comp_desc[i].active=0;
        md_ptr = &armci_comp_desc[i].mem_dsc;
        md_h = &armci_comp_desc[i].mem_dsc_hndl;
        context = NULL;
        md_ptr->start = start;
        md_ptr->length = bytes;
        md_ptr->threshold = 2;
        md_ptr->options =  PTL_MD_OP_GET | PTL_MD_OP_PUT | PTL_MD_EVENT_START_DISABLE;
        /*md_ptr->options =  PTL_MD_EVENT_START_DISABLE;*/
                      
        md_ptr->user_ptr = context;
        md_ptr->eq_handle = portals->eq_h;
        md_ptr->max_size =0;
#ifdef DEBUG
        printf("\n%d:lochdl=%p",armci_me,md_h);
#endif
#if 0 
        rc = PtlMDBind(portals->ni_h,*md_ptr, PTL_UNLINK, md_h);
        if (rc != PTL_OK){
          printf("%d:PtlMDBind: %s\n", portals->ptl_my_procid, ptl_err_str[rc]);
          armci_die("ptlmdbind failed",0);
        }
#endif
      }
      _region_compdesc_array[reg_mem->regid]=armci_comp_desc;
      return 1;
    }
    else {
      md_ptr = &reg_mem->mem_dsc;
      md_h = &reg_mem->mem_dsc_hndl;
      context = NULL;
      md_ptr->start = start;
      md_ptr->length = bytes;
      md_ptr->threshold = 2;
      md_ptr->options =  PTL_MD_OP_GET | PTL_MD_EVENT_START_DISABLE;
      /*md_ptr->options =  PTL_MD_EVENT_START_DISABLE;*/
                      
      md_ptr->user_ptr = context;
      md_ptr->eq_handle = portals->eq_h;
      md_ptr->max_size =0;
#if 0 
      rc = PtlMDBind(portals->ni_h,*md_ptr, PTL_RETAIN, md_h);
      if (rc != PTL_OK){
         printf("%d:PtlMDBind: %s\n", portals->ptl_my_procid, ptl_err_str[rc]);
         armci_die("ptlmdbind failed",0);
      }
#endif
      reg_mem->mem_dsc_save=reg_mem->mem_dsc;
      return 1;
    }
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
#ifdef DEBUG
       printf("\n%d:done waiting type=%d\n",armci_me,ev->type);
       fflush(stdout);
#endif
    
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
         temp_comp = (comp_desc *)ev->md.user_ptr;
         temp_tag = temp_comp->tag;
         temp_proc = temp_comp->dest_id;
         continue;
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
         if ((nb_tag != 0) && (nb_tag == temp_tag)){
           temp_comp->tag=-1;
           break;
         }
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
        }
        if ( cdesc && (temp_comp == cdesc)){
             if(cdesc->active == 2 || cdesc->active == 3)
                cdesc->active = 0;
                temp_comp->tag=-1;
                break;        
        }
        
        
   }
  return rc; 
  
}

comp_desc * get_free_comp_desc(int region_id, int * comp_id)
{
    comp_desc * c;     
    c = &(_region_compdesc_array[region_id][free_desc_index[region_id]]);
    while (c->active != 0){
       armci_client_complete(NULL,c->dest_id,0,c,1); 
                                  
    }
    *comp_id = (region_id*MAX_REGIONS+free_desc_index[region_id]);
#ifdef DEBUG
    printf("the value of comp_desc_id is %d\n",*comp_id);
    fflush(stdout);
#endif
    free_desc_index[region_id] = ((free_desc_index[region_id] + 1) % MAX_OUT);
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
    void * context=NULL;
    armci_ptl_local_mhdl.mem_dsc.start = ptr;
    armci_ptl_local_mhdl.mem_dsc.length = bytes;
    armci_ptl_local_mhdl.mem_dsc.max_size =0;
    armci_ptl_local_mhdl.mem_dsc.options =  PTL_MD_OP_GET | PTL_MD_EVENT_START_DISABLE;
    armci_ptl_local_mhdl.mem_dsc.user_ptr = context;
    armci_ptl_local_mhdl.mem_dsc.eq_handle = portals->eq_h;
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
    int lproc,rproc,user_memory=0;

#if 0
    printf("ENTERING CLIENT GET: src_buf is %p dstbuf=%p \n, loc_hd is %p , rem_hndl is %p, BYTES = %d\n",src_buf,dst_buf,lochdl,remhdl,bytes);
    fflush(stdout);
#endif

    /*first process information*/
    dest_proc.nid = portals->ptl_pe_procid_map[proc].nid;
    dest_proc.pid = portals->ptl_pe_procid_map[proc].pid;
    md_remote =&remhdl->mem_dsc;
    /*updating md to send*/

    /*there are 3 kinds of ARMCI memory: ARMCI_Malloc, ARMCI_Malloc_local, user
     * allocated memory. For ARMCI_Malloc, we use region specific md that
     * comes from completion descriptor.
     * For ARMCI_Malloc_local, we use the MD from the lochdl
     * For user allocated memory, we use armci_portals_fill_local_mhdl
     * which binds the user memory. We never keep track of non-armci allocated
     * memory.
     */
    if(lochdl == NULL){ /*this is user memory*/
      user_memory=1;
      cdesc = get_free_comp_desc(MAX_REGIONS,&c_info);
      md_local = &cdesc->mem_dsc;
      md_hdl_local = &cdesc->mem_dsc_hndl; 
      md_local->length=bytes;
      md_local->start=dst_buf;
    }
    else {
      if(lochdl->islocal){ /*ARMCI_Malloc_local memory*/
        md_local = &lochdl->mem_dsc_save; 
        md_hdl_local = &lochdl->mem_dsc_hndl; 
      }
      else{
        /*we need to pass region id to get corresponding md*/
        cdesc = get_free_comp_desc(lochdl->regid,&c_info);
        md_local = &cdesc->mem_dsc;
        md_hdl_local = &cdesc->mem_dsc_hndl; 
      }
    }

    offset_local = (char*)dst_buf - (char *)md_local->start;
    offset_remote = (char*)src_buf - (char *)md_remote->start;

    /*we only need md_remote for computing remote offset, this can be changed*/
    /*compute the local and remote offsets*/ 
   
#ifdef DEBUG
    print_mem_desc(md_local);
    print_mem_desc(md_remote);
    printf("%d ,the value of local desc is %p and remote desc is %p\n"
                    ,portals->rank,md_local,md_remote);
    printf (" src_buf %p, local_offset is %lu , local_len is %d\n" 
                    "rem_start  is %p, len : %d, rem_offset is %lu\n",
                    src_buf, offset_local,md_local->length,md_remote->start, 
                    md_remote->length, offset_remote);
    fflush(stdout);
#endif
    
    if(tag) *((int *)cptr) = c_info; /*TOED*/
     /*printf("\n%d:tag=%d c_info=%d",armci_me,tag,c_info);fflush(stdout);*/
    if (tag){
       cdesc->tag = tag;
       cdesc->dest_id = proc;
       cdesc->type = ARMCI_PORTALS_NBGET;
       cdesc->active = 0;
    }
    else{
       cdesc->tag = 999999;
       cdesc->dest_id = proc;
       cdesc->type = ARMCI_PORTALS_GET; 
       cdesc->active = 0;
    }
    md_local->user_ptr = (void *)cdesc;
    md_local->options =  PTL_MD_OP_GET | PTL_MD_EVENT_START_DISABLE;
#if 0 /*MDUpdate doesn't do what it is supposed to*/
    if(user_memory==0){
      do{
        rc = PtlMDUpdate(*md_hdl_local,NULL,md_local,portals->eq_h);
        printf("\n%d:trying to update\n",armci_me);fflush(stdout);
      } while (rc == PTL_MD_NO_UPDATE);
      if (rc != PTL_OK){
         printf("%d:PtlMDUpdate: %s\n", portals->rank, ptl_err_str[rc]);
         armci_die("ptlmdbind failed",0);
      }
    }
    else{
#endif
#ifdef DEBUG
      printf("\n%d:binding\n",armci_me);
#endif
      rc = PtlMDBind(portals->ni_h,*md_local, PTL_UNLINK, md_hdl_local);
      if (rc != PTL_OK){
         fprintf(stderr, "%d:PtlMDBind: %s\n", portals->rank, ptl_err_str[rc]);
         armci_die("ptlmdbind failed",0);
      }
#if 0 
    }
#endif
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


int armci_client_direct_send(int proc,void *src, void* dst, int bytes,  void **cptr, int tag, ARMCI_MEMHDL_T *lochdl, ARMCI_MEMHDL_T *remhdl )
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
    int lproc,rproc,user_memory=0;
    /*if (PtlGetRankId(proc, &dest_proc) != PTL_OK) {
        fprintf(stderr, "%d:PtlGetRankId() failed\n", portals->rank);
        exit(EXIT_FAILURE);
    }*/
    dest_proc.nid = portals->ptl_pe_procid_map[proc].nid;
    dest_proc.pid = portals->ptl_pe_procid_map[proc].pid;
    md_remote =&remhdl->mem_dsc;

    if(lochdl == NULL){ /*this is user memory*/
      user_memory=1;
      cdesc = get_free_comp_desc(MAX_REGIONS,&c_info);
      md_local = &cdesc->mem_dsc;
      md_hdl_local = &cdesc->mem_dsc_hndl; 
      md_local->length=bytes;
      md_local->start=src;
#ifdef DEBUG
      printf("\n%d:here\n",armci_me);
#endif
    }
    else {
      if(lochdl->islocal){ /*ARMCI_Malloc_local memory*/
        md_local = &lochdl->mem_dsc_save; 
        md_hdl_local = &lochdl->mem_dsc_hndl; 
      }
      else{
        /*we need to pass region id to get corresponding md*/
        cdesc = get_free_comp_desc(lochdl->regid,&c_info);
        md_local = &cdesc->mem_dsc;
        md_hdl_local = &cdesc->mem_dsc_hndl; 

      }
    }
    
    
    offset_local = (char *)src - (char *)md_local->start;
    offset_remote =(char *)dst - (char *)md_remote->start;


    if(tag) *((int *)cptr) = c_info; /*TOED*/

    if (tag){
       cdesc->tag = tag;
       cdesc->dest_id = proc;
       cdesc->type = ARMCI_PORTALS_NBPUT;
       cdesc->active = 0;
    }
    else{
       cdesc->tag = 999999;
       cdesc->dest_id = proc;
       cdesc->type = ARMCI_PORTALS_PUT; 
       cdesc->active = 0;
    }

    md_local->user_ptr = (void *)cdesc;
    md_local->options =  PTL_MD_OP_PUT | PTL_MD_EVENT_START_DISABLE;
#if 0
    if(user_memory==0){
      do{
        rc = PtlMDUpdate(*md_hdl_local,NULL,md_local,portals->eq_h);
      } while (rc == PTL_MD_NO_UPDATE);
      if (rc != PTL_OK){
         printf("%d:PtlMDUpdate: %s\n", portals->rank, ptl_err_str[rc]);
         armci_die("ptlmdbind failed",0);
      }
    }
    else{
#endif
      rc = PtlMDBind(portals->ni_h,*md_local, PTL_UNLINK, md_hdl_local);
      if (rc != PTL_OK){
         fprintf(stderr, "%d:PtlMDBind: %s\n", portals->rank, ptl_err_str[rc]);
         armci_die("ptlmdbind failed",0);
      }
#if 0
    }
#endif
    
    rc = PtlPutRegion(*md_hdl_local,offset_local,bytes,PTL_ACK_REQ,dest_proc,
                   portals->ptl,
                   0, mb,offset_remote, 0);
    if (rc != PTL_OK){
       fprintf(stderr, "%d:PtlPutRegion: %s\n", portals->rank,ptl_err_str[rc]);
       armci_die("PtlPutRegion failed",0);
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
   int rc,roff,rid;
   int proc,rinfo=*((int *)cmpl_info);
   comp_desc * c;     
   /*TOED*/
   printf("\n%d:portals_complete\n",armci_me);
   roff=rinfo%MAX_REGIONS;
   rid =(rinfo-roff)/MAX_REGIONS; 
   c = &(_region_compdesc_array[rid][roff]);
   rc = armci_client_complete(NULL,proc,tag,c,0);
   return rc;
}

void armci_network_client_deregister_memory(ARMCI_MEMHDL_T *mh)
{
}


void armci_network_server_deregister_memory(ARMCI_MEMHDL_T *mh)
{
}

#ifdef XT3
void armcill_allocate_locks(int numlocks)
{
#if 0
    int ace_any=1;
    int rc;
    rc = PtlACEntry(portals->ni_h, ace_any,
                    (ptl_process_id_t){PTL_NID_ANY, PTL_PID_ANY},
                    PTL_UID_ANY, PTL_JID_ANY, PTL_PT_INDEX_ANY);
    if (rc != PTL_OK) {
      printf("%d: PtlACEntry() failed: %s\n",
           rank, PtlErrorStr(rc));
      armci_die("PtlACEntry failed",0);
    }
    md_lock.start = &lock;
    md_lock.length = sizeof(lock);
    md_lock.threshold = PTL_MD_THRESH_INF;
    md_lock.options =
                PTL_MD_OP_PUT | PTL_MD_OP_GET |
                PTL_MD_MANAGE_REMOTE | PTL_MD_TRUNCATE |
                PTL_MD_EVENT_START_DISABLE;
    md_lock.max_size = 0;
    md_lock.user_ptr = NULL;
    md_lock.eq_handle = portals->eq_h;

    /* Lockmaster needs a match entry for clients to access lock value. 
    */
    rc = PtlMEAttach(ni_h, ptl,
                         any_id,        /* source address */
                         lock_mbits,    /* expected match bits */
                         ibits,         /* ignore bits to mask */
                         PTL_UNLINK,    /* unlink when md is unlinked */
                         PTL_INS_AFTER,
                         &me_lock_h);
    if (rc != PTL_OK) {
      printf("%d: PtlMEAttach(): %s\n",
                        rank, PtlErrorStr(rc));
      armci_die("PtlMEAttach in int_locks failed",0);
    }
    rc = PtlMDAttach(me_lock_h, md_lock, PTL_UNLINK, &md_lock_h);
    if (rc != PTL_OK) {
      printf("%d: PtlMDAttach(): %s\n",
                        rank, PtlErrorStr(rc));
      armci_die("PtlMDAttach in int_locks failed",0);
    }
#endif

}
void armcill_lock(int mutex, int proc)
{
int desc,node = armci_clus_id(proc);
int off;
}


/*\ unlock specified mutex on node where process proc is running
\*/
void armcill_unlock(int mutex, int proc)
{
int desc,node = armci_clus_id(proc);
int off;
}
#endif

