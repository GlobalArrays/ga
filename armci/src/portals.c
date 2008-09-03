/* $Id$ */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <float.h>
#include "armcip.h"
#include "message.h"
#include <stdint.h>
#include <assert.h>

#define DEBUG_COMM 0
#define DEBUG_INIT 0

#define ARMCI_PORTALS_MAX_LOCKS 16

#ifdef XT3
#include "locks.h"
typedef struct {
       int off;
       int desc;
} cnos_mutex_t;
static cnos_mutex_t *_mutex_array;
#endif

static int num_locks=0;
static long **all_locks;
static long a_p_putfrom, a_p_getinto;
ptl_md_t _armci_portals_lock_md;
ptl_handle_md_t _armci_portals_lock_md_h;
ptl_handle_me_t _armci_portals_lock_me_h;
comp_desc *_lockput_cd_array;
comp_desc *_lockget_cd_array;

typedef struct {
       void *base_ptr;
       size_t size;
       int islocal;
}aptl_reginfo_t;

typedef struct {
       aptl_reginfo_t reginfo[MAX_MEM_REGIONS];  
       int reg_count;
} rem_meminfo_t;

/*global variables and data structures */
static armci_portals_proc_t _armci_portals_proc_struct;
static armci_portals_serv_t _armci_portals_serv_struct;
static armci_portals_proc_t *portals = &_armci_portals_proc_struct;
static armci_portals_serv_t *serv_portals = &_armci_portals_serv_struct;
/*static */comp_desc _compdesc_array[NUM_COMP_DSCR];

static rem_meminfo_t *_rem_meminfo;
static aptl_reginfo_t *_tmp_rem_reginfo;

#define IN_REGION(_ptr__,_reg__) ((_ptr__)>=(_reg__.base_ptr) \
                && (_ptr__) < ( (char *)(_reg__.base_ptr)+_reg__.size))

static int ptl_initialized = 0;

ptl_ni_limits_t armci_ptl_nilimits;

int armci_init_portals(void)
{
int num_interface;
int rc;
int npes,i;
    ARMCI_PR_DBG("enter",0);
   
    rc = PtlInit(&num_interface);
    if (rc != PTL_OK) {
       printf("PtlInit() failed %d %s\n",rc, ARMCI_NET_ERRTOSTR(rc) );
       armci_die("PtlInit Failed",rc);
    }

    /*initialize data structures*/
    bzero(portals,sizeof(armci_portals_proc_t));
    bzero(serv_portals,sizeof(armci_portals_serv_t));

    _rem_meminfo = (rem_meminfo_t *)malloc(sizeof(rem_meminfo_t)*armci_nproc);
    _tmp_rem_reginfo = (aptl_reginfo_t *)malloc(sizeof(aptl_reginfo_t)*armci_nproc);
    if( _rem_meminfo==NULL || _tmp_rem_reginfo ==NULL)
      armci_die("malloc failed in init_portals",0);

    portals->ptl = ARMCI_PORTALS_PTL_NUMBER; /* our own ptl number */

    rc=PtlNIInit(IFACE_FROM_BRIDGE_AND_NALID(PTL_BRIDGE_UK,PTL_IFACE_SS), PTL_PID_ANY, NULL, &armci_ptl_nilimits, &(portals->ni_h));
    switch(rc) {
       case PTL_OK:
       case PTL_IFACE_DUP:
         break;
       default:
         printf( "PtlNIInit() failed %d error=%s\n",rc,ARMCI_NET_ERRTOSTR(rc) );
         armci_die("NIInit Failed",0);
    }
    if(DEBUG_INIT || DEBUG_COMM)
      PtlNIDebug(portals->ni_h,PTL_DEBUG_ALL);

    PtlGetId(portals->ni_h,&portals->rank);
    if(DEBUG_INIT){
      printf("%d:the rank is %d, size is %d\n",armci_me,
                      portals->rank,armci_nproc);
    }

    if ((npes = cnos_get_nidpid_map(&portals->ptl_pe_procid_map)) == -1) {
      printf(" CNOS ERROR:Getting proc id/PE map failed (npes=%d)\n", npes);
    }

    /* Allocate one shared event queue for all operations 
     * DOTO tune size. Each comp desc may generate at most 3 events, there
     * are NUM_COMP_DSCR comp desc's. 16 times that is a safe estimate for
     * size of the queue.
     */
    rc = PtlEQAlloc(portals->ni_h,16*NUM_COMP_DSCR,NULL, &(portals->eq_h));
    if (rc != PTL_OK) {
       printf("%d:PtlEQAlloc() failed: %s (%d)\n",
                            portals->rank, ARMCI_NET_ERRTOSTR(rc) , rc);
      armci_die("EQ Alloc failed",rc);
    }

    for(i=0;i<NUM_COMP_DSCR;i++){
      _compdesc_array[i].active=0;
      _compdesc_array[i].tag=-1;
      _compdesc_array[i].dest_id=-1;
      _compdesc_array[i].mem_dsc.eq_handle=portals->eq_h;
      _compdesc_array[i].mem_dsc.max_size=0;
      _compdesc_array[i].mem_dsc.threshold=2;
      _compdesc_array[i].mem_dsc.options=PTL_MD_OP_GET | PTL_MD_OP_PUT | PTL_MD_EVENT_START_DISABLE;
    }

    ptl_initialized = 1;
    portals->free_comp_desc_index=0;
    
    ARMCI_PR_DBG("exit",0);
    return 0;   
}


void armci_fini_portals()
{
    ARMCI_PR_DBG("enter",0);
    if(DEBUG_INIT){
      printf("ENTERING ARMCI_FINI_PORTALS\n");fflush(stdout);
    }    
    PtlNIFini(portals->ni_h);
    /*PtlFini();*/
    if(DEBUG_INIT){
      printf("LEAVING ARMCI_FINI_PORTALS\n");fflush(stdout);    
    }
    ARMCI_PR_DBG("exit",0);
}


void armci_pin_contig1(void *start, size_t bytes)
{
int rc;
ptl_md_t *md_ptr;
ptl_match_bits_t ignbits = 0xFFFFFFFFFFFFFF00;
ptl_process_id_t match_id;

    ARMCI_PR_DBG("enter",serv_portals->reg_count);

    if(DEBUG_COMM){
      printf("\n%d:armci_pin_contig1 start=%p bytes=%ld\n",
                      armci_me,start,bytes);fflush(stdout);
    }

    md_ptr            = &(serv_portals->meminfo[serv_portals->reg_count].md);
    md_ptr->start     = start;
    md_ptr->length    = bytes;
    md_ptr->threshold = PTL_MD_THRESH_INF;
    md_ptr->options   =  PTL_MD_OP_PUT | PTL_MD_OP_GET | PTL_MD_MANAGE_REMOTE;
    md_ptr->user_ptr  = NULL;
    md_ptr->max_size  = 0;
    md_ptr->eq_handle = PTL_EQ_NONE;

    serv_portals->meminfo[serv_portals->reg_count].mb=serv_portals->reg_count;
 
    match_id.nid = PTL_NID_ANY;
    match_id.pid = PTL_PID_ANY; 

    rc = PtlMEAttach(portals->ni_h,portals->ptl,match_id,
                    serv_portals->meminfo[serv_portals->reg_count].mb,
                    ignbits,
		    PTL_RETAIN,PTL_INS_AFTER,
		    &(serv_portals->meminfo[serv_portals->reg_count].me_h)); 
    if (rc != PTL_OK) {
      printf("%d:PtlMEAttach: %s\n", portals->rank, ARMCI_NET_ERRTOSTR(rc) );
      armci_die("portals attach error2",rc);
    }

    rc = PtlMDAttach((serv_portals->meminfo[serv_portals->reg_count].me_h),
                    *md_ptr,PTL_RETAIN,
                    &(serv_portals->meminfo[serv_portals->reg_count].md_h));
    if (rc != PTL_OK) {
      printf("%d:PtlMDAttach: %s\n", portals->rank, ARMCI_NET_ERRTOSTR(rc) );
      armci_die("portals attach error1",rc);
    }
    serv_portals->reg_count++;     

    ARMCI_PR_DBG("exit",serv_portals->reg_count);
}

void armci_pin_contig_lock(void *start, size_t bytes)
{
int rc;
ptl_md_t *md_ptr;
ptl_match_bits_t ignbits = 0xFFFFFFFFFFFFFF00;
ptl_process_id_t match_id;

    ARMCI_PR_DBG("enter",0);

    if(DEBUG_COMM){
      printf("\n%d:armci_pin_contig_lock start=%p bytes=%ld\n",
                      armci_me,start,bytes);fflush(stdout);
    }

    md_ptr            = &(_armci_portals_lock_md);
    md_ptr->start     = start;
    md_ptr->length    = bytes;
    md_ptr->threshold = PTL_MD_THRESH_INF;
    md_ptr->options   =  PTL_MD_OP_PUT | PTL_MD_OP_GET | PTL_MD_MANAGE_REMOTE;
    md_ptr->user_ptr  = NULL;
    md_ptr->max_size  = 0;
    md_ptr->eq_handle = PTL_EQ_NONE;

    match_id.nid = PTL_NID_ANY;
    match_id.pid = PTL_PID_ANY; 

    rc = PtlMEAttach(portals->ni_h,portals->ptl,match_id,
                    MAX_MEM_REGIONS+1,
                    ignbits,
                    PTL_RETAIN,PTL_INS_AFTER,
                    &(_armci_portals_lock_me_h)); 
    if (rc != PTL_OK) {
      printf("%d:PtlMEAttach: %s\n", portals->rank, ARMCI_NET_ERRTOSTR(rc) );
      armci_die("portals attach error2",rc);
    }

    rc = PtlMDAttach(_armci_portals_lock_me_h,
                    *md_ptr,PTL_RETAIN,
                    &_armci_portals_lock_md_h);
    if (rc != PTL_OK) {
      printf("%d:PtlMDAttach: %s\n", portals->rank, ARMCI_NET_ERRTOSTR(rc) );
      armci_die("portals attach error1",rc);
    }

    ARMCI_PR_DBG("exit",serv_portals->reg_count);
}

void armci_serv_register_req(void *start,int bytes, int ID)
{
int rc;
ptl_md_t *md_ptr;
ptl_match_bits_t ignbits;
ptl_process_id_t match_id;

    ARMCI_PR_DBG("enter",serv_portals->reg_count);

    if(DEBUG_COMM){
      printf("\n%d:armci_serv_register_req start=%p bytes=%d\n",
                      armci_me,start,bytes);fflush(stdout);
    }

    md_ptr            = &(serv_portals->meminfo[serv_portals->reg_count].md);
    md_ptr->start     = start;
    md_ptr->length    = bytes;
    md_ptr->threshold = PTL_MD_THRESH_INF;
    md_ptr->options   =  PTL_MD_OP_PUT | PTL_MD_OP_GET | PTL_MD_MANAGE_REMOTE;
    md_ptr->user_ptr  = NULL;
    md_ptr->max_size  = 0;
    md_ptr->eq_handle = PTL_EQ_NONE;

    serv_portals->meminfo[serv_portals->reg_count].mb=serv_portals->reg_count;
 
    match_id.nid = PTL_NID_ANY;
    match_id.pid = PTL_PID_ANY; 

    rc = PtlMEAttach(portals->ni_h,portals->ptl,match_id,
                    serv_portals->meminfo[serv_portals->reg_count].mb,
                    ignbits,
		    PTL_RETAIN,PTL_INS_AFTER,
		    &(serv_portals->meminfo[serv_portals->reg_count].me_h)); 
    if (rc != PTL_OK) {
      printf("%d:PtlMEAttach: %s\n", portals->rank, ARMCI_NET_ERRTOSTR(rc) );
      armci_die("portals attach error2",rc);
    }

    rc = PtlMDAttach((serv_portals->meminfo[serv_portals->reg_count].me_h),
                    *md_ptr,PTL_RETAIN,
                    &serv_portals->meminfo[serv_portals->reg_count].md_h);
    if (rc != PTL_OK) {
      printf("%d:PtlMDAttach: %s\n", portals->rank, ARMCI_NET_ERRTOSTR(rc) );
      armci_die("portals attach error1",rc);
    }
    serv_portals->reg_count++;     

    ARMCI_PR_DBG("exit",serv_portals->reg_count);
 
}

int armci_client_complete(ptl_event_kind_t *evt,int proc_id, int nb_tag,
                comp_desc *cdesc)
{
int rc;  
ptl_event_t ev_t;
ptl_event_t *ev=&ev_t;
comp_desc *temp_comp = NULL;
int loop=1;
int temp_proc;
    ARMCI_PR_DBG("enter",0);
    if(DEBUG_COMM){
      printf("\n%d:enter:client_complete active=%d tag=%d %d\n",armci_me,
                      cdesc->active,cdesc->tag,nb_tag);fflush(stdout);
    }
    
    while(cdesc->active!=0)
    {
      ev->type=0;
      if((rc = PtlEQWait(portals->eq_h, ev)) != PTL_OK){
        printf("%d:PtlEQWait(): %d %s\n", portals->rank,rc,
                        ARMCI_NET_ERRTOSTR(rc) ); 
        armci_die("EQWait problem",rc);
      }
      if (ev->ni_fail_type != PTL_NI_OK) {
        printf("%d:NI sent %d in event.\n",
                        portals->rank,  ev->ni_fail_type); 
        armci_die("event failure problem",0);
      }
      if(DEBUG_COMM){
        printf("\n%d:armci_client_complete:done waiting type=%d\n",armci_me,
                        ev->type);
        fflush(stdout);
      }
      
      if (ev->type == PTL_EVENT_SEND_END){
        if(DEBUG_COMM){
          printf("\n%d:armci_client_complete:event send end\n",armci_me);
          fflush(stdout);
        }
        temp_comp = (comp_desc *)ev->md.user_ptr;
        if(temp_comp->type==ARMCI_PORTALS_GETPUT ||
           temp_comp->type==ARMCI_PORTALS_NBGETPUT)
        {
          temp_comp->active=0;
          temp_comp->tag=-1;
          continue;
        }
#ifdef PUT_LOCAL_ONLY_COMPLETION
        if(temp_comp->type==ARMCI_PORTALS_PUT ||
           temp_comp->type==ARMCI_PORTALS_NBPUT)
        {
          temp_comp->active=0;
          temp_comp->tag=-1;
        }
        else
#else
        temp_comp->active++;
#endif
        continue;
      }

      else if (ev->type == PTL_EVENT_REPLY_END){
        if(DEBUG_COMM){
          printf("\n%d:client_send_complete:reply end\n",armci_me);
          fflush(stdout);
        }
        temp_comp = (comp_desc *)ev->md.user_ptr;
        temp_comp->active = 0; /*this was a get request, so we are done*/
        temp_comp->tag=-1;
        continue;
      }
      else if (ev->type == PTL_EVENT_ACK){
        if(DEBUG_COMM){
          printf("\n%d:client_send_complete:event ack\n",armci_me);
          fflush(stdout);
        }
        temp_comp = (comp_desc *)ev->md.user_ptr;
        temp_comp->active=0;
        temp_comp->tag=-1;
        armci_update_fence_array(temp_comp->dest_id,0);              
        portals->outstanding_puts--; 
      }
      else
         armci_die("armci_client_complete: unknown event",ev->type);
    }
    if(DEBUG_COMM){
      printf("\n%d:exit:client_complete active=%d tag=%d %d\n",armci_me,
                      cdesc->active,cdesc->tag,nb_tag);fflush(stdout);
    }

    ARMCI_PR_DBG("exit",0);

    return rc; 
}


comp_desc * get_free_comp_desc(int * comp_id)
{
comp_desc * c;     
int rc = PTL_OK;

    ARMCI_PR_DBG("enter",0);

    c = &(_compdesc_array[portals->free_comp_desc_index]);
    if(c->active!=0 && c->tag>0)
      armci_client_complete(NULL,c->dest_id,c->tag,c);

#ifdef PUT_LOCAL_ONLY_COMPLETION
    do {
       rc = PtlMDUnlink(c->mem_dsc_hndl);
    }while(rc==PTL_MD_IN_USE);
#endif

    *comp_id = portals->free_comp_desc_index;
    if(DEBUG_COMM){
      printf("the value of comp_desc_id is %d\n",*comp_id);
      fflush(stdout);
    }
    portals->free_comp_desc_index = (portals->free_comp_desc_index+1) % NUM_COMP_DSCR;

    ARMCI_PR_DBG("exit",0);

    return c;
}


void print_mem_desc(ptl_md_t * md)
{
    printf("%d:md : start %p : length %d\n",armci_me,md->start, md->length);
    fflush(stdout);
}



void armci_register_shmem(void *my_ptr, long size, long *idlist, long off,
       void *sptr)
{
int i=0,dst,found=0;
long id = idlist[2];
long reg_size=0;
int reg_num = serv_portals->reg_count;
extern void *armci_shm_reg_ptr(int);
extern long armci_shm_reg_size(int i, long id);
ARMCI_Group def_group;
        
    ARMCI_PR_DBG("enter",0);

    if(DEBUG_COMM){
      printf("%d:registering id=%ld size=%ld\n",armci_me,id,size);
      fflush(stdout);
    }

    bzero(_tmp_rem_reginfo,sizeof(aptl_reginfo_t)*armci_nproc);
    if(size){
      if(reg_num>=MAX_MEM_REGIONS)
        armci_die("reg_num corrupted",reg_num);
      for(i=0;i<reg_num;i++)
        if(IN_REGION(my_ptr,_rem_meminfo[armci_me].reginfo[i])){
          found=1;
          break;
        } 
      if(!found){ 
        /* record new region id */
        _tmp_rem_reginfo[armci_me].base_ptr = armci_shm_reg_ptr(i);
        _tmp_rem_reginfo[armci_me].size = armci_shm_reg_size(i,id);
        _tmp_rem_reginfo[armci_me].islocal = 0;
        armci_pin_contig1(_tmp_rem_reginfo[armci_me].base_ptr,_tmp_rem_reginfo[armci_me].size);
      }
    }
    ARMCI_Group_get_default(&def_group);
    armci_msg_group_gop_scope(SCOPE_ALL,_tmp_rem_reginfo,(sizeof(aptl_reginfo_t)*armci_nproc/sizeof(int)),"+",ARMCI_INT,&def_group);
    for(i=0;i<armci_nproc;i++)
      if(_tmp_rem_reginfo[i].size){
        reg_num = _rem_meminfo[i].reg_count;
        _rem_meminfo[i].reginfo[reg_num].base_ptr = _tmp_rem_reginfo[i].base_ptr;
        _rem_meminfo[i].reginfo[reg_num].size = _tmp_rem_reginfo[i].size;
        _rem_meminfo[i].reginfo[reg_num].islocal = _tmp_rem_reginfo[i].islocal;
        _rem_meminfo[i].reg_count++;
      }
    if(DEBUG_COMM){
      printf("%d: regist id=%ld found=%d size=%ld reg_num=%d\n",
                      armci_me,id,found,reg_size,reg_num);
      fflush(stdout);
    }
    ARMCI_PR_DBG("enter",0);
}

void armci_register_shmem_grp(void *my_ptr, long size, long *idlist, long off,
       void *sptr,ARMCI_Group *group)
{
ARMCI_Group orig_group;
    ARMCI_Group_get_default(&orig_group);
    ARMCI_Group_set_default(group);
    armci_register_shmem(my_ptr,size,idlist,off,sptr);
    ARMCI_Group_set_default(&orig_group);
}

static int _get_rem_lock_info(int proc, void *ptr,size_t bytes, size_t* offset)
{
int i;
    ARMCI_PR_DBG("enter",0);
    *offset = ((char *)ptr-(char *)all_locks[proc]);
    return (MAX_MEM_REGIONS+1);
    ARMCI_PR_DBG("exit",i);
}

static int _get_rem_info(int proc, void *ptr,size_t bytes, size_t* offset)
{
int i;
rem_meminfo_t *mem_info=&(_rem_meminfo[proc]);
aptl_reginfo_t *memreg = mem_info->reginfo;
    ARMCI_PR_DBG("enter",0);
    for(i=0;i<mem_info->reg_count;i++)
      /*for now size is not verified*/
      if(ptr>memreg[i].base_ptr && 
                      ptr< ((char *)memreg[i].base_ptr+memreg[i].size)){
        *offset = ((char *)ptr-(char *)memreg[i].base_ptr);
        ARMCI_PR_DBG("exit",i);
        return i;
      }
    ARMCI_PR_DBG("exit",i);
    armci_die("_get_rem_info, rem memory region not found",bytes);
}

void armci_client_direct_get(ptl_process_id_t dest_proc,
                ptl_size_t offset_remote, ptl_match_bits_t mb, size_t bytes,
                ptl_md_t *md_local, 
                ptl_handle_md_t *md_hdl_local)
{
int rc;
ptl_size_t offset_local = 0;

    ARMCI_PR_DBG("enter",0);

    if(DEBUG_COMM){
      printf("%d:armci_client_direct_get:BYTES = %d\n",armci_me,bytes);
      printf("\n%d:offr=%d offl=%d\n",armci_me,offset_remote,offset_local);
      fflush(stdout);
    }

    rc = PtlMDBind(portals->ni_h,*md_local, PTL_UNLINK, md_hdl_local);
    if (rc != PTL_OK){
      printf("%d:PtlMDBind: %s\n", portals->rank, ARMCI_NET_ERRTOSTR(rc) );
      armci_die("armci_client_direct_get: ptlmdbind failed",0);
    }

    rc = PtlGetRegion(*md_hdl_local,offset_local,bytes,dest_proc,portals->ptl,
                   0,mb,offset_remote);

    if (rc != PTL_OK){
      printf("%d:PtlGetRegion: %s\n", portals->rank, ARMCI_NET_ERRTOSTR(rc) );
      armci_die("PtlGetRegion failed",0); 
    }

    if(DEBUG_COMM){
      printf("\n%d:issued get\n",armci_me);fflush(stdout);
    }

    ARMCI_PR_DBG("exit",0);
}

void armci_portals_get(int proc, void *src_buf, void *dst_buf, size_t bytes,
                       void** cptr,int tag)
{
int rc;
ptl_size_t offset_local = 0, offset_remote=0;
ptl_md_t *md_local;
ptl_handle_md_t *md_hdl_local;
int rem_info;
comp_desc *cdesc;
ptl_process_id_t dest_proc;
int c_info;

    ARMCI_PR_DBG("enter",0);

    /*first remote process information*/
    dest_proc.nid = portals->ptl_pe_procid_map[proc].nid;
    dest_proc.pid = portals->ptl_pe_procid_map[proc].pid;

    /*create local xfer info*/
    cdesc = get_free_comp_desc(&c_info);
    md_local = &cdesc->mem_dsc;
    md_hdl_local = &cdesc->mem_dsc_hndl; 
    md_local->length=bytes;
    md_local->start=dst_buf;
    md_local->user_ptr = (void *)cdesc;
    md_local->options =  PTL_MD_OP_GET | PTL_MD_EVENT_START_DISABLE;

    /*get remote info*/
    rem_info = _get_rem_info(proc, src_buf, bytes, (size_t*)&offset_remote);

    cdesc->dest_id = proc;
    if (tag){
      *((comp_desc **)cptr) = cdesc;
      cdesc->tag = tag;
      cdesc->type = ARMCI_PORTALS_NBGET;
      /*printf("\n%d:get tag=%d c_info=%d
       * %p",armci_me,tag,c_info,cdesc);fflush(stdout);*/
    }
    else{
      cdesc->tag = 0;
      cdesc->type = ARMCI_PORTALS_GET; 
    }

    cdesc->active = 1;
    armci_client_direct_get(dest_proc,offset_remote,(ptl_match_bits_t)rem_info,
                bytes,md_local,md_hdl_local);

    if(!tag){ 
       armci_client_complete(NULL,proc,0,cdesc); /* check this later */
    }

    ARMCI_PR_DBG("exit",0);

}


void armci_client_nb_get(int proc, void *src_buf, int *src_stride_arr, 
                             void *dst_buf, int *dst_stride_arr, int bytes,
                             void** cptr,int tag)
{
}

void armci_client_direct_send(ptl_process_id_t dest_proc,
                ptl_size_t offset_remote, ptl_match_bits_t mb, size_t bytes,
                ptl_md_t *md_local, 
                ptl_handle_md_t *md_hdl_local)
{
int rc;
ptl_size_t offset_local = 0;

    ARMCI_PR_DBG("enter",0);

    if(DEBUG_COMM){
      printf("%d:armci_client_direct_send:BYTES = %d\n",armci_me,bytes);
      printf("\n%d:offr=%d offl=%d\n",armci_me,offset_remote,offset_local);
      fflush(stdout);
    }

    rc = PtlMDBind(portals->ni_h,*md_local, PTL_UNLINK, md_hdl_local);
    if (rc != PTL_OK){
      fprintf(stderr, "%d:PtlMDBind: %s\n", portals->rank, 
                      ARMCI_NET_ERRTOSTR(rc)); 
      armci_die("armci_client_direct_send: ptlmdbind failed",0);
    }
    
    rc = PtlPutRegion(*md_hdl_local,offset_local,bytes,
#ifdef PUT_LOCAL_ONLY_COMPLETION
                    PTL_NOACK_REQ,
#else
                    PTL_ACK_REQ,
#endif
                    dest_proc,portals->ptl,0, mb,offset_remote, 0);

    if (rc != PTL_OK){
      fprintf(stderr, "%d:PtlPutRegion: %s\n", portals->rank, 
                      ARMCI_NET_ERRTOSTR(rc) );
      armci_die("PtlPutRegion failed",0);
    }

    ARMCI_PR_DBG("exit",0);
}


void armci_portals_put(int proc, void *src_buf, void *dst_buf, size_t bytes,
                             void** cptr,int tag)
{
int rc;
ptl_size_t offset_local = 0, offset_remote=0;
ptl_md_t *md_local;
ptl_handle_md_t *md_hdl_local;
int rem_info;
comp_desc *cdesc;
ptl_process_id_t dest_proc;
int c_info;

    ARMCI_PR_DBG("enter",0);

    /*first process information*/
    dest_proc.nid = portals->ptl_pe_procid_map[proc].nid;
    dest_proc.pid = portals->ptl_pe_procid_map[proc].pid;

    /*create local xfer info*/
    cdesc = get_free_comp_desc(&c_info);
    md_local = &cdesc->mem_dsc;
    md_hdl_local = &cdesc->mem_dsc_hndl; 
    md_local->length=bytes;
    md_local->start=src_buf;
    md_local->user_ptr = (void *)cdesc;
    md_local->options =  PTL_MD_OP_PUT | PTL_MD_EVENT_START_DISABLE;
    
    /*get remote info*/
    rem_info = _get_rem_info(proc, dst_buf, bytes, (size_t*)&offset_remote);
                    

    if(DEBUG_COMM){
      printf("\n%d:offr=%d offl=%d\n",armci_me,offset_remote,offset_local);
    }

    cdesc->dest_id = proc;
    if (tag){
      *((comp_desc **)cptr) = cdesc;
      cdesc->tag = tag;
      cdesc->type = ARMCI_PORTALS_NBPUT;
      /*printf("\n%d:put tag=%d c_info=%d
       * %p",armci_me,tag,c_info,cdesc);fflush(stdout);*/
    }
    else{
      cdesc->tag = 0;
      cdesc->type = ARMCI_PORTALS_PUT; 
    }
    
    cdesc->active = 1;

    armci_client_direct_send(dest_proc,offset_remote,(ptl_match_bits_t)rem_info,
                bytes,md_local,md_hdl_local);

    armci_update_fence_array(proc,1);

    if(!tag){ 
       armci_client_complete(NULL,proc,0,cdesc); /* check this later */
    }
    else
       portals->outstanding_puts++;


    ARMCI_PR_DBG("exit",0);

}
void armci_client_nb_send(int proc, void *src_buf, int *src_stride_arr, 
                             void *dst_buf, int *dst_stride_arr, int bytes,
                             void** cptr,int tag)
                             
{
}

/*using non-blocking for multiple 1ds inside a 2d*/
void armci_network_strided(int op, void* scale, int proc,void *src_ptr,
                int src_stride_arr[], void* dst_ptr, int dst_stride_arr[],
                int count[], int stride_levels, armci_ihdl_t nb_handle)
{
int i, j,tag=0;
long idxs,idxd;    /* index offset of current block position to ptr */
int n1dim;  /* number of 1 dim block */
int bvalue_s[MAX_STRIDE_LEVEL], bunit[MAX_STRIDE_LEVEL];
int bvalue_d[MAX_STRIDE_LEVEL];
size_t bytes = count[0];
void *sptr,*dptr;
NB_CMPL_T cptr;
ptl_process_id_t dest_proc;
ptl_size_t offset_remote;
comp_desc *cdesc;
int c_info; 
ptl_md_t *md_local;
int rem_info;

    ARMCI_PR_DBG("enter",0);
    
    if(nb_handle)tag=nb_handle->tag;

    /*first remote process information*/
    dest_proc.nid = portals->ptl_pe_procid_map[proc].nid;
    dest_proc.pid = portals->ptl_pe_procid_map[proc].pid;

    rem_info = _get_rem_info(proc, (op==GET)?src_ptr:dst_ptr, bytes,
                             (size_t*)&offset_remote);

    /* number of n-element of the first dimension */
    n1dim = 1;
    for(i=1; i<=stride_levels; i++)
        n1dim *= count[i];

    /* calculate the destination indices */
    bvalue_s[0] = 0; bvalue_s[1] = 0; bunit[0] = 1; 
    bvalue_d[0] = 0; bvalue_d[1] = 0; bunit[1] = 1;
    for(i=2; i<=stride_levels; i++) {
        bvalue_s[i] = bvalue_d[i] = 0;
        bunit[i] = bunit[i-1] * count[i-1];
    }

    if(ACC(op)){ /*for now die for acc*/
      /*lock here*/
      printf("\nSHOULD NOT DO NETWORK_STRIDED FOR ACCS \n",armci_me);
      fflush(stdout);
      armci_die("network_strided called for acc",proc);
    }

    /*loop over #contig chunks*/
    for(i=0; i<n1dim; i++) {
    ptl_handle_md_t *md_hdl_local;
      tag = GET_NEXT_NBTAG();      
      idxs = 0;
      idxd = 0;
      for(j=1; j<=stride_levels; j++) {
        idxs += bvalue_s[j] * src_stride_arr[j-1];
        idxd += bvalue_d[j] * dst_stride_arr[j-1];
        if((i+1) % bunit[j] == 0) {bvalue_s[j]++;bvalue_d[j]++;}
        if(bvalue_s[j] > (count[j]-1)) bvalue_s[j] = 0;
        if(bvalue_d[j] > (count[j]-1)) bvalue_d[j] = 0;
      }
      sptr = ((char *)src_ptr)+idxs;
      dptr = ((char *)dst_ptr)+idxd;
      cdesc = get_free_comp_desc(&c_info);
      md_local = &cdesc->mem_dsc;
      md_hdl_local = &cdesc->mem_dsc_hndl;
      md_local->length=bytes;
      md_local->start=(op==GET)?dptr:sptr;
      md_local->user_ptr = (void *)cdesc;
      cdesc->dest_id = proc;
      cdesc->tag = tag;
      
      if(op==GET){
        md_local->options =  PTL_MD_OP_GET | PTL_MD_EVENT_START_DISABLE;
        cdesc->active = 1;
        cdesc->type = ARMCI_PORTALS_NBGET;
        /*
        printf("\n%d:reminfo=%d off=%d idxs=%d idxd=%d",armci_me, rem_info,
                        offset_remote, idxs, idxd);
                        */
        armci_client_direct_get( dest_proc,offset_remote+idxs,rem_info,
                        bytes,md_local,md_hdl_local);
      }
      else if(op==PUT){
        cdesc->active = 1;
        cdesc->type = ARMCI_PORTALS_NBPUT;
        md_local->options =  PTL_MD_OP_PUT | PTL_MD_EVENT_START_DISABLE;
        armci_client_direct_send(dest_proc,offset_remote+idxd,rem_info,
                bytes,md_local,md_hdl_local);
        if(op==PUT)portals->outstanding_puts++;
        armci_update_fence_array(proc,1);
      }
      else if(ACC(op)){
        assert(0);
      }
      else{
        ARMCI_PR_DBG("exit",0);
        armci_die("in network_strided unknown opcode",op);
      }
      
      armci_client_complete(NULL,proc,tag,cdesc);
    }

    if(ACC(op)){
    /*unlock here*/
    }

    if(nb_handle){
      /* completing the last call is sufficient, given ordering semantics*/
      nb_handle->tag=tag;
      nb_handle->cmpl_info=cdesc;
    }
    else{
      /*completing the last call ensures everything before it is complete this
       * is one of the main reasons why dataserver is necessary*/
      /*armci_client_complete(NULL,proc,tag,cdesc);*/
    }
    ARMCI_PR_DBG("exit",0);
}

void armci_client_direct_getput(ptl_process_id_t dest_proc,
                ptl_size_t offset_remote, ptl_match_bits_t mb, size_t bytes,
                ptl_md_t *md_local_get,ptl_md_t *md_local_put, 
                ptl_handle_md_t *md_hdl_local_get, ptl_handle_md_t
                *md_hdl_local_put)
{
int rc;
ptl_size_t offset_get = 0;
ptl_size_t offset_put = 0;

    ARMCI_PR_DBG("enter",0);

    if(DEBUG_COMM){
      printf("%d:armci_client_direct_getput:BYTES = %d\n",armci_me,bytes);
      printf("\n%d:offr=%d\n",armci_me,offset_remote);fflush(stdout);
    }

    rc = PtlGetPutRegion(*md_hdl_local_get, offset_get, *md_hdl_local_put,
                    offset_put,bytes,dest_proc, portals->ptl,0,mb,
                    offset_remote,0);
    if (rc != PTL_OK){
      printf("%d:PtlGetPutRegion: %s\n", portals->rank, ARMCI_NET_ERRTOSTR(rc) );
      armci_die("PtlGetPutRegion failed",0);
    }
    
    ARMCI_PR_DBG("exit",0);

}

/**
 * Performs the atomic swap operation. Atomic swap of the remote data at the
 * target with the data passed in the "put" memory descriptor. The original
 * contents of the remote memory region are returned and placed in the get
 * memory descriptor of the source.
 */
int armci_portals_getput(int proc,void *getinto, void *putfrom, void* dst,
                         size_t bytes, void **cptr, int tag)
{
int rc, i;
ptl_size_t offset_remote = 0;
ptl_handle_md_t *md_hdl_local_put,*md_hdl_local_get;
int rem_info;
comp_desc *cdescg,*cdesc;
ptl_process_id_t dest_proc;

    ARMCI_PR_DBG("enter",0);
    
    dest_proc.nid = portals->ptl_pe_procid_map[proc].nid;
    dest_proc.pid = portals->ptl_pe_procid_map[proc].pid;
    rem_info = _get_rem_lock_info(proc, dst, bytes, (size_t*)&offset_remote);

    cdescg = &(_lockget_cd_array[proc]);
    cdescg->active = 1;
    cdesc = &(_lockput_cd_array[proc]);
    cdesc->active = 1;

    md_hdl_local_put = &(_lockput_cd_array[proc].mem_dsc_hndl);
    md_hdl_local_get = &(_lockget_cd_array[proc].mem_dsc_hndl);

    armci_client_direct_getput(dest_proc,offset_remote,(ptl_match_bits_t)rem_info,
                    bytes,NULL,NULL,md_hdl_local_get,md_hdl_local_put);
                    
    if(DEBUG_COMM){
      printf("\n%d:issued getput to %d\n",armci_me,proc);fflush(stdout);
    }

    armci_client_complete(NULL,proc,0,cdescg); 
    armci_client_complete(NULL,proc,0,cdesc);
    ARMCI_PR_DBG("exit",0);
    return rc;
}

void armcill_allocate_locks(int num)
{
    ptl_md_t *md_local_put, *md_local_get;
    int rc, i;
    long *my_locks;

    ARMCI_PR_DBG("enter",0);
    
    num_locks = num;
    if(DEBUG_COMM)
       printf("%d:armci_allocate_locks num=%d\n", armci_me,num_locks);
    
    if(MAX_LOCKS < num)
       armci_die2("too many locks", ARMCI_PORTALS_MAX_LOCKS, num);
    
    /* allocate memory to hold lock info for all the processors */
    all_locks = malloc(armci_nproc*sizeof(long *));
    if(!all_locks) armci_die("armcill_init_locks: malloc failed",0);
    bzero(all_locks, armci_nproc*sizeof(long));
    
    /* initialize local locks */
    my_locks = malloc(num*sizeof(long));
    if(!my_locks) armci_die("armcill_init_locks: malloc failed",0);
    bzero(my_locks, num*sizeof(long));

    armci_pin_contig_lock(my_locks,num*sizeof(long));
    
    all_locks[armci_me]=my_locks;
    
    /* now we use all-reduce to exchange locks info among everybody */
    armci_exchange_address((void **)all_locks, armci_nproc);
    
    _lockput_cd_array = (comp_desc *)
                    malloc(sizeof(comp_desc)*armci_nproc); 
    _lockget_cd_array = (comp_desc *)
                    malloc(sizeof(comp_desc)*armci_nproc); 

    for(i=0;i<armci_nproc;i++){

      _lockget_cd_array[i].active  = _lockput_cd_array[i].active  = 0;
      _lockget_cd_array[i].tag     = _lockput_cd_array[i].tag     = 0;
      _lockget_cd_array[i].dest_id = _lockput_cd_array[i].dest_id = i;
      _lockget_cd_array[i].type    = ARMCI_PORTALS_GET;
      _lockput_cd_array[i].type    = ARMCI_PORTALS_GETPUT;
      
      md_local_put = &(_lockput_cd_array[i].mem_dsc);
      md_local_get = &(_lockget_cd_array[i].mem_dsc);
      
      md_local_get->eq_handle = md_local_put->eq_handle = portals->eq_h;
      md_local_get->max_size  = md_local_put->max_size  = 0;
      md_local_get->threshold = md_local_put->threshold = PTL_MD_THRESH_INF;
      md_local_put->options   = PTL_MD_OP_PUT | PTL_MD_EVENT_START_DISABLE;
      md_local_get->options   = PTL_MD_OP_GET | PTL_MD_EVENT_START_DISABLE;
      md_local_put->start     = &a_p_putfrom;
      md_local_get->start     = &a_p_getinto;
      md_local_get->length    = md_local_put->length=sizeof(long);
      md_local_put->user_ptr  = (_lockput_cd_array+i);
      md_local_get->user_ptr  = (_lockget_cd_array+i);
    
      rc = PtlMDBind(portals->ni_h, *md_local_put, 
                     PTL_RETAIN, &(_lockput_cd_array[i].mem_dsc_hndl));
      if (rc != PTL_OK){
        printf("%d:PtlMDBind: %s\n", portals->rank, ARMCI_NET_ERRTOSTR(rc)); 
        armci_die("ptlmdbind failed",0);
      }
      rc = PtlMDBind(portals->ni_h,*md_local_get, 
                      PTL_RETAIN, &(_lockget_cd_array[i].mem_dsc_hndl));
      if (rc != PTL_OK){
        printf( "%d:PtlMDBind: %s\n", portals->rank, ARMCI_NET_ERRTOSTR(rc)); 
        armci_die("ptlmdbind failed",0);
      }
    }
    ARMCI_PR_DBG("exit",0);
}

void armcill_lock(int mutex, int proc)
{
    int lockcount=1;
    
    a_p_getinto=0;
    a_p_putfrom=1;
    
    do{
       armci_portals_getput(proc, &a_p_getinto, &a_p_putfrom,
                            (all_locks[proc]+mutex), sizeof(long),
                            NULL, GET_NEXT_NBTAG());
       if(++lockcount%2) usleep(1);
    }while(a_p_getinto!=0);
}


/*\ unlock specified mutex on node where process proc is running
\*/
void armcill_unlock(int mutex, int proc)
{
    a_p_getinto=0;
    a_p_putfrom=0;
    
    armci_portals_getput(proc, &a_p_getinto, &a_p_putfrom,
                         (all_locks[proc]+mutex), sizeof(long),
                         NULL, GET_NEXT_NBTAG());
    
    if(a_p_getinto!=1) armci_die("armcill_unlock: getput failed", 0);
}

int armci_portals_rmw_(int op, int *ploc, int *prem, int extra, int proc)
{
    return(0);
}

void armci_portals_shmalloc_allocate_mem(int num_lks)
{
    void **ptr_arr;
    void *ptr;
    armci_size_t bytes = 128;
    int i;    
    
    ptr_arr    = (void**)malloc(armci_nproc*sizeof(void*));
    if(!ptr_arr) armci_die("armci_shmalloc_get_offsets: malloc failed", 0);
    bzero((char*)ptr_arr,armci_nproc*sizeof(void*));

    ARMCI_Malloc(ptr_arr,bytes);
    
    return;
}
