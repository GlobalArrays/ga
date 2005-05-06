/*interfaces for checkpointing */

/* TODO
 * work on the case if pagenum==firstpage or lastpage when writing pages
 */
#include <stdio.h>
#include <setjmp.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <sys/param.h>
#include <sys/wait.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <dirent.h>
#include <stdarg.h>
#include "armcip.h"
#include "message.h"
#include "armci_storage.h"
#include "armci_chkpt.h"

#define DEBUG 0

/*\
 * ----------CORE FUNCTIONS -----------
 * armci_init_checkpoint() - the checkpointing code is initialized 
 * armci_icheckpoint_init() - called when user request to protect this ga
 * armci_icheckpoint_finalize() - called when user requests to destroy the ga
 * armci_icheckpoint() - called every time a ga is checkpointed
 * armci_recover() - called to recoved
 * ----------SUPPORT FUNCTIONS -----------
 * armci_ckpt_pgfh() - pagefault handler, to set the list of dirty pages
 * armci_monitor_addr() - monitors address to look for and pages to set readonly
 * armci_create_record() - create the record for local part of ga being stored
 * armci_protect_pages() - to protect pages in the dirty page list
 * armci_storage_read() - reads record from storage 
 * armci_storage_write() - writes into storage
 * armci_storage_fopen() - opens a file in storage
 *
\*/

/*\ global variables
\*/
static armci_storage_record_t armci_storage_record[1001];
static int number_of_records=1;
static int mypagesize; 
int **armci_rec_ind;
static armci_page_info_t armci_dpage_info;


/* ----------SUPPORT FUNCTIONS ----------- */

/* This function is called from ARMCI sigv and sigbus handler */
int armci_ckpt_pgfh(void *addr, int errno, int fd)
{
    char *paddr;
    unsigned long pagenum;

    /*find the page number and the corresponding page aligned address*/
    pagenum = (unsigned long)((long)addr/mypagesize);
    (long)paddr = pagenum*mypagesize; 

    printf("\n%d:paddr=%p addr=%p %d\n",armci_me,paddr,addr,pagenum);
    fflush(stdout);

    /*page is being touched change page permission to READ/WRITE*/
    mprotect(paddr, mypagesize, PROT_READ | PROT_WRITE);
    
    /*mark pagenumber dirty in dirty page array*/
    armci_dpage_info.touched_page_arr[armci_dpage_info.num_touched_pages++] =
            pagenum;
    if(pagenum<armci_dpage_info.firstpage)
            armci_dpage_info.firstpage = pagenum;
    if(pagenum>armci_dpage_info.lastpage)
            armci_dpage_info.lastpage = pagenum;
    return(0);
}

#if 0
printf("\n%d:pagenum=%d first=%d last=%d\n",armci_me,pagenum,armci_storage_record[i].firstpage,(armci_storage_record[i].firstpage+armci_storage_record[i].totalpages));fflush(stdout);
#endif

static int armci_create_record(ARMCI_Group *group, int count)
{
    int recind;
    int relprocid,relmaster;
    int subs[1]={0};
    int rc;

    rc = ARMCI_Group_rank(group,&relprocid);
    /*create and broadcast new index in the records data structure*/
    if(relprocid==0)
       ARMCI_Rmw(ARMCI_FETCH_AND_ADD,&recind,armci_rec_ind[0],1,0);
    armci_msg_group_bcast_scope(SCOPE_ALL,&recind,sizeof(int),0,group);

    if(recind>1001) armci_die("create_record, failure",recind);
    armci_storage_record[recind].pid = armci_me;
    armci_storage_record[recind].rid = recind;
    armci_storage_record[recind].rel_pid = relprocid;
    memcpy(&armci_storage_record[recind].group,group,sizeof(ARMCI_Group));
    armci_storage_record[recind].user_addr = (armci_monitor_address_t *)malloc(sizeof(armci_monitor_address_t)*count);
    armci_storage_record[recind].user_addr_count=count;

    number_of_records++;

    return(recind);
}


static void armci_protect_pages(int startpagenum,int numpages)
{
    unsigned long i=0;
    for(i=startpagenum;i<startpagenum+numpages;i++){
       char *addr;
       addr =(char *)((unsigned long)(i*mypagesize));
       mprotect(addr, mypagesize,PROT_READ);
    }
}



/*\ ----------CORE FUNCTIONS -----------
\*/
/*called in armci init*/
int armci_init_checkpoint()
{
    int val=1,rc;
    mypagesize = getpagesize();

    armci_rec_ind = (int **)malloc(sizeof(int *)*armci_nproc);

    rc = ARMCI_Malloc((void **)armci_rec_ind, 2*sizeof(int));
    assert(rc==0);

    ARMCI_Register_Signal_Handler(SIGSEGV,(void *)armci_ckpt_pgfh);
}

void armci_create_chkptds(armci_ckpt_ds_t *ckptds, int count)
{
    ckptds->count=count;
    ckptds->ptr_arr=(void **)malloc(sizeof(void *)*count);
    ckptds->sz=(size_t *)malloc(sizeof(size_t)*count);
    if(ckptds->ptr_arr==NULL || ckptds->sz == NULL)
      armci_die("malloc failed in armci_create_ckptds",sizeof(size_t)*count);
}

/*called everytime a new checkpoint record is created*/
int armci_icheckpoint_init(char *filename,ARMCI_Group *grp, int savestack, 
                int saveheap, armci_ckpt_ds_t *ckptds)
{
    int rid;
    long bytes;
    void *startaddr;
    unsigned long laddr;
    int totalpages=0,i=0;

    /*create the record*/
    rid = armci_create_record(grp,ckptds->count);

    if(DEBUG){
       printf("\n%d:got rid = %d",armci_me,rid);fflush(stdout);
    }

    /*******************user pages***********************/
    if(armci_storage_record[rid].ckpt_heap){
       return;
    }
    for(i=0;i<ckptds->count;i++){
       armci_monitor_address_t *addrds =&armci_storage_record[rid].user_addr[i];
       addrds->bytes = ckptds->sz[i];
       addrds->ptr = ckptds->ptr_arr[i];
       laddr = (unsigned long)(addrds->ptr);
       addrds->firstpage = (unsigned long)((long)laddr/mypagesize);

       if(laddr%mypagesize ==0){
         totalpages = (int)(bytes/mypagesize);
         if(bytes%mypagesize)totalpages++;
       }
       else {
         int shift;
         shift = laddr%mypagesize;
         totalpages = 1+(bytes-shift)/mypagesize;
         if((bytes-shift)%mypagesize)totalpages++;
       }
       addrds->totalpages = totalpages;
       addrds->num_touched_pages = totalpages;
       addrds->touched_page_arr = malloc(totalpages*sizeof(unsigned long));
       if(addrds->touched_page_arr==NULL)
         armci_die("malloc failed in armci_icheckpoint_init",totalpages);
       addrds->touched_page_arr[0]=addrds->firstpage;
       for(i=1;i<totalpages;i++){
         addrds->touched_page_arr[i]=addrds->touched_page_arr[i-1]+1;
       }
       armci_protect_pages(addrds->firstpage,addrds->totalpages);
    }
    
    /*open the file for reading and writing*/
    armci_storage_record[rid].fileinfo.filename = malloc(sizeof(char)*strlen(filename));
    if(NULL==armci_storage_record[rid].fileinfo.filename)
      armci_die("malloc failed for filename in ga_icheckpoint_init",0);
    strcpy(armci_storage_record[rid].fileinfo.filename,filename);
    armci_storage_record[rid].fileinfo.fd = armci_storage_fopen(filename);
}


/*get the list of changed pages from touched_page_array and rewrite the 
 * changed pages*/
int armci_icheckpoint(int rid)
{
    int i,rc;
    off_t ofs;
    char *addr;
    if(!setjmp(armci_storage_record[rid].jmp)){
       if(armci_storage_record[rid].ckpt_stack){
         /*write the stack information*/
         addr = sbrk(0);
         if(addr < (char *)armci_storage_record[rid].stack_mon.ptr){ 
           /*this means change in st/he save what ever is left and reset size*/
         }
         else{
           /*nothing changed, so we probably are ok*/
         }
       }
       if(armci_storage_record[rid].ckpt_heap){
       }
       else{
         rc = armci_storage_write_ptr(armci_storage_record[rid].fileinfo.fd,
                         &armci_storage_record[rid].jmp,sizeof(jmp_buf),
                         4*sizeof(int));
         for(i=0;i<armci_storage_record[rid].user_addr_count;i++){
           armci_monitor_address_t *addrds = &armci_storage_record[rid].user_addr[i];
           ofs=(off_t)(addrds->fileoffset);
           rc = armci_storage_write_pages(armci_storage_record[rid].fileinfo.fd,addrds->firstpage,addrds->touched_page_arr,addrds->num_touched_pages,mypagesize,ofs);
           bzero(addrds->touched_page_arr,sizeof(unsigned long)*addrds->num_touched_pages);
           addrds->num_touched_pages = 0;
         }
       }
    }
    else { /*long jump brings us here */
       /*open the ckpt files*/
    }

    return(rc); /* 0 is good*/
}


int armci_irecover(int rid,int iamreplacement)
{
    int rc;
    off_t ofs;
    /*restore jmpbuf and pid and longjmp*/
    if(iamreplacement){
      rc=armci_storage_read_ptr(armci_storage_record[rid].fileinfo.fd,&armci_storage_record[rid].jmp,sizeof(jmp_buf),4*sizeof(int));
    }

    longjmp(armci_storage_record[rid].jmp,1);

    /*if we should never come here things are hosed */
    armci_die2("we should never come here",rid,iamreplacement);
    return(1);
}


int armci_icheckpoint_finalize()
{
    /*free the record id, make it available for next use*/
    /*close the files used for checkpointing*/
}
