#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <strings.h>
#include <stdlib.h>
#include "armcip.h"
#include "shmem.h"

#define DEBUG_ 0
double *armci_servmap_common_buffer;
int *client_ready_flag_array;
char *serverbuffer;
char *largebufptr;
long serv_cb_key=SERVER_GENBUF_KEY;
rcv_field_info_t *serv_rcv_field_info;
int *sbufreqdescarray;
int sbufrcvnum;
int *sbufrcvdescarray;
char *flag_array;
server_auth_t *server_auths;
extern ndes_t _armci_group;
char *server_ack_buf;
clientserv_buf_avail_auth_t *client_availbuf_auths;
char *tempbuf;
char *_sr8k_serverlargegetbuf;
int *getbuf_locked;
int getbuf_locked_auth;
int **putbuf_locked;
int putbuf_locked_auth;
int *client_pending_op_auth;
int *pending_op_count;
int *small_buf_count;
extern int armci_rdma_make_tcw(void *src, Cb_size_t off, int bytes, int desc,\
		char *flag);
extern void armci_rdma_modify_tcw(int tcwd, char *src, Cb_size_t off,\
		int bytes);
extern void armci_rdma_kick_tcw_put(int tcwd);
extern void armci_rdma_put_wait(int tcwd, char *flag);
int *clauth;

int armci_srdma_make_tcw(void *src, Cb_size_t off, int bytes, int desc,\
		int flag)
{
int rc;
Cb_msg msg;
int tcwd;
     memset(&msg, 0, sizeof(msg));
     msg.data.addr = src;
     msg.data.size = (Cb_size_t)bytes;
     if ( (rc=combuf_make_tcw(&msg, desc, off, 0, &tcwd)) != COMBUF_SUCCESS)
         armci_die("combuf_make_tcw in server failed",rc);
     if(DEBUG_){printf("%d:put dsc=%d off=%d\n",armci_me,desc,off); fflush(stdout);}
     return tcwd;
}
extern void server_set_mutex_array(char *tmp);

/*steps for server_initial_connection are
 * 0. do a get and map on the memory client created as common
 * 1. get physical memory for server buffers and map to it
 * 2. create arrays for calls to mwait
 * 3. create recv fields and update serv_rcv_info
 * 4. create a rcv field for control information 
 * 5. create space and tcw's for all client server_buf_avail arrays
*/
void armci_server_initial_connection(){
int size,bytes,extra,sbufdesc,dst,field_num,key,auth;
int snum,ssize,lnum,lsize,rc,i,k,j;
char *temp;
Cb_node_rt remote;
Cb_object_t oid,serv_cb_oid,moid;
Cb_opt_region options;
int mykey = CLIENT_SERV_COMMON_CB_KEY;
     size = ROUND_UP_PAGE(sizeof(rcv_field_info_t)*(armci_nproc+armci_nclus)\
		+(armci_nproc+1)*sizeof(int));
     if(combuf_object_get(mykey, (Cb_size_t)size,0,&oid)
               != COMBUF_SUCCESS) 
		armci_die("armci_serverinit_connection combufget buf failed",0);
     if(combuf_map(oid, 0, (Cb_size_t)size, COMBUF_COMMON_USE, (char**)&armci_servmap_common_buffer)
               != COMBUF_SUCCESS) armci_die("combuf map int amrci_serv_initial_connection for buf failed",0);
     server_ready_flag = (int *)armci_servmap_common_buffer;
     client_ready_flag_array = (int *)armci_servmap_common_buffer+1+armci_me;
	client_ready_flag = (int *)armci_servmap_common_buffer+1;
     serv_rcv_field_info = (rcv_field_info_t *)((int *)armci_servmap_common_buffer+armci_nproc+1);
     client_rcv_field_info = serv_rcv_field_info + armci_nclus;
	server_reset_memory_variables();

	server_set_mutex_array(temp);

	ssize = SMALL_MSG_SIZE;snum = SMALL_MSG_NUM;
	lnum = LARGE_MSG_NUM;lsize = LARGE_MSG_SIZE;
	bytes = ROUND_UP_PAGE(ssize*snum+lsize*lnum+128+(lnum+snum*numofbuffers+1)*FLAGSIZE+2*armci_nproc*sizeof(int));
	if((rc=combuf_object_get(serv_cb_key, (Cb_size_t)bytes, COMBUF_OBJECT_CREATE, &serv_cb_oid))
                        != COMBUF_SUCCESS) armci_die("serv_init_conn:combuf_object_get failed",rc);
     if((rc=combuf_map(serv_cb_oid, 0, (Cb_size_t)bytes, COMBUF_COMMON_USE, (char**)&serverbuffer))
                        != COMBUF_SUCCESS) armci_die("serv_init_conn:combuf map failed",rc);        
     serv_rcv_field_info[armci_clus_me].bufptr = (long)serverbuffer;
     serv_rcv_field_info[armci_clus_me].cb_key = serv_cb_key;
	flag_array = (char*)(serverbuffer+ssize*snum+lsize*lnum);
	pending_op_count=(int *)(serverbuffer+size*snum+lsize*lnum+128+(lnum+snum*numofbuffers+1)*FLAGSIZE);
	small_buf_count=pending_op_count+armci_nproc;
	for(i=0;i<armci_nproc;i++){pending_op_count[i]=0;small_buf_count[i]=0;}
	/*step 2 */
	sbufreqdescarray = (int *)malloc(sizeof(int)*(snum+lnum*numofbuffers+1));
	sbufrcvdescarray = (int *)malloc(sizeof(int)*(snum+lnum*numofbuffers+1));
	if(!sbufreqdescarray || !sbufrcvdescarray)armci_die("sbufrcvdescarray maloc failed",0);
	/*end step-2*/
	for(i=0;i<snum;i++){
		temp = ((char*)serverbuffer+ssize*i);
		if(DEBUG_)printf("\n%d:for index %d servbufptr=%p\n",armci_me,i,temp);
		memset(&options, 0, sizeof(options));
     	options.flag.addr = (char *)((char*)flag_array+i*FLAGSIZE);
     	options.flag.size = FLAGSIZE;
		if((rc=combuf_create_field(serv_cb_oid,temp,(Cb_size_t)(ssize),
				SERV_FIELD_NUM+i,0/*&options*/,0 /*COMBUF_CHECK_FLAG*/,&sbufdesc)) != COMBUF_SUCCESS)
       	armci_die("serv_init_connection-1:combuf_create_field failed",rc);
		sbufreqdescarray[i] = sbufdesc;
		if(DEBUG_){printf("\n%d:server:sbufdesc=%d for index %d\n",armci_me,sbufdesc,i);fflush(stdout);}
     }
     temp += ssize;
	putbuf_locked=(int **)malloc(sizeof(int*)*numofbuffers);
	if(!putbuf_locked)armci_die("putbuf_locked malloc failed",0);
     remote.type = CB_NODE_RELATIVE;
     remote.ndes = _armci_group;
     remote.node = armci_clus_me;
     combuf_target((Cb_node *)&remote, sizeof(remote), serv_cb_key, 0, -1, &putbuf_locked_auth);

     for(i=0;i<lnum;i++){
          temp = ((char*)serverbuffer+lsize*i+ssize*snum);
	/*temporary solution, has to be modified for more large buffers*/
		for(k=0;k<numofbuffers;k++){
			memset(&options, 0, sizeof(options));
          	options.flag.addr = (char *)((char*)flag_array+(i+snum+k)*FLAGSIZE);
          	options.flag.size = FLAGSIZE;
          	if((rc=combuf_create_field(serv_cb_oid,(temp+k*lsize/4+4),(Cb_size_t)(lsize/4-4),
				SERV_FIELD_NUM_FOR_LARGE_BUF+k,0/*&options*/,0/* COMBUF_CHECK_FLAG*/,&sbufdesc)) != COMBUF_SUCCESS)
          		armci_die("serv_init_connection-2:combuf_create_field failed",rc);
			sbufreqdescarray[k+i+snum] = sbufdesc;
			putbuf_locked[k+i]=(int *)(temp+k*lsize/4);
			*(putbuf_locked[k+i])=0;
			if(DEBUG_){
				printf("\n%d:server:sbufdesc=%d for index %d fieldsize=%d\n",armci_me,sbufdesc,k+i+snum,(lsize/4));
				fflush(stdout);
			}
		}
     }
	largebufptr=(char*)serverbuffer+ssize*snum;	
	directbuffer = largebufptr;

	/*create one large buffer on server only for handeling client get requests
	useful only in get pipeline*/
	bytes = ROUND_UP_PAGE(lsize);
	if((rc=combuf_object_get(SERVER_LARGE_GETBUF_KEY, (Cb_size_t)bytes, COMBUF_OBJECT_CREATE, &serv_cb_oid))
                        != COMBUF_SUCCESS) armci_die("serv_init_conn:combuf_object_get failed",rc);
     if((rc=combuf_map(serv_cb_oid, 0, (Cb_size_t)bytes, COMBUF_COMMON_USE, (char**)&_sr8k_serverlargegetbuf))
                        != COMBUF_SUCCESS) armci_die("serv_init_conn:combuf map failed",rc);
	getbuf_locked = (int *)_sr8k_serverlargegetbuf;
	remote.type = CB_NODE_RELATIVE;
     remote.ndes = _armci_group;
     remote.node = armci_clus_me;
	combuf_target((Cb_node *)&remote, sizeof(remote), SERVER_LARGE_GETBUF_KEY, 0, -1, &getbuf_locked_auth);
	*getbuf_locked = 0;	


	if(DEBUG_){
		printf("\nlargebufptr=%p serverbufend=%p",largebufptr,(serverbuffer+bytes));
		fflush(stdout);
	}
	
	server_auths=(server_auth_t *)malloc(sizeof(server_auth_t)*armci_nproc);
	clauth = (int *)malloc(sizeof(int)*armci_nproc);
	client_pending_op_auth = (int *)malloc(sizeof(int)*armci_nproc);
	if(!server_auths || !clauth || !client_pending_op_auth)
		armci_die("server_init amlloc failed",0);
	for(i=0;i<armci_nproc;i++){
		dst=armci_clus_id(i);
		/*if same node, we dont need to get authorization*/
		/*temporarily disabled */
		/*if(dst==armci_clus_me)continue;*/
		remote.type = CB_NODE_RELATIVE;
         	remote.ndes = _armci_group;
         	remote.node = dst;

	    /*Get a authorization of the entire client buffer,used in get*/
	 	rc = combuf_get_sendright( (Cb_node *)&remote, sizeof(remote),
                                    CLIENT_DIRECTBUF_KEY+i, CLIENT_GET_DIRECTBUF_FIELDNUM, -1, &auth);
          if(rc != COMBUF_SUCCESS){
          	printf("%d:failed\n",armci_me);fflush(stdout);sleep(1);
			armci_die("armci_client_connect_to_servers combuf_get_sendright:",rc);
          }
		clauth[i]=auth;

		/*get authorization to write client count to client*/
		rc = combuf_get_sendright( (Cb_node *)&remote, sizeof(remote),
                                    CLIENT_DIRECTBUF_KEY+i, CLIENT_PENDING_OP_FIELDNUM, -1, &auth);
          if(rc != COMBUF_SUCCESS){
               printf("%d:failed\n",armci_me);fflush(stdout);sleep(1);
               armci_die("armci_client_connect_to_servers pending op combuf_get_sendright:",rc);
          }
		client_pending_op_auth[i]=auth;
		for(k=0;k<numofbuffers;k++){
			rc = combuf_get_sendright( (Cb_node *)&remote, sizeof(remote),
                          CLIENT_DIRECTBUF_KEY+i, CLIENT_DIRECTBUF_FIELDNUM+i+k, -1, &auth);
			if(rc != COMBUF_SUCCESS){
               	printf("%d:failed\n",armci_me);fflush(stdout);sleep(1);
               	armci_die("armci_client_connect_to_servers pending op combuf_get_sendright:",rc);
          	}
			server_auths[i].lbuf_put_auths[k]=auth;
		}

		field_num=CLIENT_SMALLBUF_FIELDNUM;
		key = CLIENT_SMALLBUF_KEY+i;
		for(k=0;k<SMALL_BUFS_PER_PROCESS;k++){
			rc = combuf_get_sendright( (Cb_node *)&remote, sizeof(remote),
                                   key,field_num+k, -1, &auth);
               if(rc != COMBUF_SUCCESS){
                    printf("%d:failed\n",armci_me);fflush(stdout);sleep(1);
                    armci_die("armci_client_connect_to_servers combuf_get_sendright:",rc);
               }
			server_auths[i].put_auths[k]=auth;
		}
		rc = combuf_target( (Cb_node *)&remote, sizeof(remote),\
				 					key, 0, -1, &auth);
          if(rc != COMBUF_SUCCESS) armci_die("combuf_target:",rc);
		server_auths[i].get_auth=auth;

		if(DEBUG_){printf("\n%d: getting authorization=%d from %d and bufkey =%d\n",armci_me,auth,dst,key);fflush(stdout);}
		rc = combuf_target((Cb_node *)&remote,sizeof(remote),\
			(BUF_KEY+i-armci_clus_info[dst].master), 0, -1, &auth);
          if(rc != COMBUF_SUCCESS) 
			armci_die("combuf_target:armci_client_connect_to_servers",rc);
		server_auths[i].clientputbuf_auth=auth;

		if(DEBUG_){
			printf("\n%d:read/write from/to node=%d now\n",armci_me,dst);
			fflush(stdout);
		}
	}
	if(DEBUG_){
		printf("\n:%d:Server successfully initialized\n",armci_me);
		fflush(stdout);
	}
}

void armci_wait_for_server(){
	if(armci_me == armci_master){
# ifndef SERVER_THREAD
     	RestoreSigChldDfl();
     	armci_serv_quit();
     	armci_wait_server_process();
# endif
  }
}	

int find_rcvd_desc_index(int i){
	int j;
	for(j=0;j<SMALL_MSG_NUM+LARGE_MSG_NUM*numofbuffers+1;j++)
		if(sbufreqdescarray[j]==i)return(j);
	return(-1);
}

char * getbuffer(int i){
	return((serverbuffer+(i<SMALL_MSG_NUM?i*SMALL_MSG_SIZE:(SMALL_MSG_NUM*SMALL_MSG_SIZE+(i-SMALL_MSG_NUM)*(LARGE_MSG_SIZE/4)+4))));
}

unsigned int _sr8k_data_server_ev;
unsigned int _sr8k_data_server_smallbuf_ev;
unsigned int _sr8k_data_server_complet=0;
unsigned int _sr8k_server_smallbuf_complete=0;
int _sr8k_server_buffer_index;

void armci_sr8k_data_server(char *buf,int bufindex){
	static Cb_msg msg;
	int rc;
	request_header_t *msginfo = (request_header_t *)buf;
	if(DEBUG_){printf("\n%d:for index %d servbufptr=%p small_buf_count=%d\n",armci_me,bufindex,buf,small_buf_count[msginfo->from]);fflush(stdout);}
	armci_data_server(buf);
#if defined(PIPE_BUFSIZE)
#if 0
	if(msginfo->operation==GET && msginfo->format==STRIDED && msginfo->datalen>2*PIPE_MIN_BUFSIZE)
		combuf_swap(getbuf_locked_auth,0,0);
#endif
#endif
	if((msginfo->operation==PUT || ACC(msginfo->operation) ) && msginfo->tag.ack!=0 &&bufindex>=SMALL_MSG_NUM){
		memset(&msg, 0, sizeof(msg));
     	msg.data.addr =  (char *)(pending_op_count+msginfo->from);
		msg.data.size = sizeof(int);	
		if(DEBUG_){printf("\n%d:got ack=%d from %d for op=%d bufindex=%d\n",armci_me, msginfo->tag.ack,msginfo->from,msginfo->operation,bufindex);fflush(stdout);}
		if(_sr8k_data_server_complet==99)
			combuf_send_complete(_sr8k_data_server_ev,-1,&_sr8k_data_server_complet);
		pending_op_count[msginfo->from]=msginfo->tag.ack;
		rc=combuf_send(&msg,client_pending_op_auth[msginfo->from],sizeof(int)*armci_clus_me,0,&_sr8k_data_server_ev);
		_sr8k_data_server_complet=99;
		if (rc != COMBUF_SUCCESS) armci_die("armci_sr8k_data_server:combuf_send failed",rc);	
	}	
	if(bufindex<SMALL_MSG_NUM){
		small_buf_count[msginfo->from]++;
		memset(&msg, 0, sizeof(msg));
        msg.data.addr =  (char *)(small_buf_count+msginfo->from);
        msg.data.size = sizeof(int);
		if(_sr8k_server_smallbuf_complete==99)
          	combuf_send_complete(_sr8k_data_server_smallbuf_ev,-1,\
								&_sr8k_server_smallbuf_complete);
		rc=combuf_send(&msg,client_pending_op_auth[msginfo->from],\
			sizeof(int)*(armci_clus_me+armci_nproc),0,&_sr8k_data_server_smallbuf_ev);
        _sr8k_server_smallbuf_complete=99;
		if (rc != COMBUF_SUCCESS) 
			armci_die("armci_sr8k_data_server:combuf_send failed",rc);
	}
	else{
          combuf_swap(putbuf_locked_auth,(SMALL_MSG_SIZE*SMALL_MSG_NUM)+(bufindex-SMALL_MSG_NUM)*LARGE_MSG_SIZE/4,0);
	}
		
	if(DEBUG_){
		printf("\n%d:done %d putbuflocked  \n",armci_me,bufindex);
		fflush(stdout);
	}
}


void armci_call_data_server(){
int rc,i;
int index;
int descarraylen;
	descarraylen = SMALL_MSG_NUM+LARGE_MSG_NUM*numofbuffers;
	if(DEBUG_){printf("\nabout to enter server loop\n");fflush(stdout);}
     *server_ready_flag = 1;
	while(1){
		rc=combuf_block_mwait(sbufreqdescarray,descarraylen,COMBUF_WAIT_ONE\
				,-1,sbufrcvdescarray,&sbufrcvnum);
		for(i=0;i<sbufrcvnum;i++){
			index=find_rcvd_desc_index(sbufrcvdescarray[i]);
			_sr8k_server_buffer_index=index;
			if(index<0 || index>descarraylen)
				armci_die("wrong index in armci_call_data_server",index);
			if(DEBUG_){
				printf("\n%d:rcvd data from %d sbufrcvnum=%d\n",armci_me\
								,index,sbufrcvnum);fflush(stdout);
			}
			armci_sr8k_data_server(getbuffer(index),index);	
		}
	}
}
extern long _sr8k_armci_getbuf_ofs;
void armci_rcv_req(void *mesg, void *phdr, void *pdescr,void *pdata,int *buflen)
{
	request_header_t *msginfo = (request_header_t *)mesg;

	if(DEBUG_){printf("\n%d:in armic_rcv_req \n",armci_me);fflush(stdout);}
	if(DEBUG_){
     	printf("%d(server): got %d req (dscrlen=%d datalen=%d) from %d it was sent to msginfo->to=%d\n",
               armci_me, msginfo->operation, msginfo->dscrlen,
               msginfo->datalen, msginfo->from ,msginfo->to);
        	fflush(stdout);
    	}

    	*(void **)phdr = msginfo;
     *buflen = LARGE_MSG_SIZE/4 - sizeof(request_header_t);
    	if(msginfo->bytes) {
		if(DEBUG_){
			printf("\n%d:in armic_rcv_req aboutto set pdata to%p bytes=%d\n"\
			,armci_me,largebufptr,msginfo->bytes);fflush(stdout);
		}
        	*(void **)pdescr = msginfo+1;
		*(void **)pdata = msginfo->dscrlen + (char*)(msginfo+1);  
    	}
	else{
        	*(void**)pdescr = NULL;
	}	
	if(msginfo->operation == ATTACH){
		*(void **)pdata = msginfo->dscrlen + (char*)(msginfo+1);
		 *buflen = SMALL_MSG_SIZE - sizeof(request_header_t);
	}
	if(msginfo->operation == GET) {
		 *(void **)pdata =  msginfo->dscrlen + (char*)(msginfo+1);
		if((msginfo->bytes+sizeof(request_header_t))<SMALL_MSG_SIZE)
			*buflen = SMALL_MSG_SIZE - sizeof(request_header_t);	
#if defined(PIPE_BUFSIZE)
		else if(msginfo->format==STRIDED && msginfo->datalen>2*PIPE_MIN_BUFSIZE){
			/*this is where we use the exclusive get buffer*/
			_sr8k_armci_getbuf_ofs=msginfo->tag.ack;
		//	if(*getbuf_locked-10==msginfo->from)
				*(void **)pdata = _sr8k_serverlargegetbuf;
	//		else
	//			armci_die2("process that locked is not same as the one requested",*getbuf_locked,msginfo->from);
		}
#endif
		else{
			*buflen = LARGE_MSG_SIZE/4 - sizeof(request_header_t);
		}

	}
 
}

void armci_transport_cleanup(){

}
