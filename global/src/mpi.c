/****************************************************************************** 
* file:    mpi.c
* purpose: Determines locality info i.e., number of machines and processes
*          running on each of them.
*          This information is needed to establish optimal communication 
*          mechanisms (for example shared memory instead of message-passing)
*          between any pair of processes. 
*
*          This file must be used with MPI. If TCGMSG is used, we get the 
*          process locality info directly from the TCGMSG internal structures.
*
* author: Jarek Nieplocha
* date: Wed Sep 27 14:03:39 PDT 1995
*******************************************************************************/

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "global.h"
#include "message.h"
#include "mpi.h"

/*** data required to exploit locality and implement data server ***/
cluster_info_t GA_clus_info[MAX_CLUST];
Integer GA_proc_id;       /* process id of current process --
                           * this is NOT the GA process id
                           */
Integer GA_n_proc;        /* No. of processes */
Integer GA_n_clus;        /* No. of clusters */
Integer GA_clus_id;       /* Logical id of current cluster */

int SR_caught_sigint;     /* for compatibility with TCGMSG interface only */

#define MAX_PROC 1024     /* max no. processes used in data server model */
#define DEBUG 0

static char* merge_names(name, len)
char *name;
Integer  len;
{
     Integer jump = 1, rem, to, from, type=30000;
     Integer me = ga_msg_nodeid_(), nproc = ga_msg_nnodes_();
     Integer lenmes, lenbuf, curlen, totbuflen= nproc*HOSTNAME_LEN;
     char *work = malloc(totbuflen);

     if(!work)ga_error("GA:merge_names: malloc failed: ",0);

     strcpy(work, name); 
     curlen = len+1;

     /* prefix tree merges names in the order of processor numbering
      * in log(P) time 
      * result = name_1//name_2//...//name_P-1
      */
     do {
       jump *= 2; rem = me%jump;
       if(rem){
              to = me - rem;
              ga_msg_snd(type, work, curlen, to);
              break;
       }else{
              from = me + jump/2;
              if(from < nproc){
                 lenbuf = totbuflen - curlen;
                 ga_msg_rcv(type, work+(int)curlen, lenbuf, &lenmes, from, &to);
                 curlen += lenmes;
              }
       }
     }while (jump < nproc);
     /*     work[curlen-1]='\n';*/
     return(work);
}


static void proc_list(names)
char *names;  
{
     
#    if(DATA_SERVER)
     {
        Integer me = ga_msg_nodeid_(), nproc = ga_msg_nnodes_();
        Integer i, cluster=0;
        char *s =names, *master=names;
        Integer len, type=3000, root=0;

        GA_n_clus = 0;
        GA_clus_info[0].nslave=1;
        GA_clus_info[0].masterid=0;
        strcpy(GA_clus_info[0].hostname, names); 

        /* looks through machine names to determine locality */
        if (me==0) for(i=1; i < nproc; i++){
            s += strlen(s)+1;
            if(strcmp(s,master)){
              /* we found a new machine name on the list */
              master = s;
              GA_clus_info[++cluster].nslave=1;
              GA_clus_info[cluster].masterid=(Integer)i;
              strcpy(GA_clus_info[cluster].hostname, master); 
            }else{
              /* the process is still on the same host */
              GA_clus_info[cluster].nslave++;
            }
        }
        GA_n_clus = cluster+1;

       /* now broadcast locality info struct to all processes 
        * two steps are needed because of the unknown length
        */
        len = sizeof(Integer);
        ga_msg_brdcst(type, &GA_n_clus, len, root);
        len = (sizeof(Integer)+HOSTNAME_LEN)*GA_n_clus;
        ga_msg_brdcst(type, GA_clus_info, len, root);
     }
#    else
        MPI_Comm ga_comm;
        int proc;

        ga_mpi_communicator(&ga_comm);
        (void) MPI_Comm_size(ga_comm,&proc); 
        GA_clus_info[0].nslave=proc;
        GA_n_clus = 1;
        GA_clus_info[0].masterid=0;
        strcpy(GA_clus_info[0].hostname, names); 

#    endif

}
       

void init_msg_interface()
{
  char name[HOSTNAME_LEN], *merged;
  int  i, len;
 
  gethostname(name, HOSTNAME_LEN-1);
  len =  strlen(name);
  if(len >= HOSTNAME_LEN-1){
     fprintf(stderr,"GA:init_msg_interface: <%s> name truncated ?",name);
     sleep(1);
  }
  
  merged = merge_names(name, (Integer)len);
  proc_list(merged);
  if(ga_msg_nodeid_()==0 && DEBUG)
    for(i=0;i<GA_n_clus;i++)
       printf("%s cluster:%d nodes:%d\n", GA_clus_info[i].hostname,i, GA_clus_info[i].nslave );
}



/*\ Creates communicator for GA compute processes
\*/
void ga_mpi_communicator(GA_COMM)
MPI_Comm *GA_COMM;
{
MPI_Comm MSG_COMM;

#   ifdef MPI
       /* running with MPI library */
       MSG_COMM = MPI_COMM_WORLD;
#   else
       /* running with TCGMSG-MPI library */
       extern MPI_Comm  TCGMSG_Comm;
       MSG_COMM = TCGMSG_Comm;
#   endif

    if(ClusterMode){

        MPI_Group MSG_GROUP, GA_GROUP;
        int i, *data_servers = (int*)malloc(GA_n_clus*sizeof(int)); 

        if(!data_servers)ga_error("ga_mpi_communicator: malloc failed",0);
        for(i=0; i < GA_n_clus; i++)
           data_servers[i] = GA_clus_info[i].masterid+GA_clus_info[i].nslave-1;
      
        /* exclude data server processes from the group */ 
        MPI_Comm_group(MSG_COMM, &MSG_GROUP); 
        MPI_Group_excl(MSG_GROUP, (int)GA_n_clus, data_servers, &GA_GROUP);
        MPI_Comm_create(MSG_COMM, GA_GROUP, GA_COMM);

    } else{

        *GA_COMM = MSG_COMM;

    }
}


#ifdef MPI
void Error(string, code)
     char *string;
     Integer  code;
{
    fprintf(stdout,"%3ld: %s %ld (%#lx).\n", (long)ga_msg_nodeid_(), string,
           (long)code,(long)code);
    fflush(stdout);
    fprintf(stderr,"%3ld: %s %ld (%#lx).\n", (long)ga_msg_nodeid_(), string,
           (long)code,(long)code);
    MPI_Abort(MPI_COMM_WORLD,(int)code);
}
#endif
