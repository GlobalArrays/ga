/****************************************************************************** 
* file:    cluster.c
* purpose: Determine cluster info i.e., number of machines and processes
*          running on each of them.
*
*******************************************************************************/

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "message.h"
#include "armcip.h"

#define DEBUG 0
#define DEBUG_HACK_
#define CLUSNODES 2
#define MAX_HOSTNAME 80

/*  print info on how many cluster nodes detected */
#ifdef CLUSTER
#  define PRINT_CLUSTER_INFO 1
#else
#  define PRINT_CLUSTER_INFO 0
#endif

/*** stores cluster configuration ***/
armci_clus_t *armci_clus_info;


static char* merge_names(char *name)
{
    int jump = 1, rem, to, from;
    int lenmes, lenbuf, curlen, totbuflen= armci_nproc*HOSTNAME_LEN;
    int len = strlen(name);
    char *work = malloc(totbuflen);

    if(!work)armci_die("armci: merge_names: malloc failed: ",totbuflen);

    strcpy(work, name); 
    curlen = len+1;

    /* prefix tree merges names in the order of process numbering in log(P)time
     * result = name_1//name_2//...//name_P-1
     */
    do {
       jump *= 2; rem = armci_me%jump;
       if(rem){
              to = armci_me - rem;
              armci_msg_snd(ARMCI_TAG, work, curlen, to);
              break;
       }else{
              from = armci_me + jump/2;
              if(from < armci_nproc){
                 lenbuf = totbuflen - curlen;
                 armci_msg_rcv(ARMCI_TAG, work+curlen, lenbuf, &lenmes, from);
                 curlen += lenmes;
              }
       }
    }while (jump < armci_nproc);
    return(work);
}


static void process_hostlist(char *names)
{
#ifdef CLUSTER

    int i, cluster=0;
    char *s,*master;
    int len, root=0;

    /******** inspect list of machine names to determine locality ********/
    if (armci_me==0){
     
      /* first find out how many cluster nodes we got */
      armci_nclus =1; s=master=names; 
      for(i=1; i < armci_nproc; i++){
        s += strlen(s)+1;
        if(strcmp(s,master)){
          /* we found a new machine name on the list */
          master = s;
          armci_nclus++;
/*          fprintf(stderr,"new name %s len =%d\n",master, strlen(master));*/

        }
      }

      /* allocate memory */ 
      armci_clus_info = (armci_clus_t*)malloc(armci_nclus*sizeof(armci_clus_t));
      if(!armci_clus_info)armci_die("malloc failed for clusinfo",armci_nclus);

      /* fill the data structure  -- go through the list again */ 
      s=names;
      master="*-"; /* impossible hostname */
      cluster =0;
      for(i=0; i < armci_nproc; i++){
        if(strcmp(s,master)){
          /* we found a new machine name on the list */
          master = s;
          armci_clus_info[cluster].nslave=1;
          armci_clus_info[cluster].master=i;
          strcpy(armci_clus_info[cluster].hostname, master); 
          cluster++;
        }else{
          /* the process is still on the same host */
          armci_clus_info[cluster-1].nslave++;
        }
        s += strlen(s)+1;
      }

      if(armci_nclus != cluster)
         armci_die("inconsistency processing clusterinfo",armci_nclus);
    }
    /******** process 0 got all data                             ********/

   /* now broadcast locality info struct to all processes 
    * two steps are needed because of the unknown length of hostname list
    */
    len = sizeof(int);
    armci_msg_brdcst(&armci_nclus, len, root);

    if(armci_me){
      /* allocate memory */ 
      armci_clus_info = (armci_clus_t*)malloc(armci_nclus*sizeof(armci_clus_t));
      if(!armci_clus_info)armci_die("malloc failed for clusinfo",armci_nclus);
    }

    len = sizeof(armci_clus_t)*armci_nclus;
    armci_msg_brdcst(armci_clus_info, len, root);

    /******** all processes 0 got all data                         ********/

    /* now determine current cluster node id by comparing me to master */
    armci_clus_me = armci_nclus-1;
    for(i =0; i< armci_nclus-1; i++)
           if(armci_me < armci_clus_info[i+1].master){
              armci_clus_me=i;
              break;
           }
#else

    armci_nclus= armci_nproc;
    armci_clus_info = (armci_clus_t*)malloc(armci_nclus*sizeof(armci_clus_t));
    if(!armci_clus_info)armci_die("malloc failed for clusinfo",armci_nclus);
    armci_clus_me=0;
    armci_nclus=1;
    strcpy(armci_clus_info[0].hostname, names); 
    armci_clus_info[0].master=0;
    armci_clus_info[0].nslave=armci_nproc;
#endif

    armci_clus_first = armci_clus_info[armci_clus_me].master;
    armci_clus_last = armci_clus_first +armci_clus_info[armci_clus_me].nslave-1;

}
       

void armci_init_clusinfo()
{
  char name[MAX_HOSTNAME], *merged;
  int  i, len, limit, rc;
 
  limit = MAX_HOSTNAME-1;
  rc = gethostname(name, limit);
  if(rc < 0)armci_die("armci: gethostname failed",rc);

  len =  strlen(name);

#ifdef HOSTNAME_TRUNCATE
     /* in some cases (e.g.,SP) we can truncate hostnames to save memory */
     limit = HOSTNAME_LEN-2;
     if(len>limit)name[limit]='\0';
     len =limit;
#else
  if(len >= HOSTNAME_LEN-1)
     armci_die("armci: gethostname overrun name string length",len);
#endif

  if(DEBUG)
     fprintf(stderr,"%d: %s len=%d \n",armci_me, name, strlen(name));


  /******************* development hack *********************/
#ifdef DEBUG_HACK
  name[len]='0'+armci_me/CLUSNODES; 
  name[len+1]='\0';
  len++;
#endif
  
#ifdef CLUSTER
  merged = merge_names(name); /* create hostname list */
  process_hostlist(merged);        /* compute cluster info */
  free(merged);
#else
  process_hostlist(name);        /* compute cluster info */
#endif

  armci_master = armci_clus_info[armci_clus_me].master;

  /******************* development hack *********************/
#ifdef DEBUG_HACK
  for(i=0;i<armci_nclus;i++){
     int len=strlen(armci_clus_info[i].hostname);
/*     fprintf(stderr,"----hostlen=%d\n",len);*/
     armci_clus_info[i].hostname[len-1]='\0';
  }
#endif

  if(PRINT_CLUSTER_INFO && armci_nclus >1 && armci_me ==0){
     printf("ARMCI configured for %d cluster nodes\n", armci_nclus);
     fflush(stdout);
  }

  if(armci_me==0 && DEBUG) for(i=0;i<armci_nclus;i++)
     printf("%s cluster:%d nodes:%d master=%d\n",armci_clus_info[i].hostname,i, 
                         armci_clus_info[i].nslave,armci_clus_info[i].master);

}


/*\ find cluster node id the specified process is on
\*/
int armci_clus_id(int p)
{
int from, to, found, c;

    if(p<0 || p >= armci_nproc)armci_die("armci_clus_id: out of range",p);

    if(p < armci_clus_first){ from = 0; to = armci_clus_me;}
    else {from = armci_clus_me; to = armci_nclus;}

    found = to-1;
    for(c = from; c< to-1; c++)
        if(p < armci_clus_info[c+1].master){
              found=c;
              break;
        }

    return (found);
}
