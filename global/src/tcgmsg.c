/************************* interface to TCGMSG  *****************************/ 

#ifdef SYSV
# include<string.h>
#endif
#include "global.h"
#include "config.h"
#include "message.h"

/*** data required to exploit locality and implement data server ***/
cluster_info_t GA_clus_info[MAX_CLUST];
Integer GA_proc_id;       /* process id of current process --
                           * this is NOT the GA process id
                           */
Integer GA_n_proc;        /* No. of processes */
Integer GA_n_clus;        /* No. of clusters */
Integer GA_clus_id;       /* Logical id of current cluster */



struct cluster_info_struct {
  char *user;                     /* user name */
  char *hostname;                 /* hostname */
  long nslave;                    /* no. slave on this host */
  char *image;                    /* path executable image */
  char *workdir;                  /* work directory */
  long masterid;                  /* process no. of cluster master */
  int  swtchport;                 /* Switch port for alliant hippi */
};

#ifdef SYSV
  extern struct cluster_info_struct SR_clus_info[];
  extern long SR_proc_id;           /* Logical id of current process */
  extern long SR_n_proc;            /* No. of processes excluding dummy */
  extern long SR_n_clus;            /* No. of clusters */
  extern long SR_clus_id;           /* Logical id of current cluster */
#endif


void init_msg_interface()
{
#ifdef SYSV
  long i;

  if(MAX_CLUST < SR_n_clus)
     ga_error("GA:init_msg_interface: MAX_CLUST too small",SR_n_clus);
 
  for(i=0;i<SR_n_clus;i++){
    GA_clus_info[i].nslave = SR_clus_info[i].nslave; 
    GA_clus_info[i].masterid = SR_clus_info[i].masterid; 
    strncpy(GA_clus_info[i].hostname, SR_clus_info[i].hostname, HOSTNAME_LEN);
  }
  GA_proc_id = (Integer)SR_proc_id;
  GA_n_proc  = (Integer)SR_n_proc;
  GA_n_clus  = (Integer)SR_n_clus;
  GA_clus_id = (Integer)SR_clus_id;
#else
  GA_n_clus = 1;
  GA_n_proc = nnodes_();
  GA_clus_id = 0;
#endif
}


