/*********************** server initialization for IWAY ******************/ 

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "global.h"
#include "message.h"
#include "../../server/p2p.h"
#include "farg.h"

/*** data required to exploit locality and implement data server ***/
cluster_info_t GA_clus_info[MAX_CLUST];
Integer GA_proc_id;       /* process id of current process --
                           * this is NOT the GA process id
                           */
Integer GA_n_proc;        /* No. of processes */
Integer GA_n_clus;        /* No. of clusters */
Integer GA_clus_id;       /* Logical id of current cluster */

static get_args(iargc, iargv)
int *iargc;
char ***iargv;
{
#  if!(defined(SUNF77_2) || defined(PARAGON) || defined(SOLARIS)||defined(HPUX))
     *iargc = ARGC_;
     *iargv = ARGV_;
#  else
      extern char *strdup();
#     if defined(SUNF77_2) || defined(PARAGON) || defined (SOLARIS)
         extern int iargc_();
         extern void getarg_();
         int argc = iargc_() + 1;
#     else
#        ifndef EXTNAME
#          define hpargv_ hpargv
#          define hpargc_ hpargc
#        endif
         extern int hpargv_();
         extern int hpargc_();
         int argc = hpargc_();
#     endif
      int i, len, maxlen=256;
      static char *argv[256], arg[256];

      for (i=0; i<argc; i++) {
#        if defined(SUNF77_2) || defined(PARAGON) || defined (SOLARIS)
            getarg_(&i, arg, maxlen);
            for(len = maxlen-2; len && (arg[len] == ' '); len--);
            len++;
#        else
            len = hpargv_(&i, arg, &maxlen);
#        endif
         arg[len] = '\0';
          /* printf("%10s, len=%d\n", arg, len);  fflush(stdout); */
         argv[i] = strdup(arg);
      }
   *iargc = argc;
   *iargv = argv;
#  endif
}


void init_clusters()
{
  int argc;
  char **argv;
  Integer me=(Integer)nodeid_();
  int loc_nodes= (int)nnodes_();
  int ping=77, rping, msglen, rem_nodes;

     /** WARNING: This will not work with C main program **/
     get_args(&argc, &argv);

   
     gethostname(GA_clus_info[0].hostname, HOSTNAME_LEN);
     printf("Server process is running on host:%s\n", GA_clus_info[0].hostname);
     fflush(stdout);

     if(!server_init(argc, argv)){
         GA_n_clus = 1;
         GA_clus_id= 0;
     } else{
         GA_n_clus = 2;
         GA_clus_id= (Integer)server_clusid();
     }

     if(MAX_CLUST < GA_n_clus)
        ga_error("GA:init_clusters: MAX_CLUST too small",GA_n_clus);


     if(GA_n_clus >1){
       /* ping the other server */
       send_to_server(&ping,sizeof(ping));
       recv_from_server(&rping, &msglen);
       if(ping != rping)ga_error("server initialization failed",GA_clus_id);

       /* exchange the numbers of nodes */
       send_to_server(&loc_nodes, sizeof(int));
       recv_from_server(&rem_nodes, &msglen);
       printf("nodes in my cluster: %d in the other cluster: %d\n",
               loc_nodes,rem_nodes);
       fflush(stdout);
     }

     /* fill in the cluster info array for the current cluster */
     GA_clus_info[GA_clus_id].nslave = loc_nodes;
     GA_clus_info[GA_clus_id].masterid = GA_clus_id ? rem_nodes: 0;
     gethostname(GA_clus_info[GA_clus_id].hostname, HOSTNAME_LEN);

     if(GA_n_clus >1){
        /* exchange elements of the cluster array info with the other server */
        send_to_server(&GA_clus_info[GA_clus_id], sizeof(cluster_info_t));
        recv_from_server(&GA_clus_info[1-GA_clus_id], &msglen);

        if(msglen != sizeof(cluster_info_t))
                     ga_error("init_clusters: wrong length",GA_clus_id);
    
        if(rem_nodes != GA_clus_info[1-GA_clus_id].nslave)
                     ga_error("init_clusters: error in nslave",
                              GA_clus_info[1-GA_clus_id].nslave);
     } 
}


void init_msg_interface()
{
  int me=(int)nodeid_();
  int loc_nodes= (int)nnodes_();
  Integer loc_server = loc_nodes -1; 
  Integer type=1, len, root, i;

  GA_proc_id = (Integer)me;


  /* the last node in the cluster becomes the server */
  if(me == loc_server)init_clusters();
/*  printf("init_msg_interface I done: %d\n", GA_clus_id);*/
/*  fflush(stdout);*/

  /* brodcast cluster info array to all members of the cluster */
  len = sizeof(Integer);
  root = (Integer)loc_server;
  brdcst_(&type, &GA_n_clus, &len, &root);
  if(MAX_CLUST < GA_n_clus)
     ga_error("GA:init_msg_interface: MAX_CLUST too small",GA_n_clus);

  len = (sizeof(Integer)+HOSTNAME_LEN)*GA_n_clus;
  brdcst_(&type, GA_clus_info, &len, &root);

  len = sizeof(Integer);
  brdcst_(&type, &GA_clus_id, &len, &root);
  
  GA_n_proc = 0; 
  for(i=0;i<GA_n_clus;i++) GA_n_proc+= GA_clus_info[i].nslave;
  
/*  printf("init_msg_interface done: %d\n", GA_clus_id);*/
/*  fflush(stdout);*/
}
