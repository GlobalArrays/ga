/*$Id: tcgmsg.h,v 1.2 1995-02-02 23:13:59 d3g681 Exp $*/
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
#else
  struct cluster_info_struct SR_clus_info[1];
  long SR_proc_id;           /* Logical id of current process */
  long SR_n_proc;            /* No. of processes excluding dummy */
  long SR_n_clus;            /* No. of clusters */
  long SR_clus_id;           /* Logical id of current cluster */
#endif
