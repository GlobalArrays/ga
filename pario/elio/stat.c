#include "eliop.h"
#include "chemio.h"

 
/*\ determines directory path for a given file
\*/
int elio_dirname(const char *fname, char *dirname, int len)
{
    size_t flen;
    
    if(len< (flen =strlen(fname))) 
	ELIO_ERROR(LONGFAIL,flen);
    
    while(fname[flen] != '/' && flen >0 ) flen--;
    if(flen==0)strcpy(dirname,".");
    else {strncpy(dirname, fname, flen); dirname[flen]=(char)0;}
    
    return(ELIO_OK);
}


/*\ Stat a file (or path) to determine it's filesystem info
\*/
int  elio_stat(char *fname, stat_t *statinfo)
{
    struct  stat      ufs_stat;
    int bsize;
    
#if defined(PARAGON)
    struct statpfs  *statpfsbuf;
    struct estatfs   estatbuf;
    int              bufsz;
    char             str_avail[64];
#else
    struct  STATVFS   ufs_statfs;
#endif
    
    PABLO_start(PABLO_elio_stat); 
    
#if defined(PARAGON)
    bufsz = sizeof(struct statpfs) + SDIRS_INIT_SIZE;
    if( (statpfsbuf = (struct statpfs *) malloc(bufsz)) == NULL)
        ELIO_ERROR(ALOCFAIL,1);
    if(statpfs(fname, &estatbuf, statpfsbuf, bufsz) == 0)
	{
	    if(estatbuf.f_type == MOUNT_PFS)
		statinfo->fs = ELIO_PFS;
	    else if(estatbuf.f_type == MOUNT_UFS || estatbuf.f_type == MOUNT_NFS)
		statinfo->fs = ELIO_UFS;
	    else
		ELIO_ERROR(FTYPFAIL, 1);

	    /*blocks avail - block=1KB */ 
	    etos(estatbuf.f_bavail, str_avail);
	    if(strlen(str_avail)==10)
		fprintf(stderr,"elio_stat: possible ext. type conversion problem\n");
	    if((bsize=strlen(str_avail))>10)
		ELIO_ERROR(CONVFAIL,(long)bsize);
	    statinfo->avail = atoi(str_avail);
	} 
    else
	ELIO_ERROR(STATFAIL,1);
    free(statpfsbuf);
    return(ELIO_OK);
#else
    
    if(stat(fname, &ufs_stat) != 0)
	    ELIO_ERROR(STATFAIL, 1);

#   if defined(PIOFS)
    /*fprintf(stderr,"filesystem %d\n",ufs_stat.st_vfstype);*/
        /* according to /etc/vfs, "9" means piofs */
        if(ufs_stat.st_vfstype == 9) statinfo->fs = ELIO_PIOFS;
        else
#   endif

    statinfo->fs = ELIO_UFS;
	
    /* only regular or directory files are OK */
    if(!S_ISREG(ufs_stat.st_mode) && !S_ISDIR(ufs_stat.st_mode))
	    ELIO_ERROR(TYPEFAIL, 1);
	
#   if defined(CRAY)
	if(statfs(fname, &ufs_statfs, sizeof(ufs_statfs), 0) != 0)
#   else
        if(STATVFS(fname, &ufs_statfs) != 0)
#   endif
		ELIO_ERROR(STATFAIL,1);
	
#   if defined(CRAY)
        /* f_bfree == f_bavail -- naming changes */
        statinfo->avail = (long) ufs_statfs.f_bfree;
#   else
	statinfo->avail = (long) ufs_statfs.f_bavail;
#   endif

#   ifdef SOLARIS
	bsize = (int) ufs_statfs.f_frsize;
#   else
	bsize = (int) ufs_statfs.f_bsize;
#   endif
    
    switch (bsize) {
    case 512:  statinfo->avail /=2; break;
    case 1024: break;
    case 2048: statinfo->avail *=2; break;
    case 4096: statinfo->avail *=4; break;
    case 8192: statinfo->avail *=8; break;
    default:   { 
		double avail;
		double factor = ((double)bsize)/1024.0;
		avail = factor * (double)statinfo->avail;
		statinfo->avail = (long) avail;
               }
    }
    
#endif
    PABLO_end(PABLO_elio_stat);
    return(ELIO_OK);
}
