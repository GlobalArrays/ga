
#if HAVE_CONFIG_H
#   include "config.h"
#endif

#include <assert.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "papi.h"
#include "typesf2c.h"

FILE *fptrace=NULL;
double first_wtime;
int me, nproc;

#if HAVE_PROGNAME
extern const char * PROGNAME;
#endif

#define LOHI_BUFSIZE 200
#define LOHI_SLOTS 20

static char **lohi_bufs;
static int pnga_set_data_last_ndim=-1;

static Integer sum_intp(int ndim, Integer *lohi) {
    Integer sum=0;

    assert(ndim>=0);
    if (NULL != lohi) {
        int i;
        for (i=0; i<ndim; i++) {
            sum += lohi[i];
        }
    }

    return sum;
}

static char* expand_voidp(int type, void *value, int slot) {
    char *str=NULL, *current_str=NULL;
    int size_written;
    int total_written = 0;

    assert(slot >= 0 && slot < LOHI_SLOTS);
    str = lohi_bufs[slot];
    current_str = str;
    if (NULL == value) {
        size_written = sprintf(current_str, "NULL");
        total_written += size_written;
        assert(size_written > 0);
        assert(total_written < LOHI_BUFSIZE);
        current_str += size_written;
    }
    else {
        switch (type) {
            case C_INT:     size_written = sprintf(current_str, "%d",
                            *((int*)value));
                            break;
            case C_LONG:    size_written = sprintf(current_str, "%ld",
                            *((long*)value));
                            break;
            case C_LONGLONG:size_written = sprintf(current_str, "%lld",
                            *((long long*)value));
                            break;
            case C_FLOAT:   size_written = sprintf(current_str, "%f",
                            *((float*)value));
                            break;
            case C_DBL:     size_written = sprintf(current_str, "%lf",
                            *((double*)value));
                            break;
            case C_SCPL:    size_written = sprintf(current_str, "%f#%f",
                            ((SingleComplex*)value)->real,
                            ((SingleComplex*)value)->imag);
                            break;
            case C_DCPL:    size_written = sprintf(current_str, "%lf#%lf",
                            ((DoubleComplex*)value)->real,
                            ((DoubleComplex*)value)->imag);
                            break;
        }
        total_written += size_written;
        assert(size_written > 0);
        assert(total_written < LOHI_BUFSIZE);
        current_str += size_written;
    }

    return str;
}

static char* expand_intp(int ndim, Integer *lohi, int slot) {
    char *str=NULL, *current_str=NULL;
    int i;
    int size_written;
    int total_written = 0;

    assert(ndim>=0);
    assert(slot >= 0 && slot < LOHI_SLOTS);
    str = lohi_bufs[slot];
    current_str = str;
    if (NULL == lohi) {
        size_written = sprintf(current_str, "{NULL}");
        total_written += size_written;
        assert(size_written > 0);
        assert(total_written < LOHI_BUFSIZE);
        current_str += size_written;
    }
    else if (0 == ndim) {
        size_written = sprintf(current_str, "{}");
        total_written += size_written;
        assert(size_written > 0);
        assert(total_written < LOHI_BUFSIZE);
        current_str += size_written;
    }
    else {
        size_written = sprintf(current_str, "{%ld", lohi[0]);
        total_written += size_written;
        assert(size_written > 0);
        assert(total_written < LOHI_BUFSIZE);
        current_str += size_written;
        for (i=1; i<ndim; i++) {
            size_written = sprintf(current_str, ":%ld", lohi[i]);
            total_written += size_written;
            assert(size_written > 0);
            assert(total_written < LOHI_BUFSIZE);
            current_str += size_written;
        }
        size_written = sprintf(current_str, "}\0");
        total_written += size_written;
        assert(size_written > 0);
        assert(total_written < LOHI_BUFSIZE);
        current_str += size_written;
    }

    return str;
}  

static void init_lohi_bufs() {
    int i;
    lohi_bufs = (char**)malloc(sizeof(char*)*LOHI_SLOTS);
    for (i=0; i<LOHI_SLOTS; i++) {
        lohi_bufs[i] = (char*)malloc(LOHI_BUFSIZE);
    }
}

static void free_lohi_bufs() {
    int i;
    for (i=0; i<LOHI_SLOTS; i++) {
        free(lohi_bufs[i]);
    }
    free(lohi_bufs);
}

static void reset_lohi_bufs() {
    int i;
    for (i=0; i<LOHI_SLOTS; i++) {
        memset(&(lohi_bufs[i][0]), 0, LOHI_BUFSIZE);
    }
}

static void trace_finalize() {
    fclose(fptrace);
    free_lohi_bufs();
}

static void trace_initialize() {
    /* create files to write trace data */
    char *profile_dir=NULL;
    const char *program_name=NULL;
    char *file_name=NULL;
    struct stat f_stat;

    PMPI_Barrier(MPI_COMM_WORLD);
    PMPI_Comm_rank(MPI_COMM_WORLD, &me);
    PMPI_Comm_size(MPI_COMM_WORLD, &nproc);

    first_wtime = MPI_Wtime();
    init_lohi_bufs();

    profile_dir = getenv("PNGA_PROFILE_DIR");
#if HAVE_PROGNAME
    program_name = PROGNAME;
#else
    program_name = "unknown";
#endif
    if (0 == me) {
        int ret;

        if (!profile_dir) {
            fprintf(stderr, "You need to set PNGA_PROFILE_DIR env var\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fprintf(stderr, "PNGA_PROFILE_DIR=%s\n", profile_dir);
        if (-1 == stat(profile_dir, &f_stat)) {
            perror("stat");
            fprintf(stderr, "Cannot successfully stat to PNGA_PROFILE_DIR.\n");
            fprintf(stderr, "Check %s profile dir\n", profile_dir);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        file_name = (char *)malloc(strlen(profile_dir)
                + 1 /* / */
                + strlen(program_name)
                + 2 /* NULL termination */);
        assert(file_name);
        sprintf(file_name, "%s/%s%c\n", profile_dir, program_name, '\0');
        ret = mkdir(file_name, 0755);
        if (ret) {
            perror("mkdir");
            fprintf(stderr, "%d: profile sub-directory creation failed: pathname=%s: exiting\n", me, file_name);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        free(file_name);
    }
    PMPI_Barrier(MPI_COMM_WORLD);
    file_name = (char *)malloc(strlen(profile_dir)
            + 1 /* / */
            + strlen(program_name)
            + 1 /* / */
            + 7 /* mpi id */
            + 6 /* .trace */
            + 2 /* NULL termination */);
    assert(file_name);
    sprintf(file_name,"%s/%s/%07d.trace%c",profile_dir,program_name,me,'\0');
    fptrace = fopen(file_name,"w");
    if(!fptrace) {
        perror("fopen");
        printf("%d: trace file creation failed: file_name=%s: exiting\n", me, file_name);
        exit(0);
    }
    free(file_name);
}



void wnga_abs_value(Integer g_a)
{
    fprintf(fptrace, "%lf,pnga_abs_value,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    pnga_abs_value(g_a);
    fprintf(fptrace, "%lf,/pnga_abs_value,(%ld)\n",MPI_Wtime()-first_wtime,g_a);

}


void wnga_abs_value_patch(Integer g_a, Integer *lo, Integer *hi)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_abs_value_patch,(%ld;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,0),expand_intp(_ndim,hi,1));
    pnga_abs_value_patch(g_a, lo, hi);
    fprintf(fptrace, "%lf,/pnga_abs_value_patch,(%ld;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,2),expand_intp(_ndim,hi,3));

}


void wnga_acc(Integer g_a, Integer *lo, Integer *hi, void *buf, Integer *ld, void *alpha)
{
    int _ndim = pnga_ndim(g_a);
    Integer _atype;
    pnga_inquire_type(g_a, &_atype);
    fprintf(fptrace, "%lf,pnga_acc,(%ld;%s;%s;%p;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,0),expand_intp(_ndim,hi,1),buf,expand_intp(_ndim-1,ld,2),expand_voidp(_atype,alpha,3));
    pnga_acc(g_a, lo, hi, buf, ld, alpha);
    fprintf(fptrace, "%lf,/pnga_acc,(%ld;%s;%s;%p;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,4),expand_intp(_ndim,hi,5),buf,expand_intp(_ndim-1,ld,6),expand_voidp(_atype,alpha,7));

}


void wnga_access_block_grid_idx(Integer g_a, Integer *subscript, AccessIndex *index, Integer *ld)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_access_block_grid_idx,(%ld;%s;%p;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,subscript,0),index,expand_intp(_ndim-1,ld,1));
    pnga_access_block_grid_idx(g_a, subscript, index, ld);
    fprintf(fptrace, "%lf,/pnga_access_block_grid_idx,(%ld;%s;%p;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,subscript,2),index,expand_intp(_ndim-1,ld,3));

}


void wnga_access_block_grid_ptr(Integer g_a, Integer *index, void *ptr, Integer *ld)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_access_block_grid_ptr,(%ld;%p;%p;%s)\n",MPI_Wtime()-first_wtime,g_a,index,ptr,expand_intp(_ndim-1,ld,0));
    pnga_access_block_grid_ptr(g_a, index, ptr, ld);
    fprintf(fptrace, "%lf,/pnga_access_block_grid_ptr,(%ld;%p;%p;%s)\n",MPI_Wtime()-first_wtime,g_a,index,ptr,expand_intp(_ndim-1,ld,1));

}


void wnga_access_block_idx(Integer g_a, Integer idx, AccessIndex *index, Integer *ld)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_access_block_idx,(%ld;%ld;%p;%s)\n",MPI_Wtime()-first_wtime,g_a,idx,index,expand_intp(_ndim-1,ld,0));
    pnga_access_block_idx(g_a, idx, index, ld);
    fprintf(fptrace, "%lf,/pnga_access_block_idx,(%ld;%ld;%p;%s)\n",MPI_Wtime()-first_wtime,g_a,idx,index,expand_intp(_ndim-1,ld,1));

}


void wnga_access_block_ptr(Integer g_a, Integer idx, void *ptr, Integer *ld)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_access_block_ptr,(%ld;%ld;%p;%s)\n",MPI_Wtime()-first_wtime,g_a,idx,ptr,expand_intp(_ndim-1,ld,0));
    pnga_access_block_ptr(g_a, idx, ptr, ld);
    fprintf(fptrace, "%lf,/pnga_access_block_ptr,(%ld;%ld;%p;%s)\n",MPI_Wtime()-first_wtime,g_a,idx,ptr,expand_intp(_ndim-1,ld,1));

}


void wnga_access_block_segment_idx(Integer g_a, Integer proc, AccessIndex *index, Integer *len)
{
    fprintf(fptrace, "%lf,pnga_access_block_segment_idx,(%ld;%ld;%p;%p)\n",MPI_Wtime()-first_wtime,g_a,proc,index,len);
    pnga_access_block_segment_idx(g_a, proc, index, len);
    fprintf(fptrace, "%lf,/pnga_access_block_segment_idx,(%ld;%ld;%p;%p)\n",MPI_Wtime()-first_wtime,g_a,proc,index,len);

}


void wnga_access_block_segment_ptr(Integer g_a, Integer proc, void *ptr, Integer *len)
{
    fprintf(fptrace, "%lf,pnga_access_block_segment_ptr,(%ld;%ld;%p;%p)\n",MPI_Wtime()-first_wtime,g_a,proc,ptr,len);
    pnga_access_block_segment_ptr(g_a, proc, ptr, len);
    fprintf(fptrace, "%lf,/pnga_access_block_segment_ptr,(%ld;%ld;%p;%p)\n",MPI_Wtime()-first_wtime,g_a,proc,ptr,len);

}


void wnga_access_ghost_element(Integer g_a, AccessIndex *index, Integer subscript[], Integer ld[])
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_access_ghost_element,(%ld;%p;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,index,expand_intp(_ndim,subscript,0),expand_intp(_ndim-1,ld,1));
    pnga_access_ghost_element(g_a, index, subscript, ld);
    fprintf(fptrace, "%lf,/pnga_access_ghost_element,(%ld;%p;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,index,expand_intp(_ndim,subscript,2),expand_intp(_ndim-1,ld,3));

}


void wnga_access_ghost_element_ptr(Integer g_a, void *ptr, Integer subscript[], Integer ld[])
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_access_ghost_element_ptr,(%ld;%p;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,ptr,expand_intp(_ndim,subscript,0),expand_intp(_ndim-1,ld,1));
    pnga_access_ghost_element_ptr(g_a, ptr, subscript, ld);
    fprintf(fptrace, "%lf,/pnga_access_ghost_element_ptr,(%ld;%p;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,ptr,expand_intp(_ndim,subscript,2),expand_intp(_ndim-1,ld,3));

}


void wnga_access_ghost_ptr(Integer g_a, Integer dims[], void *ptr, Integer ld[])
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_access_ghost_ptr,(%ld;%s;%p;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,dims,0),ptr,expand_intp(_ndim-1,ld,1));
    pnga_access_ghost_ptr(g_a, dims, ptr, ld);
    fprintf(fptrace, "%lf,/pnga_access_ghost_ptr,(%ld;%s;%p;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,dims,2),ptr,expand_intp(_ndim-1,ld,3));

}


void wnga_access_ghosts(Integer g_a, Integer dims[], AccessIndex *index, Integer ld[])
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_access_ghosts,(%ld;%s;%p;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,dims,0),index,expand_intp(_ndim-1,ld,1));
    pnga_access_ghosts(g_a, dims, index, ld);
    fprintf(fptrace, "%lf,/pnga_access_ghosts,(%ld;%s;%p;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,dims,2),index,expand_intp(_ndim-1,ld,3));

}


void wnga_access_idx(Integer g_a, Integer *lo, Integer *hi, AccessIndex *index, Integer *ld)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_access_idx,(%ld;%s;%s;%p;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,0),expand_intp(_ndim,hi,1),index,expand_intp(_ndim-1,ld,2));
    pnga_access_idx(g_a, lo, hi, index, ld);
    fprintf(fptrace, "%lf,/pnga_access_idx,(%ld;%s;%s;%p;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,3),expand_intp(_ndim,hi,4),index,expand_intp(_ndim-1,ld,5));

}


void wnga_access_ptr(Integer g_a, Integer *lo, Integer *hi, void *ptr, Integer *ld)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_access_ptr,(%ld;%s;%s;%p;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,0),expand_intp(_ndim,hi,1),ptr,expand_intp(_ndim-1,ld,2));
    pnga_access_ptr(g_a, lo, hi, ptr, ld);
    fprintf(fptrace, "%lf,/pnga_access_ptr,(%ld;%s;%s;%p;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,3),expand_intp(_ndim,hi,4),ptr,expand_intp(_ndim-1,ld,5));

}


void wnga_add(void *alpha, Integer g_a, void *beta, Integer g_b, Integer g_c)
{
    Integer _atype;
    Integer _btype;
    pnga_inquire_type(g_a, &_atype);
    pnga_inquire_type(g_b, &_btype);
    fprintf(fptrace, "%lf,pnga_add,(%s;%ld;%s;%ld;%ld)\n",MPI_Wtime()-first_wtime,expand_voidp(_atype,alpha,0),g_a,expand_voidp(_btype,beta,1),g_b,g_c);
    pnga_add(alpha, g_a, beta, g_b, g_c);
    fprintf(fptrace, "%lf,/pnga_add,(%s;%ld;%s;%ld;%ld)\n",MPI_Wtime()-first_wtime,expand_voidp(_atype,alpha,2),g_a,expand_voidp(_btype,beta,3),g_b,g_c);

}


void wnga_add_constant(Integer g_a, void *alpha)
{
    Integer _atype;
    pnga_inquire_type(g_a, &_atype);
    fprintf(fptrace, "%lf,pnga_add_constant,(%ld;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_voidp(_atype,alpha,0));
    pnga_add_constant(g_a, alpha);
    fprintf(fptrace, "%lf,/pnga_add_constant,(%ld;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_voidp(_atype,alpha,1));

}


void wnga_add_constant_patch(Integer g_a, Integer *lo, Integer *hi, void *alpha)
{
    int _ndim = pnga_ndim(g_a);
    Integer _atype;
    pnga_inquire_type(g_a, &_atype);
    fprintf(fptrace, "%lf,pnga_add_constant_patch,(%ld;%s;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,0),expand_intp(_ndim,hi,1),expand_voidp(_atype,alpha,2));
    pnga_add_constant_patch(g_a, lo, hi, alpha);
    fprintf(fptrace, "%lf,/pnga_add_constant_patch,(%ld;%s;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,3),expand_intp(_ndim,hi,4),expand_voidp(_atype,alpha,5));

}


void wnga_add_diagonal(Integer g_a, Integer g_v)
{
    fprintf(fptrace, "%lf,pnga_add_diagonal,(%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,g_v);
    pnga_add_diagonal(g_a, g_v);
    fprintf(fptrace, "%lf,/pnga_add_diagonal,(%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,g_v);

}


void wnga_add_patch(void *alpha, Integer g_a, Integer *alo, Integer *ahi, void *beta, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi)
{
    int _ndim = pnga_ndim(g_a);
    Integer _atype;
    Integer _btype;
    pnga_inquire_type(g_a, &_atype);
    pnga_inquire_type(g_b, &_btype);
    fprintf(fptrace, "%lf,pnga_add_patch,(%s;%ld;%s;%s;%s;%ld;%s;%s;%ld;%s;%s)\n",MPI_Wtime()-first_wtime,expand_voidp(_atype,alpha,0),g_a,expand_intp(_ndim,alo,1),expand_intp(_ndim,ahi,2),expand_voidp(_btype,beta,3),g_b,expand_intp(_ndim,blo,4),expand_intp(_ndim,bhi,5),g_c,expand_intp(_ndim,clo,6),expand_intp(_ndim,chi,7));
    pnga_add_patch(alpha, g_a, alo, ahi, beta, g_b, blo, bhi, g_c, clo, chi);
    fprintf(fptrace, "%lf,/pnga_add_patch,(%s;%ld;%s;%s;%s;%ld;%s;%s;%ld;%s;%s)\n",MPI_Wtime()-first_wtime,expand_voidp(_atype,alpha,8),g_a,expand_intp(_ndim,alo,9),expand_intp(_ndim,ahi,10),expand_voidp(_btype,beta,11),g_b,expand_intp(_ndim,blo,12),expand_intp(_ndim,bhi,13),g_c,expand_intp(_ndim,clo,14),expand_intp(_ndim,chi,15));

}


logical wnga_allocate(Integer g_a)
{
    logical retval;
    fprintf(fptrace, "%lf,pnga_allocate,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    retval = pnga_allocate(g_a);
    fprintf(fptrace, "%lf,/pnga_allocate,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    return retval;

}


void wnga_bin_index(Integer g_bin, Integer g_cnt, Integer g_off, Integer *values, Integer *subs, Integer n, Integer sortit)
{
    fprintf(fptrace, "%lf,pnga_bin_index,(%ld;%ld;%ld;%p;%p;%ld;%ld)\n",MPI_Wtime()-first_wtime,g_bin,g_cnt,g_off,values,subs,n,sortit);
    pnga_bin_index(g_bin, g_cnt, g_off, values, subs, n, sortit);
    fprintf(fptrace, "%lf,/pnga_bin_index,(%ld;%ld;%ld;%p;%p;%ld;%ld)\n",MPI_Wtime()-first_wtime,g_bin,g_cnt,g_off,values,subs,n,sortit);

}


void wnga_bin_sorter(Integer g_bin, Integer g_cnt, Integer g_off)
{
    fprintf(fptrace, "%lf,pnga_bin_sorter,(%ld;%ld;%ld)\n",MPI_Wtime()-first_wtime,g_bin,g_cnt,g_off);
    pnga_bin_sorter(g_bin, g_cnt, g_off);
    fprintf(fptrace, "%lf,/pnga_bin_sorter,(%ld;%ld;%ld)\n",MPI_Wtime()-first_wtime,g_bin,g_cnt,g_off);

}


void wnga_brdcst(Integer type, void *buf, Integer len, Integer originator)
{
    fprintf(fptrace, "%lf,pnga_brdcst,(%ld;%p;%ld;%ld)\n",MPI_Wtime()-first_wtime,type,buf,len,originator);
    pnga_brdcst(type, buf, len, originator);
    fprintf(fptrace, "%lf,/pnga_brdcst,(%ld;%p;%ld;%ld)\n",MPI_Wtime()-first_wtime,type,buf,len,originator);

}


void wnga_check_handle(Integer g_a, char *string)
{
    fprintf(fptrace, "%lf,pnga_check_handle,(%ld;%s)\n",MPI_Wtime()-first_wtime,g_a,string);
    pnga_check_handle(g_a, string);
    fprintf(fptrace, "%lf,/pnga_check_handle,(%ld;%s)\n",MPI_Wtime()-first_wtime,g_a,string);

}


Integer wnga_cluster_nnodes()
{
    Integer retval;
    fprintf(fptrace, "%lf,pnga_cluster_nnodes,\n",MPI_Wtime()-first_wtime);
    retval = pnga_cluster_nnodes();
    fprintf(fptrace, "%lf,/pnga_cluster_nnodes,\n",MPI_Wtime()-first_wtime);
    return retval;

}


Integer wnga_cluster_nodeid()
{
    Integer retval;
    fprintf(fptrace, "%lf,pnga_cluster_nodeid,\n",MPI_Wtime()-first_wtime);
    retval = pnga_cluster_nodeid();
    fprintf(fptrace, "%lf,/pnga_cluster_nodeid,\n",MPI_Wtime()-first_wtime);
    return retval;

}


Integer wnga_cluster_nprocs(Integer node)
{
    Integer retval;
    fprintf(fptrace, "%lf,pnga_cluster_nprocs,(%ld)\n",MPI_Wtime()-first_wtime,node);
    retval = pnga_cluster_nprocs(node);
    fprintf(fptrace, "%lf,/pnga_cluster_nprocs,(%ld)\n",MPI_Wtime()-first_wtime,node);
    return retval;

}


Integer wnga_cluster_proc_nodeid(Integer proc)
{
    Integer retval;
    fprintf(fptrace, "%lf,pnga_cluster_proc_nodeid,(%ld)\n",MPI_Wtime()-first_wtime,proc);
    retval = pnga_cluster_proc_nodeid(proc);
    fprintf(fptrace, "%lf,/pnga_cluster_proc_nodeid,(%ld)\n",MPI_Wtime()-first_wtime,proc);
    return retval;

}


Integer wnga_cluster_procid(Integer node, Integer loc_proc_id)
{
    Integer retval;
    fprintf(fptrace, "%lf,pnga_cluster_procid,(%ld;%ld)\n",MPI_Wtime()-first_wtime,node,loc_proc_id);
    retval = pnga_cluster_procid(node, loc_proc_id);
    fprintf(fptrace, "%lf,/pnga_cluster_procid,(%ld;%ld)\n",MPI_Wtime()-first_wtime,node,loc_proc_id);
    return retval;

}


logical wnga_comp_patch(Integer andim, Integer *alo, Integer *ahi, Integer bndim, Integer *blo, Integer *bhi)
{
    logical retval;
    fprintf(fptrace, "%lf,pnga_comp_patch,(%ld;%s;%s;%ld;%s;%s)\n",MPI_Wtime()-first_wtime,andim,expand_intp(andim,alo,0),expand_intp(andim,ahi,1),bndim,expand_intp(andim,blo,2),expand_intp(andim,bhi,3));
    retval = pnga_comp_patch(andim, alo, ahi, bndim, blo, bhi);
    fprintf(fptrace, "%lf,/pnga_comp_patch,(%ld;%s;%s;%ld;%s;%s)\n",MPI_Wtime()-first_wtime,andim,expand_intp(andim,alo,4),expand_intp(andim,ahi,5),bndim,expand_intp(andim,blo,6),expand_intp(andim,bhi,7));
    return retval;

}


logical wnga_compare_distr(Integer g_a, Integer g_b)
{
    logical retval;
    fprintf(fptrace, "%lf,pnga_compare_distr,(%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,g_b);
    retval = pnga_compare_distr(g_a, g_b);
    fprintf(fptrace, "%lf,/pnga_compare_distr,(%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,g_b);
    return retval;

}


void wnga_copy(Integer g_a, Integer g_b)
{
    fprintf(fptrace, "%lf,pnga_copy,(%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,g_b);
    pnga_copy(g_a, g_b);
    fprintf(fptrace, "%lf,/pnga_copy,(%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,g_b);

}


void wnga_copy_patch(char *trans, Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_copy_patch,(%s;%ld;%s;%s;%ld;%s;%s)\n",MPI_Wtime()-first_wtime,trans,g_a,expand_intp(_ndim,alo,0),expand_intp(_ndim,ahi,1),g_b,expand_intp(_ndim,blo,2),expand_intp(_ndim,bhi,3));
    pnga_copy_patch(trans, g_a, alo, ahi, g_b, blo, bhi);
    fprintf(fptrace, "%lf,/pnga_copy_patch,(%s;%ld;%s;%s;%ld;%s;%s)\n",MPI_Wtime()-first_wtime,trans,g_a,expand_intp(_ndim,alo,4),expand_intp(_ndim,ahi,5),g_b,expand_intp(_ndim,blo,6),expand_intp(_ndim,bhi,7));

}


void wnga_copy_patch_dp(char *t_a, Integer g_a, Integer ailo, Integer aihi, Integer ajlo, Integer ajhi, Integer g_b, Integer bilo, Integer bihi, Integer bjlo, Integer bjhi)
{
    fprintf(fptrace, "%lf,pnga_copy_patch_dp,(%s;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld)\n",MPI_Wtime()-first_wtime,t_a,g_a,ailo,aihi,ajlo,ajhi,g_b,bilo,bihi,bjlo,bjhi);
    pnga_copy_patch_dp(t_a, g_a, ailo, aihi, ajlo, ajhi, g_b, bilo, bihi, bjlo, bjhi);
    fprintf(fptrace, "%lf,/pnga_copy_patch_dp,(%s;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld)\n",MPI_Wtime()-first_wtime,t_a,g_a,ailo,aihi,ajlo,ajhi,g_b,bilo,bihi,bjlo,bjhi);

}


logical wnga_create(Integer type, Integer ndim, Integer *dims, char *name, Integer *chunk, Integer *g_a)
{
    logical retval;
    fprintf(fptrace, "%lf,pnga_create,(%ld;%ld;%s;%s;%s;%p)\n",MPI_Wtime()-first_wtime,type,ndim,expand_intp(ndim,dims,0),name,expand_intp(ndim,chunk,1),g_a);
    retval = pnga_create(type, ndim, dims, name, chunk, g_a);
    fprintf(fptrace, "%lf,/pnga_create,(%ld;%ld;%s;%s;%s;%p)\n",MPI_Wtime()-first_wtime,type,ndim,expand_intp(ndim,dims,2),name,expand_intp(ndim,chunk,3),g_a);
    return retval;

}


logical wnga_create_bin_range(Integer g_bin, Integer g_cnt, Integer g_off, Integer *g_range)
{
    logical retval;
    fprintf(fptrace, "%lf,pnga_create_bin_range,(%ld;%ld;%ld;%p)\n",MPI_Wtime()-first_wtime,g_bin,g_cnt,g_off,g_range);
    retval = pnga_create_bin_range(g_bin, g_cnt, g_off, g_range);
    fprintf(fptrace, "%lf,/pnga_create_bin_range,(%ld;%ld;%ld;%p)\n",MPI_Wtime()-first_wtime,g_bin,g_cnt,g_off,g_range);
    return retval;

}


logical wnga_create_config(Integer type, Integer ndim, Integer *dims, char *name, Integer *chunk, Integer p_handle, Integer *g_a)
{
    logical retval;
    fprintf(fptrace, "%lf,pnga_create_config,(%ld;%ld;%s;%s;%s;%ld;%p)\n",MPI_Wtime()-first_wtime,type,ndim,expand_intp(ndim,dims,0),name,expand_intp(ndim,chunk,1),p_handle,g_a);
    retval = pnga_create_config(type, ndim, dims, name, chunk, p_handle, g_a);
    fprintf(fptrace, "%lf,/pnga_create_config,(%ld;%ld;%s;%s;%s;%ld;%p)\n",MPI_Wtime()-first_wtime,type,ndim,expand_intp(ndim,dims,2),name,expand_intp(ndim,chunk,3),p_handle,g_a);
    return retval;

}


logical wnga_create_ghosts(Integer type, Integer ndim, Integer *dims, Integer *width, char *name, Integer *chunk, Integer *g_a)
{
    logical retval;
    fprintf(fptrace, "%lf,pnga_create_ghosts,(%ld;%ld;%s;%s;%s;%s;%p)\n",MPI_Wtime()-first_wtime,type,ndim,expand_intp(ndim,dims,0),expand_intp(ndim,width,1),name,expand_intp(ndim,chunk,2),g_a);
    retval = pnga_create_ghosts(type, ndim, dims, width, name, chunk, g_a);
    fprintf(fptrace, "%lf,/pnga_create_ghosts,(%ld;%ld;%s;%s;%s;%s;%p)\n",MPI_Wtime()-first_wtime,type,ndim,expand_intp(ndim,dims,3),expand_intp(ndim,width,4),name,expand_intp(ndim,chunk,5),g_a);
    return retval;

}


logical wnga_create_ghosts_config(Integer type, Integer ndim, Integer *dims, Integer *width, char *name, Integer *chunk, Integer p_handle, Integer *g_a)
{
    logical retval;
    fprintf(fptrace, "%lf,pnga_create_ghosts_config,(%ld;%ld;%s;%s;%s;%s;%ld;%p)\n",MPI_Wtime()-first_wtime,type,ndim,expand_intp(ndim,dims,0),expand_intp(ndim,width,1),name,expand_intp(ndim,chunk,2),p_handle,g_a);
    retval = pnga_create_ghosts_config(type, ndim, dims, width, name, chunk, p_handle, g_a);
    fprintf(fptrace, "%lf,/pnga_create_ghosts_config,(%ld;%ld;%s;%s;%s;%s;%ld;%p)\n",MPI_Wtime()-first_wtime,type,ndim,expand_intp(ndim,dims,3),expand_intp(ndim,width,4),name,expand_intp(ndim,chunk,5),p_handle,g_a);
    return retval;

}


logical wnga_create_ghosts_irreg(Integer type, Integer ndim, Integer *dims, Integer *width, char *name, Integer *map, Integer *block, Integer *g_a)
{
    logical retval;
    fprintf(fptrace, "%lf,pnga_create_ghosts_irreg,(%ld;%ld;%s;%s;%s;%s;%s;%p)\n",MPI_Wtime()-first_wtime,type,ndim,expand_intp(ndim,dims,0),expand_intp(ndim,width,1),name,expand_intp(sum_intp(ndim,block),map,2),expand_intp(ndim,block,3),g_a);
    retval = pnga_create_ghosts_irreg(type, ndim, dims, width, name, map, block, g_a);
    fprintf(fptrace, "%lf,/pnga_create_ghosts_irreg,(%ld;%ld;%s;%s;%s;%s;%s;%p)\n",MPI_Wtime()-first_wtime,type,ndim,expand_intp(ndim,dims,4),expand_intp(ndim,width,5),name,expand_intp(sum_intp(ndim,block),map,6),expand_intp(ndim,block,7),g_a);
    return retval;

}


logical wnga_create_ghosts_irreg_config(Integer type, Integer ndim, Integer *dims, Integer *width, char *name, Integer *map, Integer *block, Integer p_handle, Integer *g_a)
{
    logical retval;
    fprintf(fptrace, "%lf,pnga_create_ghosts_irreg_config,(%ld;%ld;%s;%s;%s;%s;%s;%ld;%p)\n",MPI_Wtime()-first_wtime,type,ndim,expand_intp(ndim,dims,0),expand_intp(ndim,width,1),name,expand_intp(sum_intp(ndim,block),map,2),expand_intp(ndim,block,3),p_handle,g_a);
    retval = pnga_create_ghosts_irreg_config(type, ndim, dims, width, name, map, block, p_handle, g_a);
    fprintf(fptrace, "%lf,/pnga_create_ghosts_irreg_config,(%ld;%ld;%s;%s;%s;%s;%s;%ld;%p)\n",MPI_Wtime()-first_wtime,type,ndim,expand_intp(ndim,dims,4),expand_intp(ndim,width,5),name,expand_intp(sum_intp(ndim,block),map,6),expand_intp(ndim,block,7),p_handle,g_a);
    return retval;

}


Integer wnga_create_handle()
{
    Integer retval;
    fprintf(fptrace, "%lf,pnga_create_handle,\n",MPI_Wtime()-first_wtime);
    retval = pnga_create_handle();
    fprintf(fptrace, "%lf,/pnga_create_handle,\n",MPI_Wtime()-first_wtime);
    return retval;

}


logical wnga_create_irreg(Integer type, Integer ndim, Integer *dims, char *name, Integer *map, Integer *block, Integer *g_a)
{
    logical retval;
    fprintf(fptrace, "%lf,pnga_create_irreg,(%ld;%ld;%s;%s;%s;%s;%p)\n",MPI_Wtime()-first_wtime,type,ndim,expand_intp(ndim,dims,0),name,expand_intp(sum_intp(ndim,block),map,1),expand_intp(ndim,block,2),g_a);
    retval = pnga_create_irreg(type, ndim, dims, name, map, block, g_a);
    fprintf(fptrace, "%lf,/pnga_create_irreg,(%ld;%ld;%s;%s;%s;%s;%p)\n",MPI_Wtime()-first_wtime,type,ndim,expand_intp(ndim,dims,3),name,expand_intp(sum_intp(ndim,block),map,4),expand_intp(ndim,block,5),g_a);
    return retval;

}


logical wnga_create_irreg_config(Integer type, Integer ndim, Integer *dims, char *name, Integer *map, Integer *block, Integer p_handle, Integer *g_a)
{
    logical retval;
    fprintf(fptrace, "%lf,pnga_create_irreg_config,(%ld;%ld;%s;%s;%s;%s;%ld;%p)\n",MPI_Wtime()-first_wtime,type,ndim,expand_intp(ndim,dims,0),name,expand_intp(sum_intp(ndim,block),map,1),expand_intp(ndim,block,2),p_handle,g_a);
    retval = pnga_create_irreg_config(type, ndim, dims, name, map, block, p_handle, g_a);
    fprintf(fptrace, "%lf,/pnga_create_irreg_config,(%ld;%ld;%s;%s;%s;%s;%ld;%p)\n",MPI_Wtime()-first_wtime,type,ndim,expand_intp(ndim,dims,3),name,expand_intp(sum_intp(ndim,block),map,4),expand_intp(ndim,block,5),p_handle,g_a);
    return retval;

}


logical wnga_create_mutexes(Integer num)
{
    logical retval;
    fprintf(fptrace, "%lf,pnga_create_mutexes,(%ld)\n",MPI_Wtime()-first_wtime,num);
    retval = pnga_create_mutexes(num);
    fprintf(fptrace, "%lf,/pnga_create_mutexes,(%ld)\n",MPI_Wtime()-first_wtime,num);
    return retval;

}


DoublePrecision wnga_ddot_patch_dp(Integer g_a, char *t_a, Integer ailo, Integer aihi, Integer ajlo, Integer ajhi, Integer g_b, char *t_b, Integer bilo, Integer bihi, Integer bjlo, Integer bjhi)
{
    DoublePrecision retval;
    fprintf(fptrace, "%lf,pnga_ddot_patch_dp,(%ld;%s;%ld;%ld;%ld;%ld;%ld;%s;%ld;%ld;%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,t_a,ailo,aihi,ajlo,ajhi,g_b,t_b,bilo,bihi,bjlo,bjhi);
    retval = pnga_ddot_patch_dp(g_a, t_a, ailo, aihi, ajlo, ajhi, g_b, t_b, bilo, bihi, bjlo, bjhi);
    fprintf(fptrace, "%lf,/pnga_ddot_patch_dp,(%ld;%s;%ld;%ld;%ld;%ld;%ld;%s;%ld;%ld;%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,t_a,ailo,aihi,ajlo,ajhi,g_b,t_b,bilo,bihi,bjlo,bjhi);
    return retval;

}


logical wnga_destroy(Integer g_a)
{
    logical retval;
    fprintf(fptrace, "%lf,pnga_destroy,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    retval = pnga_destroy(g_a);
    fprintf(fptrace, "%lf,/pnga_destroy,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    return retval;

}


logical wnga_destroy_mutexes()
{
    logical retval;
    fprintf(fptrace, "%lf,pnga_destroy_mutexes,\n",MPI_Wtime()-first_wtime);
    retval = pnga_destroy_mutexes();
    fprintf(fptrace, "%lf,/pnga_destroy_mutexes,\n",MPI_Wtime()-first_wtime);
    return retval;

}


void wnga_diag(Integer g_a, Integer g_s, Integer g_v, DoublePrecision *eval)
{
    fprintf(fptrace, "%lf,pnga_diag,(%ld;%ld;%ld;%p)\n",MPI_Wtime()-first_wtime,g_a,g_s,g_v,eval);
    pnga_diag(g_a, g_s, g_v, eval);
    fprintf(fptrace, "%lf,/pnga_diag,(%ld;%ld;%ld;%p)\n",MPI_Wtime()-first_wtime,g_a,g_s,g_v,eval);

}


void wnga_diag_reuse(Integer reuse, Integer g_a, Integer g_s, Integer g_v, DoublePrecision *eval)
{
    fprintf(fptrace, "%lf,pnga_diag_reuse,(%ld;%ld;%ld;%ld;%p)\n",MPI_Wtime()-first_wtime,reuse,g_a,g_s,g_v,eval);
    pnga_diag_reuse(reuse, g_a, g_s, g_v, eval);
    fprintf(fptrace, "%lf,/pnga_diag_reuse,(%ld;%ld;%ld;%ld;%p)\n",MPI_Wtime()-first_wtime,reuse,g_a,g_s,g_v,eval);

}


void wnga_diag_seq(Integer g_a, Integer g_s, Integer g_v, DoublePrecision *eval)
{
    fprintf(fptrace, "%lf,pnga_diag_seq,(%ld;%ld;%ld;%p)\n",MPI_Wtime()-first_wtime,g_a,g_s,g_v,eval);
    pnga_diag_seq(g_a, g_s, g_v, eval);
    fprintf(fptrace, "%lf,/pnga_diag_seq,(%ld;%ld;%ld;%p)\n",MPI_Wtime()-first_wtime,g_a,g_s,g_v,eval);

}


void wnga_diag_std(Integer g_a, Integer g_v, DoublePrecision *eval)
{
    fprintf(fptrace, "%lf,pnga_diag_std,(%ld;%ld;%p)\n",MPI_Wtime()-first_wtime,g_a,g_v,eval);
    pnga_diag_std(g_a, g_v, eval);
    fprintf(fptrace, "%lf,/pnga_diag_std,(%ld;%ld;%p)\n",MPI_Wtime()-first_wtime,g_a,g_v,eval);

}


void wnga_diag_std_seq(Integer g_a, Integer g_v, DoublePrecision *eval)
{
    fprintf(fptrace, "%lf,pnga_diag_std_seq,(%ld;%ld;%p)\n",MPI_Wtime()-first_wtime,g_a,g_v,eval);
    pnga_diag_std_seq(g_a, g_v, eval);
    fprintf(fptrace, "%lf,/pnga_diag_std_seq,(%ld;%ld;%p)\n",MPI_Wtime()-first_wtime,g_a,g_v,eval);

}


void wnga_distribution(Integer g_a, Integer proc, Integer *lo, Integer *hi)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_distribution,(%ld;%ld;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,proc,expand_intp(_ndim,lo,0),expand_intp(_ndim,hi,1));
    pnga_distribution(g_a, proc, lo, hi);
    fprintf(fptrace, "%lf,/pnga_distribution,(%ld;%ld;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,proc,expand_intp(_ndim,lo,2),expand_intp(_ndim,hi,3));

}


void wnga_dot(int type, Integer g_a, Integer g_b, void *value)
{
    fprintf(fptrace, "%lf,pnga_dot,(%d;%ld;%ld;%p)\n",MPI_Wtime()-first_wtime,type,g_a,g_b,value);
    pnga_dot(type, g_a, g_b, value);
    fprintf(fptrace, "%lf,/pnga_dot,(%d;%ld;%ld;%p)\n",MPI_Wtime()-first_wtime,type,g_a,g_b,value);

}


void wnga_dot_patch(Integer g_a, char *t_a, Integer *alo, Integer *ahi, Integer g_b, char *t_b, Integer *blo, Integer *bhi, void *retval)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_dot_patch,(%ld;%s;%s;%s;%ld;%s;%s;%s;%p)\n",MPI_Wtime()-first_wtime,g_a,t_a,expand_intp(_ndim,alo,0),expand_intp(_ndim,ahi,1),g_b,t_b,expand_intp(_ndim,blo,2),expand_intp(_ndim,bhi,3),retval);
    pnga_dot_patch(g_a, t_a, alo, ahi, g_b, t_b, blo, bhi, retval);
    fprintf(fptrace, "%lf,/pnga_dot_patch,(%ld;%s;%s;%s;%ld;%s;%s;%s;%p)\n",MPI_Wtime()-first_wtime,g_a,t_a,expand_intp(_ndim,alo,4),expand_intp(_ndim,ahi,5),g_b,t_b,expand_intp(_ndim,blo,6),expand_intp(_ndim,bhi,7),retval);

}


logical wnga_duplicate(Integer g_a, Integer *g_b, char *array_name)
{
    logical retval;
    fprintf(fptrace, "%lf,pnga_duplicate,(%ld;%p;%s)\n",MPI_Wtime()-first_wtime,g_a,g_b,array_name);
    retval = pnga_duplicate(g_a, g_b, array_name);
    fprintf(fptrace, "%lf,/pnga_duplicate,(%ld;%p;%s)\n",MPI_Wtime()-first_wtime,g_a,g_b,array_name);
    return retval;

}


void wnga_elem_divide(Integer g_a, Integer g_b, Integer g_c)
{
    fprintf(fptrace, "%lf,pnga_elem_divide,(%ld;%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,g_b,g_c);
    pnga_elem_divide(g_a, g_b, g_c);
    fprintf(fptrace, "%lf,/pnga_elem_divide,(%ld;%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,g_b,g_c);

}


void wnga_elem_divide_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_elem_divide_patch,(%ld;%s;%s;%ld;%s;%s;%ld;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,alo,0),expand_intp(_ndim,ahi,1),g_b,expand_intp(_ndim,blo,2),expand_intp(_ndim,bhi,3),g_c,expand_intp(_ndim,clo,4),expand_intp(_ndim,chi,5));
    pnga_elem_divide_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi);
    fprintf(fptrace, "%lf,/pnga_elem_divide_patch,(%ld;%s;%s;%ld;%s;%s;%ld;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,alo,6),expand_intp(_ndim,ahi,7),g_b,expand_intp(_ndim,blo,8),expand_intp(_ndim,bhi,9),g_c,expand_intp(_ndim,clo,10),expand_intp(_ndim,chi,11));

}


void wnga_elem_maximum(Integer g_a, Integer g_b, Integer g_c)
{
    fprintf(fptrace, "%lf,pnga_elem_maximum,(%ld;%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,g_b,g_c);
    pnga_elem_maximum(g_a, g_b, g_c);
    fprintf(fptrace, "%lf,/pnga_elem_maximum,(%ld;%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,g_b,g_c);

}


void wnga_elem_maximum_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_elem_maximum_patch,(%ld;%s;%s;%ld;%s;%s;%ld;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,alo,0),expand_intp(_ndim,ahi,1),g_b,expand_intp(_ndim,blo,2),expand_intp(_ndim,bhi,3),g_c,expand_intp(_ndim,clo,4),expand_intp(_ndim,chi,5));
    pnga_elem_maximum_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi);
    fprintf(fptrace, "%lf,/pnga_elem_maximum_patch,(%ld;%s;%s;%ld;%s;%s;%ld;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,alo,6),expand_intp(_ndim,ahi,7),g_b,expand_intp(_ndim,blo,8),expand_intp(_ndim,bhi,9),g_c,expand_intp(_ndim,clo,10),expand_intp(_ndim,chi,11));

}


void wnga_elem_minimum(Integer g_a, Integer g_b, Integer g_c)
{
    fprintf(fptrace, "%lf,pnga_elem_minimum,(%ld;%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,g_b,g_c);
    pnga_elem_minimum(g_a, g_b, g_c);
    fprintf(fptrace, "%lf,/pnga_elem_minimum,(%ld;%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,g_b,g_c);

}


void wnga_elem_minimum_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_elem_minimum_patch,(%ld;%s;%s;%ld;%s;%s;%ld;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,alo,0),expand_intp(_ndim,ahi,1),g_b,expand_intp(_ndim,blo,2),expand_intp(_ndim,bhi,3),g_c,expand_intp(_ndim,clo,4),expand_intp(_ndim,chi,5));
    pnga_elem_minimum_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi);
    fprintf(fptrace, "%lf,/pnga_elem_minimum_patch,(%ld;%s;%s;%ld;%s;%s;%ld;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,alo,6),expand_intp(_ndim,ahi,7),g_b,expand_intp(_ndim,blo,8),expand_intp(_ndim,bhi,9),g_c,expand_intp(_ndim,clo,10),expand_intp(_ndim,chi,11));

}


void wnga_elem_multiply(Integer g_a, Integer g_b, Integer g_c)
{
    fprintf(fptrace, "%lf,pnga_elem_multiply,(%ld;%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,g_b,g_c);
    pnga_elem_multiply(g_a, g_b, g_c);
    fprintf(fptrace, "%lf,/pnga_elem_multiply,(%ld;%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,g_b,g_c);

}


void wnga_elem_multiply_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_elem_multiply_patch,(%ld;%s;%s;%ld;%s;%s;%ld;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,alo,0),expand_intp(_ndim,ahi,1),g_b,expand_intp(_ndim,blo,2),expand_intp(_ndim,bhi,3),g_c,expand_intp(_ndim,clo,4),expand_intp(_ndim,chi,5));
    pnga_elem_multiply_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi);
    fprintf(fptrace, "%lf,/pnga_elem_multiply_patch,(%ld;%s;%s;%ld;%s;%s;%ld;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,alo,6),expand_intp(_ndim,ahi,7),g_b,expand_intp(_ndim,blo,8),expand_intp(_ndim,bhi,9),g_c,expand_intp(_ndim,clo,10),expand_intp(_ndim,chi,11));

}


void wnga_elem_step_divide_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_elem_step_divide_patch,(%ld;%s;%s;%ld;%s;%s;%ld;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,alo,0),expand_intp(_ndim,ahi,1),g_b,expand_intp(_ndim,blo,2),expand_intp(_ndim,bhi,3),g_c,expand_intp(_ndim,clo,4),expand_intp(_ndim,chi,5));
    pnga_elem_step_divide_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi);
    fprintf(fptrace, "%lf,/pnga_elem_step_divide_patch,(%ld;%s;%s;%ld;%s;%s;%ld;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,alo,6),expand_intp(_ndim,ahi,7),g_b,expand_intp(_ndim,blo,8),expand_intp(_ndim,bhi,9),g_c,expand_intp(_ndim,clo,10),expand_intp(_ndim,chi,11));

}


void wnga_elem_stepb_divide_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_elem_stepb_divide_patch,(%ld;%s;%s;%ld;%s;%s;%ld;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,alo,0),expand_intp(_ndim,ahi,1),g_b,expand_intp(_ndim,blo,2),expand_intp(_ndim,bhi,3),g_c,expand_intp(_ndim,clo,4),expand_intp(_ndim,chi,5));
    pnga_elem_stepb_divide_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi);
    fprintf(fptrace, "%lf,/pnga_elem_stepb_divide_patch,(%ld;%s;%s;%ld;%s;%s;%ld;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,alo,6),expand_intp(_ndim,ahi,7),g_b,expand_intp(_ndim,blo,8),expand_intp(_ndim,bhi,9),g_c,expand_intp(_ndim,clo,10),expand_intp(_ndim,chi,11));

}


void wnga_error(char *string, Integer icode)
{
    fprintf(fptrace, "%lf,pnga_error,(%s;%ld)\n",MPI_Wtime()-first_wtime,string,icode);
    pnga_error(string, icode);
    fprintf(fptrace, "%lf,/pnga_error,(%s;%ld)\n",MPI_Wtime()-first_wtime,string,icode);

}


void wnga_fence()
{
    fprintf(fptrace, "%lf,pnga_fence,\n",MPI_Wtime()-first_wtime);
    pnga_fence();
    fprintf(fptrace, "%lf,/pnga_fence,\n",MPI_Wtime()-first_wtime);

}


void wnga_fill(Integer g_a, void *val)
{
    fprintf(fptrace, "%lf,pnga_fill,(%ld;%p)\n",MPI_Wtime()-first_wtime,g_a,val);
    pnga_fill(g_a, val);
    fprintf(fptrace, "%lf,/pnga_fill,(%ld;%p)\n",MPI_Wtime()-first_wtime,g_a,val);

}


void wnga_fill_patch(Integer g_a, Integer *lo, Integer *hi, void *val)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_fill_patch,(%ld;%s;%s;%p)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,0),expand_intp(_ndim,hi,1),val);
    pnga_fill_patch(g_a, lo, hi, val);
    fprintf(fptrace, "%lf,/pnga_fill_patch,(%ld;%s;%s;%p)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,2),expand_intp(_ndim,hi,3),val);

}


void wnga_gather(Integer g_a, void *v, Integer subscript[], Integer nv)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_gather,(%ld;%p;%s;%ld)\n",MPI_Wtime()-first_wtime,g_a,v,expand_intp(_ndim,subscript,0),nv);
    pnga_gather(g_a, v, subscript, nv);
    fprintf(fptrace, "%lf,/pnga_gather,(%ld;%p;%s;%ld)\n",MPI_Wtime()-first_wtime,g_a,v,expand_intp(_ndim,subscript,1),nv);

}


void wnga_gather2d(Integer g_a, void *v, Integer *i, Integer *j, Integer nv)
{
    fprintf(fptrace, "%lf,pnga_gather2d,(%ld;%p;%p;%p;%ld)\n",MPI_Wtime()-first_wtime,g_a,v,i,j,nv);
    pnga_gather2d(g_a, v, i, j, nv);
    fprintf(fptrace, "%lf,/pnga_gather2d,(%ld;%p;%p;%p;%ld)\n",MPI_Wtime()-first_wtime,g_a,v,i,j,nv);

}


void wnga_get(Integer g_a, Integer *lo, Integer *hi, void *buf, Integer *ld)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_get,(%ld;%s;%s;%p;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,0),expand_intp(_ndim,hi,1),buf,expand_intp(_ndim-1,ld,2));
    pnga_get(g_a, lo, hi, buf, ld);
    fprintf(fptrace, "%lf,/pnga_get,(%ld;%s;%s;%p;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,3),expand_intp(_ndim,hi,4),buf,expand_intp(_ndim-1,ld,5));

}


void wnga_get_block_info(Integer g_a, Integer *num_blocks, Integer *block_dims)
{
    fprintf(fptrace, "%lf,pnga_get_block_info,(%ld;%p;%p)\n",MPI_Wtime()-first_wtime,g_a,num_blocks,block_dims);
    pnga_get_block_info(g_a, num_blocks, block_dims);
    fprintf(fptrace, "%lf,/pnga_get_block_info,(%ld;%p;%p)\n",MPI_Wtime()-first_wtime,g_a,num_blocks,block_dims);

}


logical wnga_get_debug()
{
    logical retval;
    fprintf(fptrace, "%lf,pnga_get_debug,\n",MPI_Wtime()-first_wtime);
    retval = pnga_get_debug();
    fprintf(fptrace, "%lf,/pnga_get_debug,\n",MPI_Wtime()-first_wtime);
    return retval;

}


void wnga_get_diag(Integer g_a, Integer g_v)
{
    fprintf(fptrace, "%lf,pnga_get_diag,(%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,g_v);
    pnga_get_diag(g_a, g_v);
    fprintf(fptrace, "%lf,/pnga_get_diag,(%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,g_v);

}


Integer wnga_get_dimension(Integer g_a)
{
    Integer retval;
    fprintf(fptrace, "%lf,pnga_get_dimension,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    retval = pnga_get_dimension(g_a);
    fprintf(fptrace, "%lf,/pnga_get_dimension,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    return retval;

}


void wnga_get_ghost_block(Integer g_a, Integer *lo, Integer *hi, void *buf, Integer *ld)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_get_ghost_block,(%ld;%s;%s;%p;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,0),expand_intp(_ndim,hi,1),buf,expand_intp(_ndim-1,ld,2));
    pnga_get_ghost_block(g_a, lo, hi, buf, ld);
    fprintf(fptrace, "%lf,/pnga_get_ghost_block,(%ld;%s;%s;%p;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,3),expand_intp(_ndim,hi,4),buf,expand_intp(_ndim-1,ld,5));

}


Integer wnga_get_pgroup(Integer g_a)
{
    Integer retval;
    fprintf(fptrace, "%lf,pnga_get_pgroup,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    retval = pnga_get_pgroup(g_a);
    fprintf(fptrace, "%lf,/pnga_get_pgroup,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    return retval;

}


Integer wnga_get_pgroup_size(Integer grp_id)
{
    Integer retval;
    fprintf(fptrace, "%lf,pnga_get_pgroup_size,(%ld)\n",MPI_Wtime()-first_wtime,grp_id);
    retval = pnga_get_pgroup_size(grp_id);
    fprintf(fptrace, "%lf,/pnga_get_pgroup_size,(%ld)\n",MPI_Wtime()-first_wtime,grp_id);
    return retval;

}


void wnga_get_proc_grid(Integer g_a, Integer *dims)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_get_proc_grid,(%ld;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,dims,0));
    pnga_get_proc_grid(g_a, dims);
    fprintf(fptrace, "%lf,/pnga_get_proc_grid,(%ld;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,dims,1));

}


void wnga_get_proc_index(Integer g_a, Integer iproc, Integer *index)
{
    fprintf(fptrace, "%lf,pnga_get_proc_index,(%ld;%ld;%p)\n",MPI_Wtime()-first_wtime,g_a,iproc,index);
    pnga_get_proc_index(g_a, iproc, index);
    fprintf(fptrace, "%lf,/pnga_get_proc_index,(%ld;%ld;%p)\n",MPI_Wtime()-first_wtime,g_a,iproc,index);

}


void wnga_ghost_barrier()
{
    fprintf(fptrace, "%lf,pnga_ghost_barrier,\n",MPI_Wtime()-first_wtime);
    pnga_ghost_barrier();
    fprintf(fptrace, "%lf,/pnga_ghost_barrier,\n",MPI_Wtime()-first_wtime);

}


void wnga_gop(Integer type, void *x, Integer n, char *op)
{
    fprintf(fptrace, "%lf,pnga_gop,(%ld;%p;%ld;%s)\n",MPI_Wtime()-first_wtime,type,x,n,op);
    pnga_gop(type, x, n, op);
    fprintf(fptrace, "%lf,/pnga_gop,(%ld;%p;%ld;%s)\n",MPI_Wtime()-first_wtime,type,x,n,op);

}


logical wnga_has_ghosts(Integer g_a)
{
    logical retval;
    fprintf(fptrace, "%lf,pnga_has_ghosts,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    retval = pnga_has_ghosts(g_a);
    fprintf(fptrace, "%lf,/pnga_has_ghosts,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    return retval;

}


void wnga_init_fence()
{
    fprintf(fptrace, "%lf,pnga_init_fence,\n",MPI_Wtime()-first_wtime);
    pnga_init_fence();
    fprintf(fptrace, "%lf,/pnga_init_fence,\n",MPI_Wtime()-first_wtime);

}


void wnga_inquire(Integer g_a, Integer *type, Integer *ndim, Integer *dims)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_inquire,(%ld;%p;%p;%s)\n",MPI_Wtime()-first_wtime,g_a,type,ndim,expand_intp(_ndim,dims,0));
    pnga_inquire(g_a, type, ndim, dims);
    fprintf(fptrace, "%lf,/pnga_inquire,(%ld;%p;%p;%s)\n",MPI_Wtime()-first_wtime,g_a,type,ndim,expand_intp(_ndim,dims,1));

}


Integer wnga_inquire_memory()
{
    Integer retval;
    fprintf(fptrace, "%lf,pnga_inquire_memory,\n",MPI_Wtime()-first_wtime);
    retval = pnga_inquire_memory();
    fprintf(fptrace, "%lf,/pnga_inquire_memory,\n",MPI_Wtime()-first_wtime);
    return retval;

}


void wnga_inquire_name(Integer g_a, char **array_name)
{
    fprintf(fptrace, "%lf,pnga_inquire_name,(%ld;%p)\n",MPI_Wtime()-first_wtime,g_a,array_name);
    pnga_inquire_name(g_a, array_name);
    fprintf(fptrace, "%lf,/pnga_inquire_name,(%ld;%p)\n",MPI_Wtime()-first_wtime,g_a,array_name);

}


void wnga_inquire_type(Integer g_a, Integer *type)
{
    fprintf(fptrace, "%lf,pnga_inquire_type,(%ld;%p)\n",MPI_Wtime()-first_wtime,g_a,type);
    pnga_inquire_type(g_a, type);
    fprintf(fptrace, "%lf,/pnga_inquire_type,(%ld;%p)\n",MPI_Wtime()-first_wtime,g_a,type);

}


logical wnga_is_mirrored(Integer g_a)
{
    logical retval;
    fprintf(fptrace, "%lf,pnga_is_mirrored,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    retval = pnga_is_mirrored(g_a);
    fprintf(fptrace, "%lf,/pnga_is_mirrored,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    return retval;

}


void wnga_list_nodeid(Integer *list, Integer nprocs)
{
    fprintf(fptrace, "%lf,pnga_list_nodeid,(%p;%ld)\n",MPI_Wtime()-first_wtime,list,nprocs);
    pnga_list_nodeid(list, nprocs);
    fprintf(fptrace, "%lf,/pnga_list_nodeid,(%p;%ld)\n",MPI_Wtime()-first_wtime,list,nprocs);

}


Integer wnga_llt_solve(Integer g_a, Integer g_b)
{
    Integer retval;
    fprintf(fptrace, "%lf,pnga_llt_solve,(%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,g_b);
    retval = pnga_llt_solve(g_a, g_b);
    fprintf(fptrace, "%lf,/pnga_llt_solve,(%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,g_b);
    return retval;

}


logical wnga_locate(Integer g_a, Integer *subscript, Integer *owner)
{
    int _ndim = pnga_ndim(g_a);
    logical retval;
    fprintf(fptrace, "%lf,pnga_locate,(%ld;%s;%p)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,subscript,0),owner);
    retval = pnga_locate(g_a, subscript, owner);
    fprintf(fptrace, "%lf,/pnga_locate,(%ld;%s;%p)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,subscript,1),owner);
    return retval;

}


logical wnga_locate_nnodes(Integer g_a, Integer *lo, Integer *hi, Integer *np)
{
    int _ndim = pnga_ndim(g_a);
    logical retval;
    fprintf(fptrace, "%lf,pnga_locate_nnodes,(%ld;%s;%s;%p)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,0),expand_intp(_ndim,hi,1),np);
    retval = pnga_locate_nnodes(g_a, lo, hi, np);
    fprintf(fptrace, "%lf,/pnga_locate_nnodes,(%ld;%s;%s;%p)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,2),expand_intp(_ndim,hi,3),np);
    return retval;

}


Integer wnga_locate_num_blocks(Integer g_a, Integer *lo, Integer *hi)
{
    int _ndim = pnga_ndim(g_a);
    Integer retval;
    fprintf(fptrace, "%lf,pnga_locate_num_blocks,(%ld;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,0),expand_intp(_ndim,hi,1));
    retval = pnga_locate_num_blocks(g_a, lo, hi);
    fprintf(fptrace, "%lf,/pnga_locate_num_blocks,(%ld;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,2),expand_intp(_ndim,hi,3));
    return retval;

}


logical wnga_locate_region(Integer g_a, Integer *lo, Integer *hi, Integer *map, Integer *proclist, Integer *np)
{
    int _ndim = pnga_ndim(g_a);
    logical retval;
    fprintf(fptrace, "%lf,pnga_locate_region,(%ld;%s;%s;%s;%p;%p)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,0),expand_intp(_ndim,hi,1),expand_intp(_ndim,map,2),proclist,np);
    retval = pnga_locate_region(g_a, lo, hi, map, proclist, np);
    fprintf(fptrace, "%lf,/pnga_locate_region,(%ld;%s;%s;%s;%p;%p)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,3),expand_intp(_ndim,hi,4),expand_intp(_ndim,map,5),proclist,np);
    return retval;

}


void wnga_lock(Integer mutex)
{
    fprintf(fptrace, "%lf,pnga_lock,(%ld)\n",MPI_Wtime()-first_wtime,mutex);
    pnga_lock(mutex);
    fprintf(fptrace, "%lf,/pnga_lock,(%ld)\n",MPI_Wtime()-first_wtime,mutex);

}


void wnga_lu_solve(char *tran, Integer g_a, Integer g_b)
{
    fprintf(fptrace, "%lf,pnga_lu_solve,(%s;%ld;%ld)\n",MPI_Wtime()-first_wtime,tran,g_a,g_b);
    pnga_lu_solve(tran, g_a, g_b);
    fprintf(fptrace, "%lf,/pnga_lu_solve,(%s;%ld;%ld)\n",MPI_Wtime()-first_wtime,tran,g_a,g_b);

}


void wnga_lu_solve_alt(Integer tran, Integer g_a, Integer g_b)
{
    fprintf(fptrace, "%lf,pnga_lu_solve_alt,(%ld;%ld;%ld)\n",MPI_Wtime()-first_wtime,tran,g_a,g_b);
    pnga_lu_solve_alt(tran, g_a, g_b);
    fprintf(fptrace, "%lf,/pnga_lu_solve_alt,(%ld;%ld;%ld)\n",MPI_Wtime()-first_wtime,tran,g_a,g_b);

}


void wnga_lu_solve_seq(char *trans, Integer g_a, Integer g_b)
{
    fprintf(fptrace, "%lf,pnga_lu_solve_seq,(%s;%ld;%ld)\n",MPI_Wtime()-first_wtime,trans,g_a,g_b);
    pnga_lu_solve_seq(trans, g_a, g_b);
    fprintf(fptrace, "%lf,/pnga_lu_solve_seq,(%s;%ld;%ld)\n",MPI_Wtime()-first_wtime,trans,g_a,g_b);

}


void wnga_mask_sync(Integer begin, Integer end)
{
    fprintf(fptrace, "%lf,pnga_mask_sync,(%ld;%ld)\n",MPI_Wtime()-first_wtime,begin,end);
    pnga_mask_sync(begin, end);
    fprintf(fptrace, "%lf,/pnga_mask_sync,(%ld;%ld)\n",MPI_Wtime()-first_wtime,begin,end);

}


void wnga_matmul(char *transa, char *transb, void *alpha, void *beta, Integer g_a, Integer ailo, Integer aihi, Integer ajlo, Integer ajhi, Integer g_b, Integer bilo, Integer bihi, Integer bjlo, Integer bjhi, Integer g_c, Integer cilo, Integer cihi, Integer cjlo, Integer cjhi)
{
    Integer _atype;
    Integer _btype;
    pnga_inquire_type(g_a, &_atype);
    pnga_inquire_type(g_b, &_btype);
    fprintf(fptrace, "%lf,pnga_matmul,(%s;%s;%s;%s;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld)\n",MPI_Wtime()-first_wtime,transa,transb,expand_voidp(_atype,alpha,0),expand_voidp(_btype,beta,1),g_a,ailo,aihi,ajlo,ajhi,g_b,bilo,bihi,bjlo,bjhi,g_c,cilo,cihi,cjlo,cjhi);
    pnga_matmul(transa, transb, alpha, beta, g_a, ailo, aihi, ajlo, ajhi, g_b, bilo, bihi, bjlo, bjhi, g_c, cilo, cihi, cjlo, cjhi);
    fprintf(fptrace, "%lf,/pnga_matmul,(%s;%s;%s;%s;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld)\n",MPI_Wtime()-first_wtime,transa,transb,expand_voidp(_atype,alpha,2),expand_voidp(_btype,beta,3),g_a,ailo,aihi,ajlo,ajhi,g_b,bilo,bihi,bjlo,bjhi,g_c,cilo,cihi,cjlo,cjhi);

}


void wnga_matmul_mirrored(char *transa, char *transb, void *alpha, void *beta, Integer g_a, Integer ailo, Integer aihi, Integer ajlo, Integer ajhi, Integer g_b, Integer bilo, Integer bihi, Integer bjlo, Integer bjhi, Integer g_c, Integer cilo, Integer cihi, Integer cjlo, Integer cjhi)
{
    Integer _atype;
    Integer _btype;
    pnga_inquire_type(g_a, &_atype);
    pnga_inquire_type(g_b, &_btype);
    fprintf(fptrace, "%lf,pnga_matmul_mirrored,(%s;%s;%s;%s;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld)\n",MPI_Wtime()-first_wtime,transa,transb,expand_voidp(_atype,alpha,0),expand_voidp(_btype,beta,1),g_a,ailo,aihi,ajlo,ajhi,g_b,bilo,bihi,bjlo,bjhi,g_c,cilo,cihi,cjlo,cjhi);
    pnga_matmul_mirrored(transa, transb, alpha, beta, g_a, ailo, aihi, ajlo, ajhi, g_b, bilo, bihi, bjlo, bjhi, g_c, cilo, cihi, cjlo, cjhi);
    fprintf(fptrace, "%lf,/pnga_matmul_mirrored,(%s;%s;%s;%s;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld)\n",MPI_Wtime()-first_wtime,transa,transb,expand_voidp(_atype,alpha,2),expand_voidp(_btype,beta,3),g_a,ailo,aihi,ajlo,ajhi,g_b,bilo,bihi,bjlo,bjhi,g_c,cilo,cihi,cjlo,cjhi);

}


void wnga_matmul_patch(char *transa, char *transb, void *alpha, void *beta, Integer g_a, Integer alo[], Integer ahi[], Integer g_b, Integer blo[], Integer bhi[], Integer g_c, Integer clo[], Integer chi[])
{
    int _ndim = pnga_ndim(g_a);
    Integer _atype;
    Integer _btype;
    pnga_inquire_type(g_a, &_atype);
    pnga_inquire_type(g_b, &_btype);
    fprintf(fptrace, "%lf,pnga_matmul_patch,(%s;%s;%s;%s;%ld;%s;%s;%ld;%s;%s;%ld;%s;%s)\n",MPI_Wtime()-first_wtime,transa,transb,expand_voidp(_atype,alpha,0),expand_voidp(_btype,beta,1),g_a,expand_intp(_ndim,alo,2),expand_intp(_ndim,ahi,3),g_b,expand_intp(_ndim,blo,4),expand_intp(_ndim,bhi,5),g_c,expand_intp(_ndim,clo,6),expand_intp(_ndim,chi,7));
    pnga_matmul_patch(transa, transb, alpha, beta, g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi);
    fprintf(fptrace, "%lf,/pnga_matmul_patch,(%s;%s;%s;%s;%ld;%s;%s;%ld;%s;%s;%ld;%s;%s)\n",MPI_Wtime()-first_wtime,transa,transb,expand_voidp(_atype,alpha,8),expand_voidp(_btype,beta,9),g_a,expand_intp(_ndim,alo,10),expand_intp(_ndim,ahi,11),g_b,expand_intp(_ndim,blo,12),expand_intp(_ndim,bhi,13),g_c,expand_intp(_ndim,clo,14),expand_intp(_ndim,chi,15));

}


void wnga_median(Integer g_a, Integer g_b, Integer g_c, Integer g_m)
{
    fprintf(fptrace, "%lf,pnga_median,(%ld;%ld;%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,g_b,g_c,g_m);
    pnga_median(g_a, g_b, g_c, g_m);
    fprintf(fptrace, "%lf,/pnga_median,(%ld;%ld;%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,g_b,g_c,g_m);

}


void wnga_median_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi, Integer g_m, Integer *mlo, Integer *mhi)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_median_patch,(%ld;%s;%s;%ld;%s;%s;%ld;%s;%s;%ld;%p;%p)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,alo,0),expand_intp(_ndim,ahi,1),g_b,expand_intp(_ndim,blo,2),expand_intp(_ndim,bhi,3),g_c,expand_intp(_ndim,clo,4),expand_intp(_ndim,chi,5),g_m,mlo,mhi);
    pnga_median_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi, g_m, mlo, mhi);
    fprintf(fptrace, "%lf,/pnga_median_patch,(%ld;%s;%s;%ld;%s;%s;%ld;%s;%s;%ld;%p;%p)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,alo,6),expand_intp(_ndim,ahi,7),g_b,expand_intp(_ndim,blo,8),expand_intp(_ndim,bhi,9),g_c,expand_intp(_ndim,clo,10),expand_intp(_ndim,chi,11),g_m,mlo,mhi);

}


Integer wnga_memory_avail()
{
    Integer retval;
    fprintf(fptrace, "%lf,pnga_memory_avail,\n",MPI_Wtime()-first_wtime);
    retval = pnga_memory_avail();
    fprintf(fptrace, "%lf,/pnga_memory_avail,\n",MPI_Wtime()-first_wtime);
    return retval;

}


Integer wnga_memory_avail_type(Integer datatype)
{
    Integer retval;
    fprintf(fptrace, "%lf,pnga_memory_avail_type,(%ld)\n",MPI_Wtime()-first_wtime,datatype);
    retval = pnga_memory_avail_type(datatype);
    fprintf(fptrace, "%lf,/pnga_memory_avail_type,(%ld)\n",MPI_Wtime()-first_wtime,datatype);
    return retval;

}


logical wnga_memory_limited()
{
    logical retval;
    fprintf(fptrace, "%lf,pnga_memory_limited,\n",MPI_Wtime()-first_wtime);
    retval = pnga_memory_limited();
    fprintf(fptrace, "%lf,/pnga_memory_limited,\n",MPI_Wtime()-first_wtime);
    return retval;

}


void wnga_merge_distr_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_merge_distr_patch,(%ld;%s;%s;%ld;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,alo,0),expand_intp(_ndim,ahi,1),g_b,expand_intp(_ndim,blo,2),expand_intp(_ndim,bhi,3));
    pnga_merge_distr_patch(g_a, alo, ahi, g_b, blo, bhi);
    fprintf(fptrace, "%lf,/pnga_merge_distr_patch,(%ld;%s;%s;%ld;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,alo,4),expand_intp(_ndim,ahi,5),g_b,expand_intp(_ndim,blo,6),expand_intp(_ndim,bhi,7));

}


void wnga_merge_mirrored(Integer g_a)
{
    fprintf(fptrace, "%lf,pnga_merge_mirrored,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    pnga_merge_mirrored(g_a);
    fprintf(fptrace, "%lf,/pnga_merge_mirrored,(%ld)\n",MPI_Wtime()-first_wtime,g_a);

}


void wnga_msg_brdcst(Integer type, void *buffer, Integer len, Integer root)
{
    fprintf(fptrace, "%lf,pnga_msg_brdcst,(%ld;%p;%ld;%ld)\n",MPI_Wtime()-first_wtime,type,buffer,len,root);
    pnga_msg_brdcst(type, buffer, len, root);
    fprintf(fptrace, "%lf,/pnga_msg_brdcst,(%ld;%p;%ld;%ld)\n",MPI_Wtime()-first_wtime,type,buffer,len,root);

}


void wnga_msg_pgroup_sync(Integer grp_id)
{
    fprintf(fptrace, "%lf,pnga_msg_pgroup_sync,(%ld)\n",MPI_Wtime()-first_wtime,grp_id);
    pnga_msg_pgroup_sync(grp_id);
    fprintf(fptrace, "%lf,/pnga_msg_pgroup_sync,(%ld)\n",MPI_Wtime()-first_wtime,grp_id);

}


void wnga_msg_sync()
{
    fprintf(fptrace, "%lf,pnga_msg_sync,\n",MPI_Wtime()-first_wtime);
    pnga_msg_sync();
    fprintf(fptrace, "%lf,/pnga_msg_sync,\n",MPI_Wtime()-first_wtime);

}


void wnga_nbacc(Integer g_a, Integer *lo, Integer *hi, void *buf, Integer *ld, void *alpha, Integer *nbhndl)
{
    int _ndim = pnga_ndim(g_a);
    Integer _atype;
    pnga_inquire_type(g_a, &_atype);
    fprintf(fptrace, "%lf,pnga_nbacc,(%ld;%s;%s;%p;%s;%s;%p)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,0),expand_intp(_ndim,hi,1),buf,expand_intp(_ndim-1,ld,2),expand_voidp(_atype,alpha,3),nbhndl);
    pnga_nbacc(g_a, lo, hi, buf, ld, alpha, nbhndl);
    fprintf(fptrace, "%lf,/pnga_nbacc,(%ld;%s;%s;%p;%s;%s;%p)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,4),expand_intp(_ndim,hi,5),buf,expand_intp(_ndim-1,ld,6),expand_voidp(_atype,alpha,7),nbhndl);

}


void wnga_nbget(Integer g_a, Integer *lo, Integer *hi, void *buf, Integer *ld, Integer *nbhandle)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_nbget,(%ld;%s;%s;%p;%s;%p)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,0),expand_intp(_ndim,hi,1),buf,expand_intp(_ndim-1,ld,2),nbhandle);
    pnga_nbget(g_a, lo, hi, buf, ld, nbhandle);
    fprintf(fptrace, "%lf,/pnga_nbget,(%ld;%s;%s;%p;%s;%p)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,3),expand_intp(_ndim,hi,4),buf,expand_intp(_ndim-1,ld,5),nbhandle);

}


void wnga_nbget_ghost_dir(Integer g_a, Integer *mask, Integer *nbhandle)
{
    fprintf(fptrace, "%lf,pnga_nbget_ghost_dir,(%ld;%p;%p)\n",MPI_Wtime()-first_wtime,g_a,mask,nbhandle);
    pnga_nbget_ghost_dir(g_a, mask, nbhandle);
    fprintf(fptrace, "%lf,/pnga_nbget_ghost_dir,(%ld;%p;%p)\n",MPI_Wtime()-first_wtime,g_a,mask,nbhandle);

}


void wnga_nblock(Integer g_a, Integer *nblock)
{
    fprintf(fptrace, "%lf,pnga_nblock,(%ld;%p)\n",MPI_Wtime()-first_wtime,g_a,nblock);
    pnga_nblock(g_a, nblock);
    fprintf(fptrace, "%lf,/pnga_nblock,(%ld;%p)\n",MPI_Wtime()-first_wtime,g_a,nblock);

}


void wnga_nbput(Integer g_a, Integer *lo, Integer *hi, void *buf, Integer *ld, Integer *nbhandle)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_nbput,(%ld;%s;%s;%p;%s;%p)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,0),expand_intp(_ndim,hi,1),buf,expand_intp(_ndim-1,ld,2),nbhandle);
    pnga_nbput(g_a, lo, hi, buf, ld, nbhandle);
    fprintf(fptrace, "%lf,/pnga_nbput,(%ld;%s;%s;%p;%s;%p)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,3),expand_intp(_ndim,hi,4),buf,expand_intp(_ndim-1,ld,5),nbhandle);

}


Integer wnga_nbtest(Integer *nbhandle)
{
    Integer retval;
    fprintf(fptrace, "%lf,pnga_nbtest,(%p)\n",MPI_Wtime()-first_wtime,nbhandle);
    retval = pnga_nbtest(nbhandle);
    fprintf(fptrace, "%lf,/pnga_nbtest,(%p)\n",MPI_Wtime()-first_wtime,nbhandle);
    return retval;

}


void wnga_nbwait(Integer *nbhandle)
{
    fprintf(fptrace, "%lf,pnga_nbwait,(%p)\n",MPI_Wtime()-first_wtime,nbhandle);
    pnga_nbwait(nbhandle);
    fprintf(fptrace, "%lf,/pnga_nbwait,(%p)\n",MPI_Wtime()-first_wtime,nbhandle);

}


Integer wnga_ndim(Integer g_a)
{
    Integer retval;
    fprintf(fptrace, "%lf,pnga_ndim,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    retval = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,/pnga_ndim,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    return retval;

}


Integer wnga_nnodes()
{
    Integer retval;
    fprintf(fptrace, "%lf,pnga_nnodes,\n",MPI_Wtime()-first_wtime);
    retval = pnga_nnodes();
    fprintf(fptrace, "%lf,/pnga_nnodes,\n",MPI_Wtime()-first_wtime);
    return retval;

}


Integer wnga_nodeid()
{
    Integer retval;
    fprintf(fptrace, "%lf,pnga_nodeid,\n",MPI_Wtime()-first_wtime);
    retval = pnga_nodeid();
    fprintf(fptrace, "%lf,/pnga_nodeid,\n",MPI_Wtime()-first_wtime);
    return retval;

}


void wnga_norm1(Integer g_a, double *nm)
{
    fprintf(fptrace, "%lf,pnga_norm1,(%ld;%p)\n",MPI_Wtime()-first_wtime,g_a,nm);
    pnga_norm1(g_a, nm);
    fprintf(fptrace, "%lf,/pnga_norm1,(%ld;%p)\n",MPI_Wtime()-first_wtime,g_a,nm);

}


void wnga_norm_infinity(Integer g_a, double *nm)
{
    fprintf(fptrace, "%lf,pnga_norm_infinity,(%ld;%p)\n",MPI_Wtime()-first_wtime,g_a,nm);
    pnga_norm_infinity(g_a, nm);
    fprintf(fptrace, "%lf,/pnga_norm_infinity,(%ld;%p)\n",MPI_Wtime()-first_wtime,g_a,nm);

}


void wnga_pack(Integer g_a, Integer g_b, Integer g_sbit, Integer lo, Integer hi, Integer *icount)
{
    fprintf(fptrace, "%lf,pnga_pack,(%ld;%ld;%ld;%ld;%ld;%p)\n",MPI_Wtime()-first_wtime,g_a,g_b,g_sbit,lo,hi,icount);
    pnga_pack(g_a, g_b, g_sbit, lo, hi, icount);
    fprintf(fptrace, "%lf,/pnga_pack,(%ld;%ld;%ld;%ld;%ld;%p)\n",MPI_Wtime()-first_wtime,g_a,g_b,g_sbit,lo,hi,icount);

}


void wnga_patch_enum(Integer g_a, Integer lo, Integer hi, void *start, void *stride)
{
    fprintf(fptrace, "%lf,pnga_patch_enum,(%ld;%ld;%ld;%p;%p)\n",MPI_Wtime()-first_wtime,g_a,lo,hi,start,stride);
    pnga_patch_enum(g_a, lo, hi, start, stride);
    fprintf(fptrace, "%lf,/pnga_patch_enum,(%ld;%ld;%ld;%p;%p)\n",MPI_Wtime()-first_wtime,g_a,lo,hi,start,stride);

}


logical wnga_patch_intersect(Integer *lo, Integer *hi, Integer *lop, Integer *hip, Integer ndim)
{
    logical retval;
    fprintf(fptrace, "%lf,pnga_patch_intersect,(%s;%s;%p;%p;%ld)\n",MPI_Wtime()-first_wtime,expand_intp(ndim,lo,0),expand_intp(ndim,hi,1),lop,hip,ndim);
    retval = pnga_patch_intersect(lo, hi, lop, hip, ndim);
    fprintf(fptrace, "%lf,/pnga_patch_intersect,(%s;%s;%p;%p;%ld)\n",MPI_Wtime()-first_wtime,expand_intp(ndim,lo,2),expand_intp(ndim,hi,3),lop,hip,ndim);
    return retval;

}


void wnga_periodic(Integer g_a, Integer *lo, Integer *hi, void *buf, Integer *ld, void *alpha, Integer op_code)
{
    int _ndim = pnga_ndim(g_a);
    Integer _atype;
    pnga_inquire_type(g_a, &_atype);
    fprintf(fptrace, "%lf,pnga_periodic,(%ld;%s;%s;%p;%s;%s;%ld)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,0),expand_intp(_ndim,hi,1),buf,expand_intp(_ndim-1,ld,2),expand_voidp(_atype,alpha,3),op_code);
    pnga_periodic(g_a, lo, hi, buf, ld, alpha, op_code);
    fprintf(fptrace, "%lf,/pnga_periodic,(%ld;%s;%s;%p;%s;%s;%ld)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,4),expand_intp(_ndim,hi,5),buf,expand_intp(_ndim-1,ld,6),expand_voidp(_atype,alpha,7),op_code);

}


Integer wnga_pgroup_absolute_id(Integer grp, Integer pid)
{
    Integer retval;
    fprintf(fptrace, "%lf,pnga_pgroup_absolute_id,(%ld;%ld)\n",MPI_Wtime()-first_wtime,grp,pid);
    retval = pnga_pgroup_absolute_id(grp, pid);
    fprintf(fptrace, "%lf,/pnga_pgroup_absolute_id,(%ld;%ld)\n",MPI_Wtime()-first_wtime,grp,pid);
    return retval;

}


void wnga_pgroup_brdcst(Integer grp_id, Integer type, void *buf, Integer len, Integer originator)
{
    fprintf(fptrace, "%lf,pnga_pgroup_brdcst,(%ld;%ld;%p;%ld;%ld)\n",MPI_Wtime()-first_wtime,grp_id,type,buf,len,originator);
    pnga_pgroup_brdcst(grp_id, type, buf, len, originator);
    fprintf(fptrace, "%lf,/pnga_pgroup_brdcst,(%ld;%ld;%p;%ld;%ld)\n",MPI_Wtime()-first_wtime,grp_id,type,buf,len,originator);

}


Integer wnga_pgroup_create(Integer *list, Integer count)
{
    Integer retval;
    fprintf(fptrace, "%lf,pnga_pgroup_create,(%p;%ld)\n",MPI_Wtime()-first_wtime,list,count);
    retval = pnga_pgroup_create(list, count);
    fprintf(fptrace, "%lf,/pnga_pgroup_create,(%p;%ld)\n",MPI_Wtime()-first_wtime,list,count);
    return retval;

}


logical wnga_pgroup_destroy(Integer grp)
{
    logical retval;
    fprintf(fptrace, "%lf,pnga_pgroup_destroy,(%ld)\n",MPI_Wtime()-first_wtime,grp);
    retval = pnga_pgroup_destroy(grp);
    fprintf(fptrace, "%lf,/pnga_pgroup_destroy,(%ld)\n",MPI_Wtime()-first_wtime,grp);
    return retval;

}


Integer wnga_pgroup_get_default()
{
    Integer retval;
    fprintf(fptrace, "%lf,pnga_pgroup_get_default,\n",MPI_Wtime()-first_wtime);
    retval = pnga_pgroup_get_default();
    fprintf(fptrace, "%lf,/pnga_pgroup_get_default,\n",MPI_Wtime()-first_wtime);
    return retval;

}


Integer wnga_pgroup_get_mirror()
{
    Integer retval;
    fprintf(fptrace, "%lf,pnga_pgroup_get_mirror,\n",MPI_Wtime()-first_wtime);
    retval = pnga_pgroup_get_mirror();
    fprintf(fptrace, "%lf,/pnga_pgroup_get_mirror,\n",MPI_Wtime()-first_wtime);
    return retval;

}


Integer wnga_pgroup_get_world()
{
    Integer retval;
    fprintf(fptrace, "%lf,pnga_pgroup_get_world,\n",MPI_Wtime()-first_wtime);
    retval = pnga_pgroup_get_world();
    fprintf(fptrace, "%lf,/pnga_pgroup_get_world,\n",MPI_Wtime()-first_wtime);
    return retval;

}


void wnga_pgroup_gop(Integer p_grp, Integer type, void *x, Integer n, char *op)
{
    fprintf(fptrace, "%lf,pnga_pgroup_gop,(%ld;%ld;%p;%ld;%s)\n",MPI_Wtime()-first_wtime,p_grp,type,x,n,op);
    pnga_pgroup_gop(p_grp, type, x, n, op);
    fprintf(fptrace, "%lf,/pnga_pgroup_gop,(%ld;%ld;%p;%ld;%s)\n",MPI_Wtime()-first_wtime,p_grp,type,x,n,op);

}


Integer wnga_pgroup_nnodes(Integer grp)
{
    Integer retval;
    fprintf(fptrace, "%lf,pnga_pgroup_nnodes,(%ld)\n",MPI_Wtime()-first_wtime,grp);
    retval = pnga_pgroup_nnodes(grp);
    fprintf(fptrace, "%lf,/pnga_pgroup_nnodes,(%ld)\n",MPI_Wtime()-first_wtime,grp);
    return retval;

}


Integer wnga_pgroup_nodeid(Integer grp)
{
    Integer retval;
    fprintf(fptrace, "%lf,pnga_pgroup_nodeid,(%ld)\n",MPI_Wtime()-first_wtime,grp);
    retval = pnga_pgroup_nodeid(grp);
    fprintf(fptrace, "%lf,/pnga_pgroup_nodeid,(%ld)\n",MPI_Wtime()-first_wtime,grp);
    return retval;

}


void wnga_pgroup_set_default(Integer grp)
{
    fprintf(fptrace, "%lf,pnga_pgroup_set_default,(%ld)\n",MPI_Wtime()-first_wtime,grp);
    pnga_pgroup_set_default(grp);
    fprintf(fptrace, "%lf,/pnga_pgroup_set_default,(%ld)\n",MPI_Wtime()-first_wtime,grp);

}


Integer wnga_pgroup_split(Integer grp, Integer grp_num)
{
    Integer retval;
    fprintf(fptrace, "%lf,pnga_pgroup_split,(%ld;%ld)\n",MPI_Wtime()-first_wtime,grp,grp_num);
    retval = pnga_pgroup_split(grp, grp_num);
    fprintf(fptrace, "%lf,/pnga_pgroup_split,(%ld;%ld)\n",MPI_Wtime()-first_wtime,grp,grp_num);
    return retval;

}


Integer wnga_pgroup_split_irreg(Integer grp, Integer mycolor)
{
    Integer retval;
    fprintf(fptrace, "%lf,pnga_pgroup_split_irreg,(%ld;%ld)\n",MPI_Wtime()-first_wtime,grp,mycolor);
    retval = pnga_pgroup_split_irreg(grp, mycolor);
    fprintf(fptrace, "%lf,/pnga_pgroup_split_irreg,(%ld;%ld)\n",MPI_Wtime()-first_wtime,grp,mycolor);
    return retval;

}


void wnga_pgroup_sync(Integer grp_id)
{
    fprintf(fptrace, "%lf,pnga_pgroup_sync,(%ld)\n",MPI_Wtime()-first_wtime,grp_id);
    pnga_pgroup_sync(grp_id);
    fprintf(fptrace, "%lf,/pnga_pgroup_sync,(%ld)\n",MPI_Wtime()-first_wtime,grp_id);

}


void wnga_print(Integer g_a)
{
    fprintf(fptrace, "%lf,pnga_print,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    pnga_print(g_a);
    fprintf(fptrace, "%lf,/pnga_print,(%ld)\n",MPI_Wtime()-first_wtime,g_a);

}


void wnga_print_distribution(int fstyle, Integer g_a)
{
    fprintf(fptrace, "%lf,pnga_print_distribution,(%d;%ld)\n",MPI_Wtime()-first_wtime,fstyle,g_a);
    pnga_print_distribution(fstyle, g_a);
    fprintf(fptrace, "%lf,/pnga_print_distribution,(%d;%ld)\n",MPI_Wtime()-first_wtime,fstyle,g_a);

}


void wnga_print_file(FILE *file, Integer g_a)
{
    fprintf(fptrace, "%lf,pnga_print_file,(%p;%ld)\n",MPI_Wtime()-first_wtime,file,g_a);
    pnga_print_file(file, g_a);
    fprintf(fptrace, "%lf,/pnga_print_file,(%p;%ld)\n",MPI_Wtime()-first_wtime,file,g_a);

}


void wnga_print_patch(Integer g_a, Integer *lo, Integer *hi, Integer pretty)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_print_patch,(%ld;%s;%s;%ld)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,0),expand_intp(_ndim,hi,1),pretty);
    pnga_print_patch(g_a, lo, hi, pretty);
    fprintf(fptrace, "%lf,/pnga_print_patch,(%ld;%s;%s;%ld)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,2),expand_intp(_ndim,hi,3),pretty);

}


void wnga_print_patch2d(Integer g_a, Integer ilo, Integer ihi, Integer jlo, Integer jhi, Integer pretty)
{
    fprintf(fptrace, "%lf,pnga_print_patch2d,(%ld;%ld;%ld;%ld;%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,ilo,ihi,jlo,jhi,pretty);
    pnga_print_patch2d(g_a, ilo, ihi, jlo, jhi, pretty);
    fprintf(fptrace, "%lf,/pnga_print_patch2d,(%ld;%ld;%ld;%ld;%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,ilo,ihi,jlo,jhi,pretty);

}


void wnga_print_patch_file(FILE *file, Integer g_a, Integer *lo, Integer *hi, Integer pretty)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_print_patch_file,(%p;%ld;%s;%s;%ld)\n",MPI_Wtime()-first_wtime,file,g_a,expand_intp(_ndim,lo,0),expand_intp(_ndim,hi,1),pretty);
    pnga_print_patch_file(file, g_a, lo, hi, pretty);
    fprintf(fptrace, "%lf,/pnga_print_patch_file,(%p;%ld;%s;%s;%ld)\n",MPI_Wtime()-first_wtime,file,g_a,expand_intp(_ndim,lo,2),expand_intp(_ndim,hi,3),pretty);

}


void wnga_print_patch_file2d(FILE *file, Integer g_a, Integer ilo, Integer ihi, Integer jlo, Integer jhi, Integer pretty)
{
    fprintf(fptrace, "%lf,pnga_print_patch_file2d,(%p;%ld;%ld;%ld;%ld;%ld;%ld)\n",MPI_Wtime()-first_wtime,file,g_a,ilo,ihi,jlo,jhi,pretty);
    pnga_print_patch_file2d(file, g_a, ilo, ihi, jlo, jhi, pretty);
    fprintf(fptrace, "%lf,/pnga_print_patch_file2d,(%p;%ld;%ld;%ld;%ld;%ld;%ld)\n",MPI_Wtime()-first_wtime,file,g_a,ilo,ihi,jlo,jhi,pretty);

}


void wnga_print_stats()
{
    fprintf(fptrace, "%lf,pnga_print_stats,\n",MPI_Wtime()-first_wtime);
    pnga_print_stats();
    fprintf(fptrace, "%lf,/pnga_print_stats,\n",MPI_Wtime()-first_wtime);

}


void wnga_proc_topology(Integer g_a, Integer proc, Integer *subscript)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_proc_topology,(%ld;%ld;%s)\n",MPI_Wtime()-first_wtime,g_a,proc,expand_intp(_ndim,subscript,0));
    pnga_proc_topology(g_a, proc, subscript);
    fprintf(fptrace, "%lf,/pnga_proc_topology,(%ld;%ld;%s)\n",MPI_Wtime()-first_wtime,g_a,proc,expand_intp(_ndim,subscript,1));

}


void wnga_put(Integer g_a, Integer *lo, Integer *hi, void *buf, Integer *ld)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_put,(%ld;%s;%s;%p;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,0),expand_intp(_ndim,hi,1),buf,expand_intp(_ndim-1,ld,2));
    pnga_put(g_a, lo, hi, buf, ld);
    fprintf(fptrace, "%lf,/pnga_put,(%ld;%s;%s;%p;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,3),expand_intp(_ndim,hi,4),buf,expand_intp(_ndim-1,ld,5));

}


void wnga_randomize(Integer g_a, void *val)
{
    fprintf(fptrace, "%lf,pnga_randomize,(%ld;%p)\n",MPI_Wtime()-first_wtime,g_a,val);
    pnga_randomize(g_a, val);
    fprintf(fptrace, "%lf,/pnga_randomize,(%ld;%p)\n",MPI_Wtime()-first_wtime,g_a,val);

}


Integer wnga_read_inc(Integer g_a, Integer *subscript, Integer inc)
{
    int _ndim = pnga_ndim(g_a);
    Integer retval;
    fprintf(fptrace, "%lf,pnga_read_inc,(%ld;%s;%ld)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,subscript,0),inc);
    retval = pnga_read_inc(g_a, subscript, inc);
    fprintf(fptrace, "%lf,/pnga_read_inc,(%ld;%s;%ld)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,subscript,1),inc);
    return retval;

}


void wnga_recip(Integer g_a)
{
    fprintf(fptrace, "%lf,pnga_recip,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    pnga_recip(g_a);
    fprintf(fptrace, "%lf,/pnga_recip,(%ld)\n",MPI_Wtime()-first_wtime,g_a);

}


void wnga_recip_patch(Integer g_a, Integer *lo, Integer *hi)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_recip_patch,(%ld;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,0),expand_intp(_ndim,hi,1));
    pnga_recip_patch(g_a, lo, hi);
    fprintf(fptrace, "%lf,/pnga_recip_patch,(%ld;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,2),expand_intp(_ndim,hi,3));

}


void wnga_release(Integer g_a, Integer *lo, Integer *hi)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_release,(%ld;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,0),expand_intp(_ndim,hi,1));
    pnga_release(g_a, lo, hi);
    fprintf(fptrace, "%lf,/pnga_release,(%ld;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,2),expand_intp(_ndim,hi,3));

}


void wnga_release_block(Integer g_a, Integer iblock)
{
    fprintf(fptrace, "%lf,pnga_release_block,(%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,iblock);
    pnga_release_block(g_a, iblock);
    fprintf(fptrace, "%lf,/pnga_release_block,(%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,iblock);

}


void wnga_release_block_grid(Integer g_a, Integer *index)
{
    fprintf(fptrace, "%lf,pnga_release_block_grid,(%ld;%p)\n",MPI_Wtime()-first_wtime,g_a,index);
    pnga_release_block_grid(g_a, index);
    fprintf(fptrace, "%lf,/pnga_release_block_grid,(%ld;%p)\n",MPI_Wtime()-first_wtime,g_a,index);

}


void wnga_release_block_segment(Integer g_a, Integer iproc)
{
    fprintf(fptrace, "%lf,pnga_release_block_segment,(%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,iproc);
    pnga_release_block_segment(g_a, iproc);
    fprintf(fptrace, "%lf,/pnga_release_block_segment,(%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,iproc);

}


void wnga_release_ghost_element(Integer g_a, Integer subscript[])
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_release_ghost_element,(%ld;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,subscript,0));
    pnga_release_ghost_element(g_a, subscript);
    fprintf(fptrace, "%lf,/pnga_release_ghost_element,(%ld;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,subscript,1));

}


void wnga_release_ghosts(Integer g_a)
{
    fprintf(fptrace, "%lf,pnga_release_ghosts,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    pnga_release_ghosts(g_a);
    fprintf(fptrace, "%lf,/pnga_release_ghosts,(%ld)\n",MPI_Wtime()-first_wtime,g_a);

}


void wnga_release_update(Integer g_a, Integer *lo, Integer *hi)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_release_update,(%ld;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,0),expand_intp(_ndim,hi,1));
    pnga_release_update(g_a, lo, hi);
    fprintf(fptrace, "%lf,/pnga_release_update,(%ld;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,2),expand_intp(_ndim,hi,3));

}


void wnga_release_update_block(Integer g_a, Integer iblock)
{
    fprintf(fptrace, "%lf,pnga_release_update_block,(%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,iblock);
    pnga_release_update_block(g_a, iblock);
    fprintf(fptrace, "%lf,/pnga_release_update_block,(%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,iblock);

}


void wnga_release_update_block_grid(Integer g_a, Integer *index)
{
    fprintf(fptrace, "%lf,pnga_release_update_block_grid,(%ld;%p)\n",MPI_Wtime()-first_wtime,g_a,index);
    pnga_release_update_block_grid(g_a, index);
    fprintf(fptrace, "%lf,/pnga_release_update_block_grid,(%ld;%p)\n",MPI_Wtime()-first_wtime,g_a,index);

}


void wnga_release_update_block_segment(Integer g_a, Integer iproc)
{
    fprintf(fptrace, "%lf,pnga_release_update_block_segment,(%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,iproc);
    pnga_release_update_block_segment(g_a, iproc);
    fprintf(fptrace, "%lf,/pnga_release_update_block_segment,(%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,iproc);

}


void wnga_release_update_ghost_element(Integer g_a, Integer subscript[])
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_release_update_ghost_element,(%ld;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,subscript,0));
    pnga_release_update_ghost_element(g_a, subscript);
    fprintf(fptrace, "%lf,/pnga_release_update_ghost_element,(%ld;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,subscript,1));

}


void wnga_release_update_ghosts(Integer g_a)
{
    fprintf(fptrace, "%lf,pnga_release_update_ghosts,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    pnga_release_update_ghosts(g_a);
    fprintf(fptrace, "%lf,/pnga_release_update_ghosts,(%ld)\n",MPI_Wtime()-first_wtime,g_a);

}


void wnga_scale(Integer g_a, void *alpha)
{
    Integer _atype;
    pnga_inquire_type(g_a, &_atype);
    fprintf(fptrace, "%lf,pnga_scale,(%ld;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_voidp(_atype,alpha,0));
    pnga_scale(g_a, alpha);
    fprintf(fptrace, "%lf,/pnga_scale,(%ld;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_voidp(_atype,alpha,1));

}


void wnga_scale_cols(Integer g_a, Integer g_v)
{
    fprintf(fptrace, "%lf,pnga_scale_cols,(%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,g_v);
    pnga_scale_cols(g_a, g_v);
    fprintf(fptrace, "%lf,/pnga_scale_cols,(%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,g_v);

}


void wnga_scale_patch(Integer g_a, Integer *lo, Integer *hi, void *alpha)
{
    int _ndim = pnga_ndim(g_a);
    Integer _atype;
    pnga_inquire_type(g_a, &_atype);
    fprintf(fptrace, "%lf,pnga_scale_patch,(%ld;%s;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,0),expand_intp(_ndim,hi,1),expand_voidp(_atype,alpha,2));
    pnga_scale_patch(g_a, lo, hi, alpha);
    fprintf(fptrace, "%lf,/pnga_scale_patch,(%ld;%s;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,3),expand_intp(_ndim,hi,4),expand_voidp(_atype,alpha,5));

}


void wnga_scale_rows(Integer g_a, Integer g_v)
{
    fprintf(fptrace, "%lf,pnga_scale_rows,(%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,g_v);
    pnga_scale_rows(g_a, g_v);
    fprintf(fptrace, "%lf,/pnga_scale_rows,(%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,g_v);

}


void wnga_scan_add(Integer g_a, Integer g_b, Integer g_sbit, Integer lo, Integer hi, Integer excl)
{
    fprintf(fptrace, "%lf,pnga_scan_add,(%ld;%ld;%ld;%ld;%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,g_b,g_sbit,lo,hi,excl);
    pnga_scan_add(g_a, g_b, g_sbit, lo, hi, excl);
    fprintf(fptrace, "%lf,/pnga_scan_add,(%ld;%ld;%ld;%ld;%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,g_b,g_sbit,lo,hi,excl);

}


void wnga_scan_copy(Integer g_a, Integer g_b, Integer g_sbit, Integer lo, Integer hi)
{
    fprintf(fptrace, "%lf,pnga_scan_copy,(%ld;%ld;%ld;%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,g_b,g_sbit,lo,hi);
    pnga_scan_copy(g_a, g_b, g_sbit, lo, hi);
    fprintf(fptrace, "%lf,/pnga_scan_copy,(%ld;%ld;%ld;%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,g_b,g_sbit,lo,hi);

}


void wnga_scatter(Integer g_a, void *v, Integer *subscript, Integer nv)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_scatter,(%ld;%p;%s;%ld)\n",MPI_Wtime()-first_wtime,g_a,v,expand_intp(_ndim,subscript,0),nv);
    pnga_scatter(g_a, v, subscript, nv);
    fprintf(fptrace, "%lf,/pnga_scatter,(%ld;%p;%s;%ld)\n",MPI_Wtime()-first_wtime,g_a,v,expand_intp(_ndim,subscript,1),nv);

}


void wnga_scatter2d(Integer g_a, void *v, Integer *i, Integer *j, Integer nv)
{
    fprintf(fptrace, "%lf,pnga_scatter2d,(%ld;%p;%p;%p;%ld)\n",MPI_Wtime()-first_wtime,g_a,v,i,j,nv);
    pnga_scatter2d(g_a, v, i, j, nv);
    fprintf(fptrace, "%lf,/pnga_scatter2d,(%ld;%p;%p;%p;%ld)\n",MPI_Wtime()-first_wtime,g_a,v,i,j,nv);

}


void wnga_scatter_acc(Integer g_a, void *v, Integer subscript[], Integer nv, void *alpha)
{
    int _ndim = pnga_ndim(g_a);
    Integer _atype;
    pnga_inquire_type(g_a, &_atype);
    fprintf(fptrace, "%lf,pnga_scatter_acc,(%ld;%p;%s;%ld;%s)\n",MPI_Wtime()-first_wtime,g_a,v,expand_intp(_ndim,subscript,0),nv,expand_voidp(_atype,alpha,1));
    pnga_scatter_acc(g_a, v, subscript, nv, alpha);
    fprintf(fptrace, "%lf,/pnga_scatter_acc,(%ld;%p;%s;%ld;%s)\n",MPI_Wtime()-first_wtime,g_a,v,expand_intp(_ndim,subscript,2),nv,expand_voidp(_atype,alpha,3));

}


void wnga_scatter_acc2d(Integer g_a, void *v, Integer *i, Integer *j, Integer nv, void *alpha)
{
    Integer _atype;
    pnga_inquire_type(g_a, &_atype);
    fprintf(fptrace, "%lf,pnga_scatter_acc2d,(%ld;%p;%p;%p;%ld;%s)\n",MPI_Wtime()-first_wtime,g_a,v,i,j,nv,expand_voidp(_atype,alpha,0));
    pnga_scatter_acc2d(g_a, v, i, j, nv, alpha);
    fprintf(fptrace, "%lf,/pnga_scatter_acc2d,(%ld;%p;%p;%p;%ld;%s)\n",MPI_Wtime()-first_wtime,g_a,v,i,j,nv,expand_voidp(_atype,alpha,1));

}


void wnga_select_elem(Integer g_a, char *op, void *val, Integer *subscript)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_select_elem,(%ld;%s;%p;%s)\n",MPI_Wtime()-first_wtime,g_a,op,val,expand_intp(_ndim,subscript,0));
    pnga_select_elem(g_a, op, val, subscript);
    fprintf(fptrace, "%lf,/pnga_select_elem,(%ld;%s;%p;%s)\n",MPI_Wtime()-first_wtime,g_a,op,val,expand_intp(_ndim,subscript,1));

}


void wnga_set_array_name(Integer g_a, char *array_name)
{
    fprintf(fptrace, "%lf,pnga_set_array_name,(%ld;%s)\n",MPI_Wtime()-first_wtime,g_a,array_name);
    pnga_set_array_name(g_a, array_name);
    fprintf(fptrace, "%lf,/pnga_set_array_name,(%ld;%s)\n",MPI_Wtime()-first_wtime,g_a,array_name);

}


void wnga_set_block_cyclic(Integer g_a, Integer *dims)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_set_block_cyclic,(%ld;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,dims,0));
    pnga_set_block_cyclic(g_a, dims);
    fprintf(fptrace, "%lf,/pnga_set_block_cyclic,(%ld;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,dims,1));

}


void wnga_set_block_cyclic_proc_grid(Integer g_a, Integer *dims, Integer *proc_grid)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_set_block_cyclic_proc_grid,(%ld;%s;%p)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,dims,0),proc_grid);
    pnga_set_block_cyclic_proc_grid(g_a, dims, proc_grid);
    fprintf(fptrace, "%lf,/pnga_set_block_cyclic_proc_grid,(%ld;%s;%p)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,dims,1),proc_grid);

}


void wnga_set_chunk(Integer g_a, Integer *chunk)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_set_chunk,(%ld;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,chunk,0));
    pnga_set_chunk(g_a, chunk);
    fprintf(fptrace, "%lf,/pnga_set_chunk,(%ld;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,chunk,1));

}


void wnga_set_debug(logical flag)
{
    fprintf(fptrace, "%lf,pnga_set_debug,(%ld)\n",MPI_Wtime()-first_wtime,flag);
    pnga_set_debug(flag);
    fprintf(fptrace, "%lf,/pnga_set_debug,(%ld)\n",MPI_Wtime()-first_wtime,flag);

}


void wnga_set_diagonal(Integer g_a, Integer g_v)
{
    fprintf(fptrace, "%lf,pnga_set_diagonal,(%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,g_v);
    pnga_set_diagonal(g_a, g_v);
    fprintf(fptrace, "%lf,/pnga_set_diagonal,(%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,g_v);

}


void wnga_set_ghost_corner_flag(Integer g_a, logical flag)
{
    fprintf(fptrace, "%lf,pnga_set_ghost_corner_flag,(%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,flag);
    pnga_set_ghost_corner_flag(g_a, flag);
    fprintf(fptrace, "%lf,/pnga_set_ghost_corner_flag,(%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,flag);

}


logical wnga_set_ghost_info(Integer g_a)
{
    logical retval;
    fprintf(fptrace, "%lf,pnga_set_ghost_info,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    retval = pnga_set_ghost_info(g_a);
    fprintf(fptrace, "%lf,/pnga_set_ghost_info,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    return retval;

}


void wnga_set_ghosts(Integer g_a, Integer *width)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_set_ghosts,(%ld;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,width,0));
    pnga_set_ghosts(g_a, width);
    fprintf(fptrace, "%lf,/pnga_set_ghosts,(%ld;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,width,1));

}


void wnga_set_irreg_distr(Integer g_a, Integer *map, Integer *block)
{
    fprintf(fptrace, "%lf,pnga_set_irreg_distr,(%ld;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(sum_intp(pnga_set_data_last_ndim,block),map,0),expand_intp(pnga_set_data_last_ndim,block,1));
    pnga_set_irreg_distr(g_a, map, block);
    fprintf(fptrace, "%lf,/pnga_set_irreg_distr,(%ld;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(sum_intp(pnga_set_data_last_ndim,block),map,2),expand_intp(pnga_set_data_last_ndim,block,3));

}


void wnga_set_irreg_flag(Integer g_a, logical flag)
{
    fprintf(fptrace, "%lf,pnga_set_irreg_flag,(%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,flag);
    pnga_set_irreg_flag(g_a, flag);
    fprintf(fptrace, "%lf,/pnga_set_irreg_flag,(%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,flag);

}


void wnga_set_memory_limit(Integer mem_limit)
{
    fprintf(fptrace, "%lf,pnga_set_memory_limit,(%ld)\n",MPI_Wtime()-first_wtime,mem_limit);
    pnga_set_memory_limit(mem_limit);
    fprintf(fptrace, "%lf,/pnga_set_memory_limit,(%ld)\n",MPI_Wtime()-first_wtime,mem_limit);

}


void wnga_set_pgroup(Integer g_a, Integer p_handle)
{
    fprintf(fptrace, "%lf,pnga_set_pgroup,(%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,p_handle);
    pnga_set_pgroup(g_a, p_handle);
    fprintf(fptrace, "%lf,/pnga_set_pgroup,(%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,p_handle);

}


void wnga_set_restricted(Integer g_a, Integer *list, Integer size)
{
    fprintf(fptrace, "%lf,pnga_set_restricted,(%ld;%p;%ld)\n",MPI_Wtime()-first_wtime,g_a,list,size);
    pnga_set_restricted(g_a, list, size);
    fprintf(fptrace, "%lf,/pnga_set_restricted,(%ld;%p;%ld)\n",MPI_Wtime()-first_wtime,g_a,list,size);

}


void wnga_set_restricted_range(Integer g_a, Integer lo_proc, Integer hi_proc)
{
    fprintf(fptrace, "%lf,pnga_set_restricted_range,(%ld;%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,lo_proc,hi_proc);
    pnga_set_restricted_range(g_a, lo_proc, hi_proc);
    fprintf(fptrace, "%lf,/pnga_set_restricted_range,(%ld;%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,lo_proc,hi_proc);

}


logical wnga_set_update4_info(Integer g_a)
{
    logical retval;
    fprintf(fptrace, "%lf,pnga_set_update4_info,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    retval = pnga_set_update4_info(g_a);
    fprintf(fptrace, "%lf,/pnga_set_update4_info,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    return retval;

}


logical wnga_set_update5_info(Integer g_a)
{
    logical retval;
    fprintf(fptrace, "%lf,pnga_set_update5_info,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    retval = pnga_set_update5_info(g_a);
    fprintf(fptrace, "%lf,/pnga_set_update5_info,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    return retval;

}


void wnga_shift_diagonal(Integer g_a, void *c)
{
    fprintf(fptrace, "%lf,pnga_shift_diagonal,(%ld;%p)\n",MPI_Wtime()-first_wtime,g_a,c);
    pnga_shift_diagonal(g_a, c);
    fprintf(fptrace, "%lf,/pnga_shift_diagonal,(%ld;%p)\n",MPI_Wtime()-first_wtime,g_a,c);

}


Integer wnga_solve(Integer g_a, Integer g_b)
{
    Integer retval;
    fprintf(fptrace, "%lf,pnga_solve,(%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,g_b);
    retval = pnga_solve(g_a, g_b);
    fprintf(fptrace, "%lf,/pnga_solve,(%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,g_b);
    return retval;

}


Integer wnga_spd_invert(Integer g_a)
{
    Integer retval;
    fprintf(fptrace, "%lf,pnga_spd_invert,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    retval = pnga_spd_invert(g_a);
    fprintf(fptrace, "%lf,/pnga_spd_invert,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    return retval;

}


void wnga_step_bound_info(Integer g_xx, Integer g_vv, Integer g_xxll, Integer g_xxuu, void *boundmin, void *wolfemin, void *boundmax)
{
    fprintf(fptrace, "%lf,pnga_step_bound_info,(%ld;%ld;%ld;%ld;%p;%p;%p)\n",MPI_Wtime()-first_wtime,g_xx,g_vv,g_xxll,g_xxuu,boundmin,wolfemin,boundmax);
    pnga_step_bound_info(g_xx, g_vv, g_xxll, g_xxuu, boundmin, wolfemin, boundmax);
    fprintf(fptrace, "%lf,/pnga_step_bound_info,(%ld;%ld;%ld;%ld;%p;%p;%p)\n",MPI_Wtime()-first_wtime,g_xx,g_vv,g_xxll,g_xxuu,boundmin,wolfemin,boundmax);

}


void wnga_step_bound_info_patch(Integer g_xx, Integer *xxlo, Integer *xxhi, Integer g_vv, Integer *vvlo, Integer *vvhi, Integer g_xxll, Integer *xxlllo, Integer *xxllhi, Integer g_xxuu, Integer *xxuulo, Integer *xxuuhi, void *boundmin, void *wolfemin, void *boundmax)
{
    fprintf(fptrace, "%lf,pnga_step_bound_info_patch,(%ld;%p;%p;%ld;%p;%p;%ld;%p;%p;%ld;%p;%p;%p;%p;%p)\n",MPI_Wtime()-first_wtime,g_xx,xxlo,xxhi,g_vv,vvlo,vvhi,g_xxll,xxlllo,xxllhi,g_xxuu,xxuulo,xxuuhi,boundmin,wolfemin,boundmax);
    pnga_step_bound_info_patch(g_xx, xxlo, xxhi, g_vv, vvlo, vvhi, g_xxll, xxlllo, xxllhi, g_xxuu, xxuulo, xxuuhi, boundmin, wolfemin, boundmax);
    fprintf(fptrace, "%lf,/pnga_step_bound_info_patch,(%ld;%p;%p;%ld;%p;%p;%ld;%p;%p;%ld;%p;%p;%p;%p;%p)\n",MPI_Wtime()-first_wtime,g_xx,xxlo,xxhi,g_vv,vvlo,vvhi,g_xxll,xxlllo,xxllhi,g_xxuu,xxuulo,xxuuhi,boundmin,wolfemin,boundmax);

}


void wnga_step_mask_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_step_mask_patch,(%ld;%s;%s;%ld;%s;%s;%ld;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,alo,0),expand_intp(_ndim,ahi,1),g_b,expand_intp(_ndim,blo,2),expand_intp(_ndim,bhi,3),g_c,expand_intp(_ndim,clo,4),expand_intp(_ndim,chi,5));
    pnga_step_mask_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi);
    fprintf(fptrace, "%lf,/pnga_step_mask_patch,(%ld;%s;%s;%ld;%s;%s;%ld;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,alo,6),expand_intp(_ndim,ahi,7),g_b,expand_intp(_ndim,blo,8),expand_intp(_ndim,bhi,9),g_c,expand_intp(_ndim,clo,10),expand_intp(_ndim,chi,11));

}


void wnga_step_max(Integer g_a, Integer g_b, void *retval)
{
    fprintf(fptrace, "%lf,pnga_step_max,(%ld;%ld;%p)\n",MPI_Wtime()-first_wtime,g_a,g_b,retval);
    pnga_step_max(g_a, g_b, retval);
    fprintf(fptrace, "%lf,/pnga_step_max,(%ld;%ld;%p)\n",MPI_Wtime()-first_wtime,g_a,g_b,retval);

}


void wnga_step_max_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, void *result)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_step_max_patch,(%ld;%s;%s;%ld;%s;%s;%p)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,alo,0),expand_intp(_ndim,ahi,1),g_b,expand_intp(_ndim,blo,2),expand_intp(_ndim,bhi,3),result);
    pnga_step_max_patch(g_a, alo, ahi, g_b, blo, bhi, result);
    fprintf(fptrace, "%lf,/pnga_step_max_patch,(%ld;%s;%s;%ld;%s;%s;%p)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,alo,4),expand_intp(_ndim,ahi,5),g_b,expand_intp(_ndim,blo,6),expand_intp(_ndim,bhi,7),result);

}


void wnga_strided_acc(Integer g_a, Integer *lo, Integer *hi, Integer *skip, void *buf, Integer *ld, void *alpha)
{
    int _ndim = pnga_ndim(g_a);
    Integer _atype;
    pnga_inquire_type(g_a, &_atype);
    fprintf(fptrace, "%lf,pnga_strided_acc,(%ld;%s;%s;%p;%p;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,0),expand_intp(_ndim,hi,1),skip,buf,expand_intp(_ndim-1,ld,2),expand_voidp(_atype,alpha,3));
    pnga_strided_acc(g_a, lo, hi, skip, buf, ld, alpha);
    fprintf(fptrace, "%lf,/pnga_strided_acc,(%ld;%s;%s;%p;%p;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,4),expand_intp(_ndim,hi,5),skip,buf,expand_intp(_ndim-1,ld,6),expand_voidp(_atype,alpha,7));

}


void wnga_strided_get(Integer g_a, Integer *lo, Integer *hi, Integer *skip, void *buf, Integer *ld)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_strided_get,(%ld;%s;%s;%p;%p;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,0),expand_intp(_ndim,hi,1),skip,buf,expand_intp(_ndim-1,ld,2));
    pnga_strided_get(g_a, lo, hi, skip, buf, ld);
    fprintf(fptrace, "%lf,/pnga_strided_get,(%ld;%s;%s;%p;%p;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,3),expand_intp(_ndim,hi,4),skip,buf,expand_intp(_ndim-1,ld,5));

}


void wnga_strided_put(Integer g_a, Integer *lo, Integer *hi, Integer *skip, void *buf, Integer *ld)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_strided_put,(%ld;%s;%s;%p;%p;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,0),expand_intp(_ndim,hi,1),skip,buf,expand_intp(_ndim-1,ld,2));
    pnga_strided_put(g_a, lo, hi, skip, buf, ld);
    fprintf(fptrace, "%lf,/pnga_strided_put,(%ld;%s;%s;%p;%p;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,3),expand_intp(_ndim,hi,4),skip,buf,expand_intp(_ndim-1,ld,5));

}


void wnga_summarize(Integer verbose)
{
    fprintf(fptrace, "%lf,pnga_summarize,(%ld)\n",MPI_Wtime()-first_wtime,verbose);
    pnga_summarize(verbose);
    fprintf(fptrace, "%lf,/pnga_summarize,(%ld)\n",MPI_Wtime()-first_wtime,verbose);

}


void wnga_symmetrize(Integer g_a)
{
    fprintf(fptrace, "%lf,pnga_symmetrize,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    pnga_symmetrize(g_a);
    fprintf(fptrace, "%lf,/pnga_symmetrize,(%ld)\n",MPI_Wtime()-first_wtime,g_a);

}


void wnga_sync()
{
    fprintf(fptrace, "%lf,pnga_sync,\n",MPI_Wtime()-first_wtime);
    pnga_sync();
    fprintf(fptrace, "%lf,/pnga_sync,\n",MPI_Wtime()-first_wtime);

}


double wnga_timer()
{
    double retval;
    fprintf(fptrace, "%lf,pnga_timer,\n",MPI_Wtime()-first_wtime);
    retval = pnga_timer();
    fprintf(fptrace, "%lf,/pnga_timer,\n",MPI_Wtime()-first_wtime);
    return retval;

}


Integer wnga_total_blocks(Integer g_a)
{
    Integer retval;
    fprintf(fptrace, "%lf,pnga_total_blocks,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    retval = pnga_total_blocks(g_a);
    fprintf(fptrace, "%lf,/pnga_total_blocks,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    return retval;

}


void wnga_transpose(Integer g_a, Integer g_b)
{
    fprintf(fptrace, "%lf,pnga_transpose,(%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,g_b);
    pnga_transpose(g_a, g_b);
    fprintf(fptrace, "%lf,/pnga_transpose,(%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,g_b);

}


Integer wnga_type_c2f(Integer type)
{
    Integer retval;
    fprintf(fptrace, "%lf,pnga_type_c2f,(%ld)\n",MPI_Wtime()-first_wtime,type);
    retval = pnga_type_c2f(type);
    fprintf(fptrace, "%lf,/pnga_type_c2f,(%ld)\n",MPI_Wtime()-first_wtime,type);
    return retval;

}


Integer wnga_type_f2c(Integer type)
{
    Integer retval;
    fprintf(fptrace, "%lf,pnga_type_f2c,(%ld)\n",MPI_Wtime()-first_wtime,type);
    retval = pnga_type_f2c(type);
    fprintf(fptrace, "%lf,/pnga_type_f2c,(%ld)\n",MPI_Wtime()-first_wtime,type);
    return retval;

}


void wnga_unlock(Integer mutex)
{
    fprintf(fptrace, "%lf,pnga_unlock,(%ld)\n",MPI_Wtime()-first_wtime,mutex);
    pnga_unlock(mutex);
    fprintf(fptrace, "%lf,/pnga_unlock,(%ld)\n",MPI_Wtime()-first_wtime,mutex);

}


void wnga_unpack(Integer g_a, Integer g_b, Integer g_sbit, Integer lo, Integer hi, Integer *icount)
{
    fprintf(fptrace, "%lf,pnga_unpack,(%ld;%ld;%ld;%ld;%ld;%p)\n",MPI_Wtime()-first_wtime,g_a,g_b,g_sbit,lo,hi,icount);
    pnga_unpack(g_a, g_b, g_sbit, lo, hi, icount);
    fprintf(fptrace, "%lf,/pnga_unpack,(%ld;%ld;%ld;%ld;%ld;%p)\n",MPI_Wtime()-first_wtime,g_a,g_b,g_sbit,lo,hi,icount);

}


void wnga_update1_ghosts(Integer g_a)
{
    fprintf(fptrace, "%lf,pnga_update1_ghosts,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    pnga_update1_ghosts(g_a);
    fprintf(fptrace, "%lf,/pnga_update1_ghosts,(%ld)\n",MPI_Wtime()-first_wtime,g_a);

}


logical wnga_update2_ghosts(Integer g_a)
{
    logical retval;
    fprintf(fptrace, "%lf,pnga_update2_ghosts,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    retval = pnga_update2_ghosts(g_a);
    fprintf(fptrace, "%lf,/pnga_update2_ghosts,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    return retval;

}


logical wnga_update3_ghosts(Integer g_a)
{
    logical retval;
    fprintf(fptrace, "%lf,pnga_update3_ghosts,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    retval = pnga_update3_ghosts(g_a);
    fprintf(fptrace, "%lf,/pnga_update3_ghosts,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    return retval;

}


logical wnga_update44_ghosts(Integer g_a)
{
    logical retval;
    fprintf(fptrace, "%lf,pnga_update44_ghosts,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    retval = pnga_update44_ghosts(g_a);
    fprintf(fptrace, "%lf,/pnga_update44_ghosts,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    return retval;

}


logical wnga_update4_ghosts(Integer g_a)
{
    logical retval;
    fprintf(fptrace, "%lf,pnga_update4_ghosts,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    retval = pnga_update4_ghosts(g_a);
    fprintf(fptrace, "%lf,/pnga_update4_ghosts,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    return retval;

}


logical wnga_update55_ghosts(Integer g_a)
{
    logical retval;
    fprintf(fptrace, "%lf,pnga_update55_ghosts,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    retval = pnga_update55_ghosts(g_a);
    fprintf(fptrace, "%lf,/pnga_update55_ghosts,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    return retval;

}


logical wnga_update5_ghosts(Integer g_a)
{
    logical retval;
    fprintf(fptrace, "%lf,pnga_update5_ghosts,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    retval = pnga_update5_ghosts(g_a);
    fprintf(fptrace, "%lf,/pnga_update5_ghosts,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    return retval;

}


logical wnga_update6_ghosts(Integer g_a)
{
    logical retval;
    fprintf(fptrace, "%lf,pnga_update6_ghosts,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    retval = pnga_update6_ghosts(g_a);
    fprintf(fptrace, "%lf,/pnga_update6_ghosts,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    return retval;

}


logical wnga_update7_ghosts(Integer g_a)
{
    logical retval;
    fprintf(fptrace, "%lf,pnga_update7_ghosts,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    retval = pnga_update7_ghosts(g_a);
    fprintf(fptrace, "%lf,/pnga_update7_ghosts,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    return retval;

}


logical wnga_update_ghost_dir(Integer g_a, Integer pdim, Integer pdir, logical pflag)
{
    logical retval;
    fprintf(fptrace, "%lf,pnga_update_ghost_dir,(%ld;%ld;%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,pdim,pdir,pflag);
    retval = pnga_update_ghost_dir(g_a, pdim, pdir, pflag);
    fprintf(fptrace, "%lf,/pnga_update_ghost_dir,(%ld;%ld;%ld;%ld)\n",MPI_Wtime()-first_wtime,g_a,pdim,pdir,pflag);
    return retval;

}


void wnga_update_ghosts(Integer g_a)
{
    fprintf(fptrace, "%lf,pnga_update_ghosts,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    pnga_update_ghosts(g_a);
    fprintf(fptrace, "%lf,/pnga_update_ghosts,(%ld)\n",MPI_Wtime()-first_wtime,g_a);

}


logical wnga_uses_ma()
{
    logical retval;
    fprintf(fptrace, "%lf,pnga_uses_ma,\n",MPI_Wtime()-first_wtime);
    retval = pnga_uses_ma();
    fprintf(fptrace, "%lf,/pnga_uses_ma,\n",MPI_Wtime()-first_wtime);
    return retval;

}


logical wnga_uses_proc_grid(Integer g_a)
{
    logical retval;
    fprintf(fptrace, "%lf,pnga_uses_proc_grid,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    retval = pnga_uses_proc_grid(g_a);
    fprintf(fptrace, "%lf,/pnga_uses_proc_grid,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    return retval;

}


logical wnga_valid_handle(Integer g_a)
{
    logical retval;
    fprintf(fptrace, "%lf,pnga_valid_handle,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    retval = pnga_valid_handle(g_a);
    fprintf(fptrace, "%lf,/pnga_valid_handle,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    return retval;

}


Integer wnga_verify_handle(Integer g_a)
{
    Integer retval;
    fprintf(fptrace, "%lf,pnga_verify_handle,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    retval = pnga_verify_handle(g_a);
    fprintf(fptrace, "%lf,/pnga_verify_handle,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    return retval;

}


DoublePrecision wnga_wtime()
{
    DoublePrecision retval;
    fprintf(fptrace, "%lf,pnga_wtime,\n",MPI_Wtime()-first_wtime);
    retval = pnga_wtime();
    fprintf(fptrace, "%lf,/pnga_wtime,\n",MPI_Wtime()-first_wtime);
    return retval;

}


void wnga_zero(Integer g_a)
{
    fprintf(fptrace, "%lf,pnga_zero,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    pnga_zero(g_a);
    fprintf(fptrace, "%lf,/pnga_zero,(%ld)\n",MPI_Wtime()-first_wtime,g_a);

}


void wnga_zero_diagonal(Integer g_a)
{
    fprintf(fptrace, "%lf,pnga_zero_diagonal,(%ld)\n",MPI_Wtime()-first_wtime,g_a);
    pnga_zero_diagonal(g_a);
    fprintf(fptrace, "%lf,/pnga_zero_diagonal,(%ld)\n",MPI_Wtime()-first_wtime,g_a);

}


void wnga_zero_patch(Integer g_a, Integer *lo, Integer *hi)
{
    int _ndim = pnga_ndim(g_a);
    fprintf(fptrace, "%lf,pnga_zero_patch,(%ld;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,0),expand_intp(_ndim,hi,1));
    pnga_zero_patch(g_a, lo, hi);
    fprintf(fptrace, "%lf,/pnga_zero_patch,(%ld;%s;%s)\n",MPI_Wtime()-first_wtime,g_a,expand_intp(_ndim,lo,2),expand_intp(_ndim,hi,3));

}


void wnga_set_data(Integer g_a, Integer ndim, Integer *dims, Integer type)
{
    pnga_set_data_last_ndim = ndim;
    fprintf(fptrace, "%lf,pnga_set_data,(%ld;%ld;%s;%ld)\n",MPI_Wtime()-first_wtime,g_a,ndim,expand_intp(ndim,dims,0),type);
    pnga_set_data(g_a, ndim, dims, type);
    fprintf(fptrace, "%lf,/pnga_set_data,(%ld;%ld;%s;%ld)\n",MPI_Wtime()-first_wtime,g_a,ndim,expand_intp(ndim,dims,1),type);

}

void wnga_initialize()
{
    static int count_pnga_initialize=0;

    ++count_pnga_initialize;
    if (1 == count_pnga_initialize) {
        trace_initialize();
    }
    fprintf(fptrace, "%lf,pnga_initialize,\n",MPI_Wtime()-first_wtime);
    pnga_initialize();
    fprintf(fptrace, "%lf,/pnga_initialize,\n",MPI_Wtime()-first_wtime);

}

void wnga_initialize_ltd(Integer limit)
{
    static int count_pnga_initialize_ltd=0;

    ++count_pnga_initialize_ltd;
    if (1 == count_pnga_initialize_ltd) {
        trace_initialize();
    }
    fprintf(fptrace, "%lf,pnga_initialize_ltd,(%ld)\n",MPI_Wtime()-first_wtime,limit);
    pnga_initialize_ltd(limit);
    fprintf(fptrace, "%lf,/pnga_initialize_ltd,(%ld)\n",MPI_Wtime()-first_wtime,limit);

}

void wnga_terminate()
{
    static int count_pnga_terminate=0;

    ++count_pnga_terminate;
    fprintf(fptrace, "%lf,pnga_terminate,\n",MPI_Wtime()-first_wtime);
    pnga_terminate();
    fprintf(fptrace, "%lf,/pnga_terminate,\n",MPI_Wtime()-first_wtime);

    /* don't dump info if terminate more than once */
    if (1 == count_pnga_terminate) {
        trace_finalize();
    }
}

