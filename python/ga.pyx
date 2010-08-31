"""
The Global Arrays (GA) Python interface.

This module exports the GA C API, with a few enhancements.  This module also
provides the GlobalArray object-oriented class.

"""
# keep the ga functions alphabetical since this is going to be a huge file!

from libc.stdlib cimport malloc,free
from gah cimport *
import numpy as np
cimport numpy as np

np.import_array()

TYPE_BASE  = 1000
C_CHAR     = (TYPE_BASE + 0)
C_INT      = (TYPE_BASE + 1)
C_LONG     = (TYPE_BASE + 2)
C_FLT      = (TYPE_BASE + 3)
C_DBL      = (TYPE_BASE + 4)
C_LDBL     = (TYPE_BASE + 5)
C_SCPL     = (TYPE_BASE + 6)
C_DCPL     = (TYPE_BASE + 7)
C_LDCPL    = (TYPE_BASE + 8)
F_BYTE     = (TYPE_BASE + 9)
F_INT      = (TYPE_BASE + 10)
F_LOG      = (TYPE_BASE + 11)
F_REAL     = (TYPE_BASE + 12)
F_DBL      = (TYPE_BASE + 13)
F_SCPL     = (TYPE_BASE + 14)
F_DCPL     = (TYPE_BASE + 15)
C_LONGLONG = (TYPE_BASE + 16)

_to_typenum = {
        C_INT: np.NPY_INT,
        C_LONG: np.NPY_LONG,
        C_LONGLONG: np.NPY_LONGLONG,
        C_FLT: np.NPY_FLOAT,
        C_DBL: np.NPY_DOUBLE
        }

_to_dtype = {
        C_INT: np.intc,
        C_LONG: np.long,
        C_LONGLONG: np.longlong,
        C_FLT: np.single,
        C_DBL: np.double
        }

cdef void* _gapy_malloc(size_t bytes, int align, char *name):
    return malloc(bytes)

cdef void _gapy_free(void *ptr):
    free(ptr)

def abs_value(int g_a, lo=None, hi=None):
    """Take element-wise absolute value of the array or patch.
    
    This is a collective operation.

    Positional arguments:
    g_a -- the array handle

    Keyword arguments:
    lo -- lower bound patch coordinates, inclusive
    hi -- higher bound patch coordinates, inclusive
    
    """
    cdef np.ndarray[np.int64_t, ndim=1] lo_nd, hi_nd
    if lo is None and hi is None:
        GA_Abs_value(g_a)
    else:
        lo_nd,hi_nd = _lohi(g_a,lo,hi)
        GA_Abs_value_patch64(g_a, <int64_t*>lo_nd.data, <int64_t*>hi_nd.data)

def acc(int g_a, lo, hi, buffer, alpha=None):
    """Combines data from buffer with data in the global array patch.
    
    The buffer array is assumed to be have the same number of
    dimensions as the global array.  If the buffer is not contiguous, a
    contiguous copy will be made.
    
        global array section (lo[],hi[]) += alpha * buffer

    This is a one-sided and atomic operation.

    Positional arguments:
    g_a    -- the array handle
    lo     -- lower bound patch coordinates, inclusive
    hi     -- higher bound patch coordinates, inclusive
    buffer -- an array-like object with same shape as indicated patch

    Keyword arguments:
    alpha  -- multiplier

    """
    cdef np.ndarray[np.int64_t, ndim=1] lo_nd, hi_nd, ld_nd, shape
    cdef np.ndarray buffer_nd
    cdef int gtype=inquire_type(g_a)
    cdef int       ialpha=1
    cdef long      lalpha=1
    cdef long long llalpha=1
    cdef float     falpha=1.0
    cdef double    dalpha=1.0
    cdef void     *valpha=NULL
    dtype = _to_dtype[gtype]
    buffer_nd = np.asarray(buffer, dtype=dtype)
    lo_nd = np.asarray(lo, dtype=np.int64)
    hi_nd = np.asarray(hi, dtype=np.int64)
    shape = hi_nd-lo_nd+1
    ld_nd = shape[1:]
    if buffer_nd.dtype != dtype or not buffer_nd.flags['C_CONTIGUOUS']:
        buffer_nd = np.ascontiguousarray(buffer_nd, dtype=dtype)
    buffer_nd = np.reshape(buffer_nd, shape)
    valpha = _convert_multiplier(gtype, alpha,
            &ialpha, &lalpha, &llalpha, &falpha, &dalpha)
    NGA_Acc64(g_a, <int64_t*>lo_nd.data, <int64_t*>hi_nd.data,
            <void*>buffer_nd.data, <int64_t*>ld_nd.data, valpha)

def access(int g_a, lo=None, hi=None):
    """Returns local array patch.
    
    This routine allows to access directly, in place elements in the local
    section of a global array. It useful for writing new GA operations.
    If no patch is specified, the entire local patch is returned.  If this
    process does not own any data, None is returned.
    
    Note: The entire local data is always accessed, but if a smaller patch is
    requested, an appropriately sliced ndarray is returned.

    Each call to ga.access has to be followed by a call to either ga.release
    or ga.release_update. You can access in this fashion only local data.
    Since the data is shared with other processes, you need to consider issues
    of mutual exclusion.

    This operation is local. 

    Positional arguments:
    g_a -- the array handle

    Keyword arguments:
    lo -- lower bound patch coordinates, inclusive
    hi -- higher bound patch coordinates, inclusive
    
    Returns:
    ndarray representing local patch

    """
    cdef np.ndarray[np.int64_t, ndim=1] lo_nd, hi_nd
    cdef np.ndarray[np.int64_t, ndim=1] ld_nd, lo_dst, hi_dst, dims_nd
    cdef int i, gtype=inquire_type(g_a)
    cdef int dimlen=GA_Ndim(g_a), typenum=_to_typenum[gtype]
    cdef void *ptr
    cdef np.npy_intp *dims = NULL
    # first things first, if no data is owned, return None
    lo_dst,hi_dst = distribution(g_a)
    if lo_dst[0] < 0 or hi_dst[0] < 0:
        return None
    # always access the entire local data
    ld_nd = np.zeros(dimlen-1, dtype=np.int64)
    NGA_Access64(g_a, <int64_t*>lo_dst.data, <int64_t*>hi_dst.data, &ptr,
            <int64_t*>ld_nd.data);
    dims_nd = hi_dst-lo_dst+1
    # must convert int64_t ndarray shape to npy_intp array
    dims = <np.npy_intp*>malloc(dimlen*sizeof(np.npy_intp))
    for i in range(dimlen):
        dims[i] = dims_nd[i]
    array = np.PyArray_SimpleNewFromData(dimlen, dims, typenum, ptr)
    free(dims)
    if lo is not None or hi is not None:
        if lo is not None:
            lo_nd = np.asarray(lo, dtype=np.int64)
        else:
            lo_nd = lo_dst
        if hi is not None:
            hi_nd = np.asarray(hi, dtype=np.int64)
        else:
            hi_nd = hi_dst
        # sanity checks
        if np.sometrue(lo_nd>hi_nd):
            raise ValueError,"lo>hi lo=%s hi=%s"%(lo_nd,hi_nd)
        if np.sometrue(lo_nd<lo_dst):
            raise ValueError,"lo out of bounds lo_dst=%s lo=%s"%(lo_dst,lo_nd)
        if np.sometrue(hi_nd>hi_dst):
            raise ValueError,"hi out of bounds hi_dst=%s hi=%s"%(hi_dst,hi_nd)
        slices = []
        for i in range(dimlen):
            slices.append(slice(lo_nd[i]-lo_dst[i],hi_nd[i]-hi_dst[i]))
        return array[slices]
    return array

def access_block(int g_a, int idx):
    """Returns local array patch for a block-cyclic distribution.
    
    This routine allows to access directly, in place elements in the local
    section of a global array. It useful for writing new GA operations.
    
    Each call to ga.access_block has to be followed by a call to either
    ga.release_block or ga.release_update_block. You can access in this
    fashion only local data.  Since the data is shared with other processes,
    you need to consider issues of mutual exclusion.

    This operation is local. 

    Positional arguments:
    g_a -- the array handle
    idx -- the block index

    Returns:
    ndarray representing local block

    """
    raise NotImplementedError

def access_block_grid(int g_a, subscript):
    """Returns local array patch for a SCALAPACK block-cyclic distribution.

    The subscript array contains the subscript of the block in the array of
    blocks. This subscript is based on the location of the block in a grid,
    each of whose dimensions is equal to the number of blocks that fit along
    that dimension.

    Each call to ga.access_block_grid has to be followed by a call to either
    ga.release_block_grid or ga.release_update_block_grid. You can access in
    this fashion only local data.  Since the data is shared with other
    processes, you need to consider issues of mutual exclusion.

    This operation is local. 

    Positional arguments:
    g_a       -- the array handle
    subscript -- subscript of the block in the array

    Returns:
    ndarray representing local block

    """
    raise NotImplementedError

def access_block_segment(int g_a, int proc):
    """Do not use.

    This function can be used to gain access to the all the locally held data
    on a particular processor that is associated with a block-cyclic
    distributed array.

    The data  inside this segment has a lot of additional structure so this
    function is not generally useful to developers. It is primarily used
    inside the GA library to implement other GA routines. Each call to
    ga_access_block_segment should be followed by a call to either
    NGA_Release_block_segment or NGA_Release_update_block_segment.

    This is a local operation.

    Positional arguments:
    g_a  -- the array handle
    proc -- processor ID

    Returns:
    ndarray representing local block

    """
    raise NotImplementedError

def access_ghost_element(int g_a, subscript, ld):
    """Returns a scalar ndarray representing the requested ghost element.

    This function can be used to return a pointer to any data element in the
    locally held portion of the global array and can be used to directly
    access ghost cell data. The array subscript refers to the local index of
    the  element relative to the origin of the local patch (which is assumed
    to be indexed by (0,0,...)).

    This is a  local operation. 

    Positional arguments:
    g_a       -- the array handle
    subscript -- array-like of integers that index desired element

    Returns:
    ndarray scalar representing local block

    """
    raise NotImplementedError

def access_ghosts(int g_a):
    """Returns ndarray representing local patch with ghost cells.

    This routine will provide access to the ghost cell data residing on each
    processor. Calls to NGA_Access_ghosts should normally follow a call to
    NGA_Distribution  that returns coordinates of the visible data patch
    associated with a processor. You need to make sure that the coordinates of
    the patch are valid (test values returned from NGA_Distribution).

    You can only access local data.

    This operation is local.

    Positional arguments:
    g_a       -- the array handle

    Returns:
    ndarray scalar representing local block with ghost cells

    """
    raise NotImplementedError

def add(int g_a, int g_b, int g_c, alpha=None, beta=None, alo=None, ahi=None,
        blo=None, bhi=None, clo=None, chi=None):
    """Element-wise addition of two arrays.

    The arrays must be the same shape and identically aligned.
        c = alpha * a  +  beta * b
    The result (c) may replace one of the input arrays (a/b).

    Patches of arrays (which must have the same number of elements) may also
    be added together elementw=-wise, if patch coordinates are specified.
        c[][] = alpha * a[][] + beta * b[][]. 

    This is a collective operation. 

    Positional arguments:
    g_a    -- the array handle
    g_b    -- the array handle
    g_c    -- the array handle

    Keyword arguments:
    alpha -- multiplier
    beta  -- multiplier
    alo   -- lower bound patch coordinates of g_a, inclusive
    ahi   -- higher bound patch coordinates of g_a, inclusive
    blo   -- lower bound patch coordinates of g_b, inclusive
    bhi   -- higher bound patch coordinates of g_b, inclusive
    clo   -- lower bound patch coordinates of g_c, inclusive
    chi   -- higher bound patch coordinates of g_c, inclusive

    """
    cdef np.ndarray[np.int64_t, ndim=1] alo_nd, ahi_nd
    cdef np.ndarray[np.int64_t, ndim=1] blo_nd, bhi_nd
    cdef np.ndarray[np.int64_t, ndim=1] clo_nd, chi_nd
    cdef int gtype=inquire_type(g_a)
    cdef int       ialpha=1,     ibeta=1
    cdef long      lalpha=1,     lbeta=1
    cdef long long llalpha=1,    llbeta=1
    cdef float     falpha=1.0,   fbeta=1.0
    cdef double    dalpha=1.0,   dbeta=1.0
    cdef void     *valpha=NULL, *vbeta=NULL
    dtype = _to_dtype[gtype]
    valpha = _convert_multiplier(gtype, alpha,
            &ialpha, &lalpha, &llalpha, &falpha, &dalpha)
    vbeta = _convert_multiplier(gtype, beta,
            &ibeta, &lbeta, &llbeta, &fbeta, &dbeta)
    if (alo is None and ahi is None
            and blo is None and bhi is None
            and clo is None and chi is None):
        GA_Add(valpha, g_a, vbeta, g_b, g_c)
    else:
        alo_nd,ahi_nd = _lohi(g_a,alo,ahi)
        blo_nd,bhi_nd = _lohi(g_b,blo,bhi)
        clo_nd,chi_nd = _lohi(g_c,clo,chi)
        NGA_Add_patch64(
                valpha, g_a, <int64_t*>alo_nd.data, <int64_t*>ahi_nd.data,
                 vbeta, g_b, <int64_t*>blo_nd.data, <int64_t*>bhi_nd.data,
                        g_c, <int64_t*>clo_nd.data, <int64_t*>chi_nd.data)

def add_constant(int g_a, alpha, lo=None, hi=None):
    """Adds the constant alpha to each element of the array. 

    This operation is collective.

    Positional arguments:
    g_a   -- the array handle
    alpha -- the constant to add

    Keyword arguments:
    lo    -- lower bound patch coordinates, inclusive
    hi    -- higher bound patch coordinates, inclusive

    """
    cdef np.ndarray[np.int64_t, ndim=1] lo_nd, hi_nd
    cdef int gtype=inquire_type(g_a)
    cdef int       ialpha=1,     ibeta=1
    cdef long      lalpha=1,     lbeta=1
    cdef long long llalpha=1,    llbeta=1
    cdef float     falpha=1.0,   fbeta=1.0
    cdef double    dalpha=1.0,   dbeta=1.0
    cdef void     *valpha=NULL, *vbeta=NULL
    valpha = _convert_multiplier(gtype, alpha,
            &ialpha, &lalpha, &llalpha, &falpha, &dalpha)
    if lo is None and hi is None:
        GA_Add_constant(g_a, valpha)
    else:
        lo_nd,hi_nd = _lohi(g_a,lo,hi)
        GA_Add_constant_patch64(
                g_a, <int64_t*>lo_nd.data, <int64_t*>hi_nd.data, valpha)

def add_diagonal(int g_a, int g_v):
    """Adds the elements of the vector g_v to the diagonal of matrix g_a.

    This operation is collective.

    Positional arguments:
    g_a -- the array handle
    g_v -- the vector handle

    """
    GA_Add_diagonal(g_a, g_v)

def allocate(int g_a):
    """Allocates memory for the handle obtained using ga.create_handle.

    At a minimum, the ga.set_data function must be called before the memory is
    allocated. Other ga.set_xxx functions can also be called before invoking
    this function.

    This is a collective operation. 

    Positional arguments:
    g_a -- the array handle

    Returns:
    TODO

    """
    return GA_Allocate(g_a)

def brdcst(np.ndarray buffer, int root):
    """Broadcast from process root to all other processes.

    If the buffer is not contiguous, an error is raised.  This operation is
    provided only for convenience purposes: it is available regardless of the
    message-passing library that GA is running with.

    This is a collective operation. 

    Positional arguments:
    buffer -- the ndarray message
    root   -- the process which is sending

    Returns:
    The buffer in case a temporary was passed in.

    """
    if not buffer.flags['C_CONTIGUOUS']:
        raise ValueError, "the buffer must be contiguous"
    if buffer.ndim != 1:
        raise ValueError, "the buffer must be one-dimensional"
    GA_Brdcst(buffer.data, len(buffer)*buffer.itemsize, root)

def check_handle(int g_a, char *message):
    """Checks that the array handle g_a is valid.
    
    If not, calls ga.error withe the provided string.

    This operation is local.

    """
    GA_Check_handle(g_a, message)

def cluster_nnodes():
    """Returns the total number of nodes that the program is running on.

    On SMP architectures, this will be less than or equal to the total number
    of processors.

    This is a  local operation.

    """
    return GA_Cluster_nnodes()

def cluster_nodeid(int proc=-1):
    """Returns the node ID of this process or the given process.

    On SMP architectures with more than one processor per node, several
    processes may return the same node id.

    This is a local operation.

    Keyword arguments:
    proc -- process ID to lookup

    """
    if proc >= 0:
        return GA_Cluster_proc_nodeid(proc)
    return GA_Cluster_nodeid()

def cluster_nprocs(int inode):
    """Returns the number of processors available on the given node.

    This is a local operation.

    """
    return GA_Cluster_nprocs(inode)

def cluster_procid(int inode, int iproc):
    """Returns the proc ID associated with node inode and local proc ID iproc.

    If node inode has N processors, then the value of iproc lies between 0 and
    N-1.

    This is a  local operation. 

    """
    return GA_Cluster_procid(inode, iproc)

def compare_distr(int g_a, int g_b):
    """Compares the distributions of two global arrays.

    This is a collective operation.

    Returns:
    True if distributions are identical and False when they are not

    """
    if GA_Compare_distr(g_a, g_b) == 0:
        return True
    return False

cdef void* _convert_multiplier(int gtype, value,
        int *iv, long *lv, long long *llv, float *fv, double *dv):
    if gtype == C_INT:
        if value is not None:
            iv[0] = value
        return iv
    elif gtype == C_LONG:
        if value is not None:
            lv[0] = value
        return lv
    elif gtype == C_LONGLONG:
        if value is not None:
            llv[0] = value
        return llv
    elif gtype == C_FLT:
        if value is not None:
            fv[0] = value
        return fv
    elif gtype == C_DBL:
        if value is not None:
            dv[0] = value
        return dv
    else:
        raise TypeError, "type of g_a not recognized"

def copy(int g_a, int g_b, alo=None, ahi=None, blo=None, bhi=None,
        bint trans=False):
    """Copies elements from array g_a into array g_b.

    For the operation over the entire arrays, the arrays must be the same
    type, shape, and identically aligned.  No transpose is allowed in this
    case.

    For patch operations, the patches of arrays may be of different shapes but
    must have the same number of elements. Patches must be nonoverlapping (if
    g_a=g_b).  Transposes are allowed for patch operations.

    This is a collective operation. 

    Positional arguments:
    g_a   -- the array handle copying from
    g_b   -- the array handle copying to

    Keyword arguments:
    alo   -- lower bound patch coordinates of g_a, inclusive
    ahi   -- higher bound patch coordinates of g_a, inclusive
    blo   -- lower bound patch coordinates of g_b, inclusive
    bhi   -- higher bound patch coordinates of g_b, inclusive
    trans -- whether the transpose operator should be applied True=applied
             
    """
    cdef np.ndarray[np.int64_t, ndim=1] alo_nd, ahi_nd
    cdef np.ndarray[np.int64_t, ndim=1] blo_nd, bhi_nd
    cdef char trans_c
    if alo is None and ahi is None and blo is None and bhi is None:
        GA_Copy(g_a, g_b)
    else:
        alo_nd,ahi_nd = _lohi(g_a,alo,ahi)
        blo_nd,bhi_nd = _lohi(g_b,blo,bhi)
        if trans:
            trans_c = "T"
        else:
            trans_c = "N"
        NGA_Copy_patch64(trans_c,
                g_a, <int64_t*>alo_nd.data, <int64_t*>ahi_nd.data,
                g_b, <int64_t*>blo_nd.data, <int64_t*>bhi_nd.data)

def create(int gtype, dims, char *name="", chunk=None, int pgroup=-1):
    """Creates an n-dimensional array using the regular distribution model.

    The array can be distributed evenly or not. The control over the
    distribution is accomplished by specifying chunk (block) size for all or
    some of array dimensions. For example, for a 2-dimensional array, setting
    chunk[0]=dim[0] gives distribution by vertical strips (chunk[0]*dims[0]);
    setting chunk[1]=dim[1] gives distribution by horizontal strips
    (chunk[1]*dims[1]). Actual chunks will be modified so that they are at
    least the size of the minimum and each process has either zero or one
    chunk. Specifying chunk[i] as <1 will cause that dimension to be
    distributed evenly.

    As a convenience, when chunk is omitted or None, the entire array is
    distributed evenly.

    Positional arguments:
    gtype  -- the type of the array
    dims   -- array-like shape of the array

    Keyword arguments:
    name   -- the name of the array
    chunk  -- see above
    pgroup -- create array only as part of this processor group

    Returns:
    a non-zero array handle means the call was succesful.

    This is a collective operation. 

    """
    cdef np.ndarray[np.int64_t, ndim=1] dims_nd, chunk_nd=None
    dims_nd = np.asarray(dims, dtype=np.int64)
    if chunk:
        chunk_nd = np.asarray(chunk, dtype=np.int64)
        return NGA_Create_config64(gtype, len(dims_nd), <int64_t*>dims_nd.data,
                name, <int64_t*>chunk_nd.data, pgroup)
    else:
        return NGA_Create_config64(gtype, len(dims_nd), <int64_t*>dims_nd.data,
                name, NULL, pgroup)

def create_ghosts(int gtype, dims, width, char *name="", chunk=None,
        int pgroup=-1):
    """Creates an array with a layer of ghost cells around the visible data.

    The array can be distributed evenly or not evenly. The control over the
    distribution is accomplished by specifying chunk (block) size for all or
    some of the array dimensions. For example, for a 2-dimensional array,
    setting chunk(1)=dim(1) gives distribution by vertical strips
    (chunk(1)*dims(1)); setting chunk(2)=dim(2) gives distribution by
    horizontal strips (chunk(2)*dims(2)). Actual chunks will be modified so
    that they are at least the size of the minimum and each process has either
    zero or one chunk. Specifying chunk(i) as <1 will cause that dimension
    (i-th) to be distributed evenly. The  width of the ghost cell layer in
    each dimension is specified using the array width().  The local data of
    the global array residing on each processor will have a layer width[n]
    ghosts cells wide on either side of the visible data along the dimension
    n.

    Positional arguments:
    gtype  -- the type of the array
    dims   -- array-like shape of the array
    width  -- array-like of ghost cell widths

    Keyword arguments:
    name   -- the name of the array
    chunk  -- see above
    pgroup -- create array only as part of this processor group

    Returns:
    a non-zero array handle means the call was successful.

    This is a collective operation. 

    """
    cdef np.ndarray[np.int64_t, ndim=1] dims_nd, chunk_nd, width_nd
    dims_nd = np.asarray(dims, dtype=np.int64)
    width_nd = np.asarray(width, dtype=np.int64)
    if chunk:
        chunk_nd = np.asarray(chunk, dtype=np.int64)
        return NGA_Create_ghosts_config64(gtype, len(dims_nd),
                <int64_t*>dims_nd.data, <int64_t*>width_nd.data, name,
                <int64_t*>chunk_nd.data, pgroup)
    else:
        return NGA_Create_ghosts_config64(gtype, len(dims_nd),
                <int64_t*>dims_nd.data, <int64_t*>width_nd.data, name,
                NULL, pgroup)

def create_handle():
    """Returns a global array handle that can be used to create a new array.
    
    The sequence of operations is to begin with a call to ga.create_handle to
    get a new array handle. The attributes of the array, such as dimension,
    size, type, etc. can then be set using successive calls to the ga.set_xxx
    subroutines. When all array attributes have been set, the ga.allocate
    subroutine is called and the global array is actually created and memory
    for it is allocated.

    This is a collective operation.

    """
    return GA_Create_handle()

def create_irreg(int gtype, dims, block, map, char *name="", int pgroup=-1):
    """Creates an array by following the user-specified distribution.

    The distribution is specified as a Cartesian product of distributions for
    each dimension. The array indices start at 0. For example, the following
    figure demonstrates distribution of a 2-dimensional array 8x10 on 6 (or
    more) processors. nblock[2]=[3,2], the size of map array is s=5 and array
    map contains the following elements map=[0,2,6, 0, 5]. The distribution is
    nonuniform because, P1 and P4 get 20 elements each and processors
    P0,P2,P3, and P5 only 10 elements each.

    This is a collective operation.

    Positional arguments:
    gtype  -- the type of the array
    dims   -- array-like shape of the array
    block  -- array-like number of blocks each dimension is divided into
    map    -- array-like starting index for each block; len(map) == sum of all
              elements of nblock array

    Keyword arguments:
    name   -- the name of the array
    pgroup -- create array only as part of this processor group
    
    Returns:
    integer handle representing the array; a non-zero value indicates success

    """
    cdef np.ndarray[np.int64_t, ndim=1] dims_nd, block_nd, map_nd
    dims_nd = np.asarray(dims, dtype=np.int64)
    block_nd = np.asarray(block, dtype=np.int64)
    map_nd = np.asarray(map, dtype=np.int64)
    return NGA_Create_irreg_config64(gtype, len(dims_nd),
            <int64_t*>dims_nd.data, name,
            <int64_t*>block_nd.data, <int64_t*>map_nd.data, pgroup)

def create_ghosts_irreg(int gtype, dims, width, block, map, char *name="",
        int pgroup=-1):
    """Creates an array with a layer of ghost cells around the visible data.

    The distribution is specified as a Cartesian product of distributions for
    each dimension. For example, the following figure demonstrates
    distribution of a 2-dimensional array 8x10 on 6 (or more) processors.
    nblock(2)=[3,2], the size of map array is s=5 and array map contains the
    following elements map=[1,3,7, 1, 6]. The distribution is nonuniform
    because, P1 and P4 get 20 elements each and processors P0,P2,P3, and P5
    only 10 elements each. 

    The array width[] is used to control the width of the ghost cell boundary
    around the visible data on each processor. The local data of the global
    array residing on each processor will have a layer width[n] ghosts cells
    wide on either side of the visible data along the dimension n.

    This is a collective operation. 

    Positional arguments:
    gtype  -- the type of the array
    dims   -- array-like shape of the array
    width  -- array-like of ghost cell widths
    block  -- array-like number of blocks each dimension is divided into
    map    -- array-like starting index for each block; len(map) == sum of all
              elements of nblock array

    Keyword arguments:
    name   -- the name of the array
    pgroup -- create array only as part of this processor group
    
    Returns:
    a non-zero array handle means the call was succesful

    """
    cdef np.ndarray[np.int64_t, ndim=1] dims_nd, width_nd, block_nd, map_nd
    dims_nd = np.asarray(dims, dtype=np.int64)
    width_nd = np.asarray(width, dtype=np.int64)
    block_nd = np.asarray(block, dtype=np.int64)
    map_nd = np.asarray(map, dtype=np.int64)
    return NGA_Create_ghosts_irreg_config64(gtype, len(dims_nd),
            <int64_t*>dims_nd.data, <int64_t*>width_nd.data, name,
            <int64_t*>block_nd.data, <int64_t*>map_nd.data, pgroup)

def create_mutexes(int number):
    """Creates a set containing the number of mutexes.

    Mutex is a simple synchronization object used to protect Critical
    Sections. Only one set of mutexes can exist at a time. Array of mutexes
    can be created and destroyed as many times as needed.

    Mutexes are numbered: 0, ..., number -1.

    This is a collective operation. 

    Positional arguments:
    number -- the number of mutexes to create

    Returns:
    True on success, False on failure

    """
    if GA_Create_mutexes(number) == 0:
        return True
    return False

def destroy(int g_a):
    """Deallocates the array and frees any associated resources.

    This is a collective operation.

    """
    GA_Destroy(g_a)

def destroy_mutexes():
    """Destroys the set of mutexes created with ga_create_mutexes.
    
    Returns:
    True if the operation succeeded; False when failed

    This is a collective operation. 

    """
    if GA_Destroy_mutexes() == 0:
        return True
    return False

def diag(int g_a, int g_s, int g_v, evalues=None):
    """Solve the generalized eigen-value problem.

    The input matrices are not overwritten or destroyed.
    
    Positional arguments:
    g_a -- the array handle of the matrix to diagonalize
    g_s -- the array handle of the metric
    g_v -- the array handle to return evecs

    Returns:
    All eigen-values as an ndarray in ascending order.

    This is a collective operation. 

    """
    if evalues is None:
        gtype,dims = inquire(g_a)
        evalues = np.ndarray((dims[0]), dtype=_to_dtype(gtype))
    else:
        evalues = np.asarray(evalues)
    GA_Diag(g_a, g_s, g_v, <void*>evalues.data)
    return evalues

def diag_reuse(int control, int g_a, int g_s, int g_v, evalues=None):
    """Solve the generalized eigen-value problem.

    Recommended for REPEATED calls if g_s is unchanged.
    The input matrices are not overwritten or destroyed.
    
    Positional arguments:
    control --  0 indicates first call to the eigensolver
               >0 consecutive calls (reuses factored g_s)
               <0 only erases factorized g_s; g_v and eval unchanged
                  (should be called after previous use if another
                  eigenproblem, i.e., different g_a and g_s, is to
                  be solved) 
    g_a     -- the array handle of the matrix to diagonalize
    g_s     -- the array handle of the metric
    g_v     -- the array handle to return evecs

    Returns:
    All eigen-values as an ndarray in ascending order.

    This is a collective operation. 

    """
    if evalues is None:
        gtype,dims = inquire(g_a)
        evalues = np.ndarray((dims[0]), dtype=_to_dtype(gtype))
    else:
        evalues = np.asarray(evalues)
    GA_Diag_reuse(control, g_a, g_s, g_v, <void*>evalues.data)
    return evalues

def diag_std(int g_a, int g_v, evalues=None):
    """Solve the standard (non-generalized) eigenvalue problem.

    The input matrix is neither overwritten nor destroyed.
    
    Positional arguments:
    g_a -- the array handle of the matrix to diagonalize
    g_v -- the array handle to return evecs

    Returns:
    all eigenvectors via the g_v global array, and eigenvalues as an ndarray
    in ascending order

    This is a collective operation. 

    """
    if evalues is None:
        gtype,dims = inquire(g_a)
        evalues = np.ndarray((dims[0]), dtype=_to_dtype(gtype))
    else:
        evalues = np.asarray(evalues)
    GA_Diag_std(g_a, g_v, <void*>evalues.data)
    return evalues

def distribution(int g_a, int iproc=-1):
    """Return the distribution given to iproc.

    If iproc is not specified, then ga.nodeid() is used.  The range is
    returned as -1 for lo and -2 for hi if no elements are owned by
    iproc.
    
    """
    cdef int ndim = GA_Ndim(g_a)
    cdef np.ndarray[np.int64_t, ndim=1] lo = np.zeros((ndim), dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] hi = np.zeros((ndim), dtype=np.int64)
    if iproc < 0:
        iproc = GA_Nodeid()
    NGA_Distribution64(g_a, iproc, <int64_t*>lo.data, <int64_t*>hi.data)
    return lo,hi

def dot(int g_a, int g_b, alo=None, ahi=None, blo=None, bhi=None,
        bint ta=False, bint tb=False):
    """Computes the element-wise dot product of two arrays.

    Arrays must be of the same type and same number of elements.
    Patch operation allows for possibly transposed patches.

    This is a collective operation.

    Positional arguments:
    g_a -- the array handle
    g_b -- the array handle

    Keyword arguments:
    alo -- lower bound patch coordinates of g_a, inclusive
    ahi -- higher bound patch coordinates of g_a, inclusive
    blo -- lower bound patch coordinates of g_b, inclusive
    bhi -- higher bound patch coordinates of g_b, inclusive
    ta  -- whether the transpose operator should be applied to g_a True=applied
    tb  -- whether the transpose operator should be applied to g_b True=applied

    Returns:
    SUM_ij a(i,j)*b(i,j)

    """
    cdef np.ndarray[np.int64_t, ndim=1] alo_nd, ahi_nd
    cdef np.ndarray[np.int64_t, ndim=1] blo_nd, bhi_nd
    cdef char ta_c, tb_c
    cdef int gtype=inquire_type(g_a)
    if alo is None and ahi is None and blo is None and bhi is None:
        if gtype == C_INT:
            return GA_Idot(g_a, g_b)
        elif gtype == C_LONG:
            return GA_Ldot(g_a, g_b)
        elif gtype == C_LONGLONG:
            return GA_Lldot(g_a, g_b)
        elif gtype == C_FLT:
            return GA_Fdot(g_a, g_b)
        elif gtype == C_DBL:
            return GA_Ddot(g_a, g_b)
        else:
            raise TypeError
    else:
        alo_nd,ahi_nd = _lohi(g_a,alo,ahi)
        blo_nd,bhi_nd = _lohi(g_b,blo,bhi)
        if ta:
            ta_c = "T"
        else:
            ta_c = "N"
        if tb:
            tb_c = "T"
        else:
            tb_c = "N"
        if gtype == C_INT:
            return NGA_Idot_patch64(
                    g_a, ta_c, <int64_t*>alo_nd.data, <int64_t*>ahi_nd.data,
                    g_b, tb_c, <int64_t*>blo_nd.data, <int64_t*>bhi_nd.data)
        elif gtype == C_LONG:
            return NGA_Ldot_patch64(
                    g_a, ta_c, <int64_t*>alo_nd.data, <int64_t*>ahi_nd.data,
                    g_b, tb_c, <int64_t*>blo_nd.data, <int64_t*>bhi_nd.data)
        elif gtype == C_LONGLONG:
            return NGA_Lldot_patch64(
                    g_a, ta_c, <int64_t*>alo_nd.data, <int64_t*>ahi_nd.data,
                    g_b, tb_c, <int64_t*>blo_nd.data, <int64_t*>bhi_nd.data)
        elif gtype == C_FLT:
            return NGA_Fdot_patch64(
                    g_a, ta_c, <int64_t*>alo_nd.data, <int64_t*>ahi_nd.data,
                    g_b, tb_c, <int64_t*>blo_nd.data, <int64_t*>bhi_nd.data)
        elif gtype == C_DBL:
            return NGA_Ddot_patch64(
                    g_a, ta_c, <int64_t*>alo_nd.data, <int64_t*>ahi_nd.data,
                    g_b, tb_c, <int64_t*>blo_nd.data, <int64_t*>bhi_nd.data)
        else:
            raise TypeError
    
def duplicate(int g_a, char *name=""):
    """Creates a new array by applying all the properties of another existing
    array.
    
    Positional arguments:
    g_a -- the array handle

    Keyword arguments:
    name -- the new name of the created array

    Returns:
    a non-zero array handle means the call was succesful.

    This is a collective operation. 

    """
    return GA_Duplicate(g_a, name)

def elem_divide(int g_a, int g_b, int g_c, alo=None, ahi=None, blo=None,
        bhi=None, clo=None, chi=None):
    """Computes the element-wise quotient of the two arrays.

    Arrays or array patches must be of the same types and same number of
    elements. For two-dimensional arrays:

                c(i, j)  = a(i,j)/b(i,j)

    The result (c) may replace one of the input arrays (a/b).
    If one of the elements of array g_b is zero, the quotient for the element
    of g_c will be set to GA_NEGATIVE_INFINITY. 

    This is a collective operation. 

    Positional arguments:
    g_a    -- the array handle
    g_b    -- the array handle
    g_c    -- the array handle

    Keyword arguments:
    alo   -- lower bound patch coordinates of g_a, inclusive
    ahi   -- higher bound patch coordinates of g_a, inclusive
    blo   -- lower bound patch coordinates of g_b, inclusive
    bhi   -- higher bound patch coordinates of g_b, inclusive
    clo   -- lower bound patch coordinates of g_c, inclusive
    chi   -- higher bound patch coordinates of g_c, inclusive

    """
    cdef np.ndarray[np.int64_t, ndim=1] alo_nd, ahi_nd
    cdef np.ndarray[np.int64_t, ndim=1] blo_nd, bhi_nd
    cdef np.ndarray[np.int64_t, ndim=1] clo_nd, chi_nd
    if (alo is None and ahi is None
            and blo is None and bhi is None
            and clo is None and chi is None):
        GA_Elem_divide(g_a, g_b, g_c)
    else:
        alo_nd,ahi_nd = _lohi(g_a,alo,ahi)
        blo_nd,bhi_nd = _lohi(g_b,blo,bhi)
        clo_nd,chi_nd = _lohi(g_c,clo,chi)
        GA_Elem_divide_patch64(
                g_a, <int64_t*>alo_nd.data, <int64_t*>ahi_nd.data,
                g_b, <int64_t*>blo_nd.data, <int64_t*>bhi_nd.data,
                g_c, <int64_t*>clo_nd.data, <int64_t*>chi_nd.data)

def elem_maximum(int g_a, int g_b, int g_c, alo=None, ahi=None, blo=None,
        bhi=None, clo=None, chi=None):
    """Computes the element-wise maximum of the two arrays.

    Arrays or array patches must be of the same types and same number of
    elements. For two-dimensional arrays:

        c(i, j)  = max(a(i,j),b(i,j))

    If the data type is complex, then
        c(i, j).real = max{ |a(i,j)|, |b(i,j)|} while c(i,j).image = 0
    The result (c) may replace one of the input arrays (a/b).

    This is a collective operation. 

    Positional arguments:
    g_a    -- the array handle
    g_b    -- the array handle
    g_c    -- the array handle

    Keyword arguments:
    alo   -- lower bound patch coordinates of g_a, inclusive
    ahi   -- higher bound patch coordinates of g_a, inclusive
    blo   -- lower bound patch coordinates of g_b, inclusive
    bhi   -- higher bound patch coordinates of g_b, inclusive
    clo   -- lower bound patch coordinates of g_c, inclusive
    chi   -- higher bound patch coordinates of g_c, inclusive

    """
    cdef np.ndarray[np.int64_t, ndim=1] alo_nd, ahi_nd
    cdef np.ndarray[np.int64_t, ndim=1] blo_nd, bhi_nd
    cdef np.ndarray[np.int64_t, ndim=1] clo_nd, chi_nd
    if (alo is None and ahi is None
            and blo is None and bhi is None
            and clo is None and chi is None):
        GA_Elem_maximum(g_a, g_b, g_c)
    else:
        alo_nd,ahi_nd = _lohi(g_a,alo,ahi)
        blo_nd,bhi_nd = _lohi(g_b,blo,bhi)
        clo_nd,chi_nd = _lohi(g_c,clo,chi)
        GA_Elem_maximum_patch64(
                g_a, <int64_t*>alo_nd.data, <int64_t*>ahi_nd.data,
                g_b, <int64_t*>blo_nd.data, <int64_t*>bhi_nd.data,
                g_c, <int64_t*>clo_nd.data, <int64_t*>chi_nd.data)

def elem_minimum(int g_a, int g_b, int g_c, alo=None, ahi=None, blo=None,
        bhi=None, clo=None, chi=None):
    """Computes the element-wise minimum of the two arrays.

    Arrays or array patches must be of the same types and same number of
    elements. For two-dimensional arrays:

        c(i, j)  = min(a(i,j),b(i,j))

    If the data type is complex, then
        c(i, j).real = min{ |a(i,j)|, |b(i,j)|} while c(i,j).image = 0
    The result (c) may replace one of the input arrays (a/b).

    This is a collective operation. 

    Positional arguments:
    g_a    -- the array handle
    g_b    -- the array handle
    g_c    -- the array handle

    Keyword arguments:
    alo   -- lower bound patch coordinates of g_a, inclusive
    ahi   -- higher bound patch coordinates of g_a, inclusive
    blo   -- lower bound patch coordinates of g_b, inclusive
    bhi   -- higher bound patch coordinates of g_b, inclusive
    clo   -- lower bound patch coordinates of g_c, inclusive
    chi   -- higher bound patch coordinates of g_c, inclusive

    """
    cdef np.ndarray[np.int64_t, ndim=1] alo_nd, ahi_nd
    cdef np.ndarray[np.int64_t, ndim=1] blo_nd, bhi_nd
    cdef np.ndarray[np.int64_t, ndim=1] clo_nd, chi_nd
    if (alo is None and ahi is None
            and blo is None and bhi is None
            and clo is None and chi is None):
        GA_Elem_minimum(g_a, g_b, g_c)
    else:
        alo_nd,ahi_nd = _lohi(g_a,alo,ahi)
        blo_nd,bhi_nd = _lohi(g_b,blo,bhi)
        clo_nd,chi_nd = _lohi(g_c,clo,chi)
        GA_Elem_minimum_patch64(
                g_a, <int64_t*>alo_nd.data, <int64_t*>ahi_nd.data,
                g_b, <int64_t*>blo_nd.data, <int64_t*>bhi_nd.data,
                g_c, <int64_t*>clo_nd.data, <int64_t*>chi_nd.data)

def elem_multiply(int g_a, int g_b, int g_c, alo=None, ahi=None, blo=None,
        bhi=None, clo=None, chi=None):
    """Computes the element-wise product of the two arrays.

    Arrays or array patches must be of the same types and same number of
    elements. For two-dimensional arrays:

                c(i, j)  = a(i,j)*b(i,j)

    The result (c) may replace one of the input arrays (a/b).

    This is a collective operation. 

    Positional arguments:
    g_a    -- the array handle
    g_b    -- the array handle
    g_c    -- the array handle

    Keyword arguments:
    alo   -- lower bound patch coordinates of g_a, inclusive
    ahi   -- higher bound patch coordinates of g_a, inclusive
    blo   -- lower bound patch coordinates of g_b, inclusive
    bhi   -- higher bound patch coordinates of g_b, inclusive
    clo   -- lower bound patch coordinates of g_c, inclusive
    chi   -- higher bound patch coordinates of g_c, inclusive

    """
    cdef np.ndarray[np.int64_t, ndim=1] alo_nd, ahi_nd
    cdef np.ndarray[np.int64_t, ndim=1] blo_nd, bhi_nd
    cdef np.ndarray[np.int64_t, ndim=1] clo_nd, chi_nd
    if (alo is None and ahi is None
            and blo is None and bhi is None
            and clo is None and chi is None):
        GA_Elem_multiply(g_a, g_b, g_c)
    else:
        alo_nd,ahi_nd = _lohi(g_a,alo,ahi)
        blo_nd,bhi_nd = _lohi(g_b,blo,bhi)
        clo_nd,chi_nd = _lohi(g_c,clo,chi)
        GA_Elem_multiply_patch64(
                g_a, <int64_t*>alo_nd.data, <int64_t*>ahi_nd.data,
                g_b, <int64_t*>blo_nd.data, <int64_t*>bhi_nd.data,
                g_c, <int64_t*>clo_nd.data, <int64_t*>chi_nd.data)

def fence():
    """Blocks the calling process until all the data transfers corresponding
    to GA operations called after ga.init_fence() complete.
    
    For example, since ga.put might return before the data reaches the final
    destination, ga_init_fence and ga_fence allow processes to wait until the
    data tranfer is fully completed:

        ga.init_fence()
        ga.put(g_a, ...)
        ga.fence()

    ga.fence() must be called after ga.init_fence(). A barrier, ga.sync(),
    assures completion of all data transfers and implicitly cancels all
    outstanding ga.init_fence() calls. ga.init_fence() and ga.fence() must be
    used in pairs, multiple calls to ga.fence() require the same number of
    corresponding ga.init_fence() calls. ga.init_fence()/ga_fence() pairs can
    be nested.

    ga.fence() works for multiple GA operations. For example:

        ga.init_fence()
        ga.put(g_a, ...)
        ga.scatter(g_a, ...)
        ga.put(g_b, ...)
        ga.fence()

    The calling process will be blocked until data movements initiated by two
    calls to ga_put and one ga_scatter complete.
    
    """
    GA_Fence()

def fill(int g_a, value, lo=None, hi=None):
    """Assign a single value to all elements in the array or patch.
    
    Positional arguments:
    g_a -- the array handle

    Keyword arguments:
    lo -- lower bound patch coordinates, inclusive
    hi -- higher bound patch coordinates, inclusive
    
    """
    cdef np.ndarray[np.int64_t, ndim=1] lo_nd, hi_nd
    cdef int       ivalue
    cdef long      lvalue
    cdef long long llvalue
    cdef float     fvalue
    cdef double    dvalue
    cdef void     *vvalue
    cdef int gtype=inquire_type(g_a)
    vvalue = _convert_multiplier(gtype, value, &ivalue, &lvalue, &llvalue,
            &fvalue, &dvalue)
    if lo is None and hi is None:
        GA_Fill(g_a, &dvalue)
    else:
        lo_nd,hi_nd = _lohi(g_a,lo,hi)
        NGA_Fill_patch64(
                g_a, <int64_t*>lo_nd.data, <int64_t*>hi_nd.data, vvalue)

def gather(int g_a, subsarray, np.ndarray values=None):
    """Gathers array elements from a global array into a local array.

    subsarray will be converted to an ndarray if it is not one already.  A
    two-dimensional array is allowed so long as its shape is (n,ndim) where n
    is the number of elements to gather and ndim is the number of dimensions
    of the target array.  Also, subsarray must be contiguous.

    For example, if the subsarray were two-dimensional::

        for k in range(n):
            v[k] = g_a[subsarray[k,0],subsarray[k,1],subsarray[k,2]...]

    For example, if the subsarray were one-dimensional::

        for k in range(n):
            base = n*ndim
            v[k] = g_a[subsarray[base+0],subsarray[base+1],subsarray[base+2]...]

    This is a one-sided operation. 

    """
    cdef np.ndarray[np.int64_t, ndim=1] subsarray1_nd = None
    cdef np.ndarray[np.int64_t, ndim=2] subsarray2_nd = None
    cdef int gtype = inquire_type(g_a)
    cdef int ndim = GA_Ndim(g_a)
    cdef int64_t n
    # prepare subsarray
    try:
        subsarray1_nd = np.asarray(subsarray, dtype=np.int64)
        n = len(subsarray1_nd) / ndim
    except ValueError:
        try:
            subsarray2_nd = np.asarray(subsarray, dtype=np.int64)
            n = len(subsarray2_nd) # length of first dimension of subsarray2_nd
        except ValueError:
            raise ValueError, "subsarray must be either 1- or 2-dimensional"
    # prepare values array
    if values is None:
        values = np.ndarray(n, dtype=_to_dtype[gtype])
    else:
        if values.ndim != 1:
            raise ValueError, "values must be one-dimensional"
        if not values.flags['C_CONTIGUOUS']:
            raise ValueError, "values must be contiguous"
        if len(values) < n:
            raise ValueError, "values was not large enough"
    if subsarray1_nd is not None:
        NGA_Gather_flat64(g_a, <void*>values.data,
                <int64_t*>subsarray1_nd.data, n)
    elif subsarray2_nd is not None:
        NGA_Gather_flat64(g_a, <void*>values.data,
                <int64_t*>subsarray2_nd.data, n)
    else:
        raise ValueError, "how did this happen?"
    return values

def gemm(bint ta, bint tb, int64_t m, int64_t n, int64_t k,
        alpha, int g_a, int g_b, beta, int g_c):
    """Performs one of the matrix-matrix operations.
    
    C := alpha*op( A )*op( B ) + beta*C

    where op( X ) is one of

        op( X ) = X   or   op( X ) = X',
        alpha and beta are scalars, and
        A, B and C are matrices, with
        op( A ) an m by k matrix,
        op( B ) a  k by n matrix, and
        C an m by n matrix.

    On entry, ta specifies the form of op( A ) to be used in the
    matrix multiplication as follows:
        ta = 'N' or 'n', op( A ) = A.
        ta = 'T' or 't', op( A ) = A'.

    This is a collective operation. 
    
    """
    raise NotImplementedError

def get(int g_a, lo=None, hi=None, np.ndarray buffer=None):
    """Copies data from global array section to the local array buffer.
    
    The local array is assumed to be have the same number of dimensions as the
    global array. Any detected inconsitencies/errors in the input arguments
    are fatal.

    This is a one-sided operation.

    Positional arguments:
    g_a -- the array handle
    lo  -- a 1D array-like object, or None
    hi  -- a 1D array-like object, or None

    Keyword arguments:
    buffer -- 

    Returns:
    The local array buffer.
    
    """
    cdef np.ndarray[np.int64_t, ndim=1] lo_nd, hi_nd, ld_nd, shape
    cdef int gtype=inquire_type(g_a)
    lo_nd,hi_nd = _lohi(g_a,lo,hi)
    shape = hi_nd-lo_nd+1
    ld_nd = shape[1:]
    if buffer is None:
        buffer = np.ndarray(shape, dtype=_to_dtype[gtype])
    else:
        # TODO perform check for shapes matching
        if buffer.dtype != _to_dtype[gtype]:
            raise ValueError, "buffer is wrong type :: buffer=%s != %s" % (
                    buffer.dtype, _to_dtype[gtype])
    NGA_Get64(g_a, <int64_t*>lo_nd.data, <int64_t*>hi_nd.data,
            <void*>buffer.data, <int64_t*>ld_nd.data)
    return buffer

def gop(X, char *op):
    """Global operation.

    X(1:N) is a vector present on each process. gop 'sums' elements of X
    accross all nodes using the commutative operator op. The result is
    broadcast to all nodes. Supported operations include '+', '*', 'max',
    'min', 'absmax', 'absmin'. The use of lowerecase for operators is
    necessary.

    X must be a contiguous array-like.  X is not guaranteed to be modified
    in-place so use as:

    >>> value = ga.gop((1,2,3), "+")

    This operation is provided only for convenience purposes: it is available
    regardless of the message-passing library that GA is running with.

    This is a collective operation. 

    """
    cdef np.ndarray X_nd = np.asarray(X)
    if not X_nd.flags['C_CONTIGUOUS']:
        raise ValueError, "X must be contiguous"
    if X_nd.dtype == np.intc:
        GA_Igop(<int*>X_nd.data, len(X_nd), op)
    elif X_nd.dtype == np.long:
        GA_Lgop(<long*>X_nd.data, len(X_nd), op)
    elif X_nd.dtype == np.longlong:
        GA_Llgop(<long long*>X_nd.data, len(X_nd), op)
    elif X_nd.dtype == np.single:
        GA_Fgop(<float*>X_nd.data, len(X_nd), op)
    elif X_nd.dtype == np.double:
        GA_Dgop(<double*>X_nd.data, len(X_nd), op)
    else:
        raise TypeError, "type not supported by ga.gop %s" % X_nd.dtype
    return X_nd

def gop_add(X):
    return gop(X, "+")

def gop_multiply(X):
    return gop(X, "*")

def gop_max(X):
    return gop(X, "max")

def gop_min(X):
    return gop(X, "min")

def gop_absmax(X):
    return gop(X, "absmax")

def gop_absmin(X):
    return gop(X, "absmin")

def initialize():
    GA_Initialize()
    GA_Register_stack_memory(_gapy_malloc, _gapy_free)

def inquire(int g_a):
    cdef int gtype
    cdef int ndim = GA_Ndim(g_a)
    cdef np.ndarray[np.int64_t, ndim=1] dims=np.zeros((ndim), dtype=np.int64)
    NGA_Inquire64(g_a, &gtype, &ndim, <int64_t*>dims.data)
    return gtype,dims

cpdef np.ndarray[np.int64_t, ndim=1] inquire_dims(int g_a):
    cdef int gtype
    cdef np.ndarray[np.int64_t, ndim=1] dims
    gtype,dims = inquire(g_a)
    return dims

cpdef int inquire_type(int g_a):
    cdef int gtype
    cdef np.ndarray[np.int64_t, ndim=1] dims
    gtype,dims = inquire(g_a)
    return gtype

def _lohi(int g_a, lo, hi):
    """Utility function which converts and/or prepares a lo/hi combination.

    Functions which take a patch specification can use this to convert the
    given lo and/or hi into ndarrays using numpy.asarray.
    If lo is not given, it is replaced with an array of zeros.
    If hi is not given, it is replaced with the last index in each dimension.

    Positional arguments:
    g_a -- the array handle
    lo  -- a 1D array-like object, or None
    hi  -- a 1D array-like object, or None

    Returns:
    The converted lo and hi ndarrays.

    """
    cdef np.ndarray[np.int64_t, ndim=1] lo_nd, hi_nd
    if lo is None:
        lo_nd = np.zeros((GA_Ndim(g_a)), dtype=np.int64)
    else:
        lo_nd = np.asarray(lo, dtype=np.int64)
    if hi is None:
        hi_nd = inquire_dims(g_a)-1
    else:
        hi_nd = np.asarray(hi, dtype=np.int64)
    return lo_nd,hi_nd

cpdef int ndim(int g_a):
    """Returns the number of dimensions in array represented by the handle g_a.

    This operation is local.
    
    """
    return GA_Ndim(g_a)

cpdef int nnodes():
    """TODO"""
    return GA_Nnodes()

cpdef int nodeid():
    """TODO"""
    return GA_Nodeid()

def randomize(int g_a, val):
    """Fill array with random values in [0,val)."""
    cdef int gtype=inquire_type(g_a)
    cdef int       ival=1
    cdef long      lval=1
    cdef long long llval=1
    cdef float     fval=1.0
    cdef double    dval=1.0
    cdef void     *vval=NULL
    if gtype == C_INT:
        if val:
            ival = val
        vval = &ival
    elif gtype == C_LONG:
        if val:
            lval = val
        vval = &lval
    elif gtype == C_LONGLONG:
        if val:
            llval = val
        vval = &llval
    elif gtype == C_FLT:
        if val:
            fval = val
        vval = &fval
    elif gtype == C_DBL:
        if val:
            dval = val
        vval = &dval
    else:
        raise TypeError, "type of g_a not recognized"
    GA_Randomize(g_a, vval)

def print_stdout(int g_a):
    """Prints an entire array to the standard output."""
    GA_Print(g_a)

def release(int g_a, lo=None, hi=None):
    """TODO"""
    _release_common(g_a, lo, hi, False)

cdef _release_common(int g_a, lo, hi, bint update):
    """TODO"""
    cdef np.ndarray[np.int64_t, ndim=1] lo_nd, hi_nd, lo_dst, hi_dst
    # first things first, if no data is owned, return silently
    lo_dst,hi_dst = distribution(g_a)
    if lo_dst[0] < 0 or hi_dst[0] < 0:
        return
    if lo is not None:
        lo_nd = np.asarray(lo, dtype=np.int64)
    else:
        lo_nd = lo_dst
    if hi is not None:
        hi_nd = np.asarray(hi, dtype=np.int64)
    else:
        hi_nd = hi_dst
    # sanity checks
    if np.sometrue(lo_nd>hi_nd):
        raise ValueError,"lo>hi lo=%s hi=%s"%(lo_nd,hi_nd)
    if np.sometrue(lo_nd<lo_dst):
        raise ValueError,"lo out of bounds lo_dst=%s lo=%s"%(lo_dst,lo_nd)
    if np.sometrue(hi_nd>hi_dst):
        raise ValueError,"hi out of bounds hi_dst=%s hi=%s"%(hi_dst,hi_nd)
    if update:
        NGA_Release_update64(g_a, <int64_t*>lo_nd.data, <int64_t*>hi_nd.data)
    else:
        NGA_Release64(g_a, <int64_t*>lo_nd.data, <int64_t*>hi_nd.data)

def release_update(int g_a, lo=None, hi=None):
    """TODO"""
    _release_common(g_a, lo, hi, True)
