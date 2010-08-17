# keep the ga functions alphabetical since this is going to be a huge file!

from libc.stdlib cimport malloc,free
from gah cimport *
import numpy as np
cimport numpy as np
from numpy cimport NPY_INT, npy_intp

cdef extern from "numpy/arrayobject.h":
    cdef object PyArray_SimpleNewFromData(int nd, npy_intp *dims,
            int typenum, void *data)
    cdef void import_array()

import_array()

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

def abs_value(int g_a, lo=None, hi=None):
    """Take element-wise absolute value of the array."""
    cdef np.ndarray[np.int64_t] lo_nd, hi_nd
    if lo and hi:
        lo_nd = np.asarray(lo, dtype=np.int64)
        hi_nd = np.asarray(hi, dtype=np.int64)
        GA_Abs_value_patch64(g_a, <int64_t*>lo_nd.data, <int64_t*>hi_nd.data)
    else:
        GA_Abs_value(g_a)

def acc(int g_a, lo, hi, buf, alpha=None):
    """Combines data from local array buffer with data in the global array
    section. The local array is assumed to be have the same number of
    dimensions as the global array."""
    cdef int type=inquire_type(g_a)
    cdef np.ndarray[np.int64_t] lo_nd, hi_nd, ld_nd
    cdef int       ialpha=1
    cdef long      lalpha=1
    cdef long long llalpha=1
    cdef float     falpha=1.0
    cdef double    dalpha=1.0
    cdef np.ndarray contiguous
    buf = np.asarray(buf)
    lo_nd = np.asarray(lo, dtype=np.int64)
    hi_nd = np.asarray(hi, dtype=np.int64)
    ld_nd = np.asarray(buf.shape, dtype=np.int64)
    if type == C_INT:
        if buf.dtype != np.int or not buf.flags['C_CONTIGUOUS']:
            contiguous = np.ascontiguousarray(buf, dtype=np.int)
        if alpha:
            ialpha = alpha
        NGA_Acc64(g_a, <int64_t*>lo_nd.data, <int64_t*>hi_nd.data,
                <int*>contiguous.data, (<int64_t*>ld_nd.data)+1, &ialpha)
    elif type == C_LONG:
        if buf.dtype != np.long or not buf.flags['C_CONTIGUOUS']:
            contiguous = np.ascontiguousarray(buf, dtype=np.long)
        if alpha:
            lalpha = alpha
        NGA_Acc64(g_a, <int64_t*>lo_nd.data, <int64_t*>hi_nd.data,
                <long*>contiguous.data, (<int64_t*>ld_nd.data)+1, &lalpha)
    elif type == C_LONGLONG:
        if buf.dtype != np.longlong or not buf.flags['C_CONTIGUOUS']:
            contiguous = np.ascontiguousarray(buf, dtype=np.longlong)
        if alpha:
            lalpha = alpha
        NGA_Acc64(g_a, <int64_t*>lo_nd.data, <int64_t*>hi_nd.data,
                <long long*>contiguous.data, (<int64_t*>ld_nd.data)+1, &lalpha)
    elif type == C_FLT:
        if buf.dtype != np.float or not buf.flags['C_CONTIGUOUS']:
            contiguous = np.ascontiguousarray(buf, dtype=np.float)
        if alpha:
            lalpha = alpha
        NGA_Acc64(g_a, <int64_t*>lo_nd.data, <int64_t*>hi_nd.data,
                <float*>contiguous.data, (<int64_t*>ld_nd.data)+1, &lalpha)
    elif type == C_DBL:
        if buf.dtype != np.double or not buf.flags['C_CONTIGUOUS']:
            contiguous = np.ascontiguousarray(buf, dtype=np.double)
        if alpha:
            lalpha = alpha
        NGA_Acc64(g_a, <int64_t*>lo_nd.data, <int64_t*>hi_nd.data,
                <double*>contiguous.data, (<int64_t*>ld_nd.data)+1, &lalpha)
    else:
        raise TypeError, "type of g_a not recognized"

def access(int g_a, lo, hi):
    """Provides access to the specified patch of a global array."""
    cdef np.ndarray[np.int64_t] lo_nd, hi_nd, ld_nd
    cdef int type=inquire_type(g_a)
    cdef void *ptr
    cdef npy_intp dims
    lo_nd = np.asarray(lo)
    hi_nd = np.asarray(hi)
    ld_nd = np.ndarray(len(lo)-1, dtype=np.int64)
    NGA_Access64(g_a, <int64_t*>lo_nd.data, <int64_t*>hi_nd.data, &ptr,
            <int64_t*>ld_nd.data);
    if type == C_INT:
        return PyArray_SimpleNewFromData(len(lo), &dims, NPY_INT, ptr)
        pass
    elif type == C_LONG:
        pass
    elif type == C_LONGLONG:
        pass
    elif type == C_FLT:
        pass
    elif type == C_DBL:
        pass

def create(int type, dims, char *name, chunk=None):
    """Creates an ndim-dimensional array using the regular distribution model
    and returns integer handle representing the array."""
    cdef np.ndarray[np.int64_t] dims_nd, chunk_nd=None
    dims_nd = np.asarray(dims, dtype=np.int64)
    if chunk:
        chunk_nd = np.asarray(chunk, dtype=np.int64)
        return NGA_Create64(type, len(dims_nd), <int64_t*>dims_nd.data,
                name, <int64_t*>chunk_nd.data)
    else:
        return NGA_Create64(type, len(dims_nd), <int64_t*>dims_nd.data,
                name, NULL)

def initialize():
    GA_Initialize()

def inquire_type(int g_a):
    cdef int type, ndim
    cdef int64_t *dims
    ndim = GA_Ndim(g_a)
    dims = <int64_t*>malloc(sizeof(int64_t) * ndim)
    NGA_Inquire64(g_a, &type, &ndim, dims)
    free(dims)
    return type
