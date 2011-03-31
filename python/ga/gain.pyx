import math
import sys

from mpi4py import MPI
import ga
import util

import numpy as np
cimport numpy as np
import numpy.core.umath as umath

# because it's just useful to have around
me = ga.nodeid()
nproc = ga.nnodes()

gatypes = {
np.dtype(np.int32): ga.C_INT,
np.dtype(np.int64): ga.C_LONG,
np.dtype(np.float32): ga.C_FLOAT,
np.dtype(np.float64): ga.C_DBL,
np.dtype(np.complex64): ga.C_SCPL,
np.dtype(np.complex128): ga.C_DCPL,
}

def _lohi_slice(lo, hi):
    return map(lambda x,y: slice(x,y), lo, hi)

class ndarray(object):
    """ndarray(shape, dtype=float, buffer=None, offset=0,
            strides=None, order=None)

    An array object represents a multidimensional, homogeneous array
    of fixed-size items.  An associated data-type object describes the
    format of each element in the array (its byte-order, how many bytes it
    occupies in memory, whether it is an integer, a floating point number,
    or something else, etc.)

    Arrays should be constructed using `array`, `zeros` or `empty` (refer
    to the See Also section below).  The parameters given here refer to
    a low-level method (`ndarray(...)`) for instantiating an array.

    For more information, refer to the `numpy` module and examine the
    the methods and attributes of an array.

    Parameters
    ----------
    (for the __new__ method; see Notes below)

    shape : tuple of ints
        Shape of created array.
    dtype : data-type, optional
        Any object that can be interpreted as a numpy data type.
    buffer : object exposing buffer interface, optional
        Used to fill the array with data.
    offset : int, optional
        Offset of array data in buffer.
    strides : tuple of ints, optional
        Strides of data in memory.
    order : {'C', 'F'}, optional
        Row-major or column-major order.

    Parameters added for Global Arrays
    ----------------------------------
    base : ndarray
        Should be a "gainarray".  Used during view creation so that a new
        Global Array is not created and other attributes from base are copied
        into the view ndarray.

    Attributes
    ----------
    T : ndarray
        Transpose of the array.
    data : buffer
        The array's elements, in memory.
    dtype : dtype object
        Describes the format of the elements in the array.
    flags : dict
        Dictionary containing information related to memory use, e.g.,
        'C_CONTIGUOUS', 'OWNDATA', 'WRITEABLE', etc.
    flat : numpy.flatiter object
        Flattened version of the array as an iterator.  The iterator
        allows assignments, e.g., ``x.flat = 3`` (See `ndarray.flat` for
        assignment examples; TODO).
    imag : ndarray
        Imaginary part of the array.
    real : ndarray
        Real part of the array.
    size : int
        Number of elements in the array.
    itemsize : int
        The memory use of each array element in bytes.
    nbytes : int
        The total number of bytes required to store the array data,
        i.e., ``itemsize * size``.
    ndim : int
        The array's number of dimensions.
    shape : tuple of ints
        Shape of the array.
    strides : tuple of ints
        The step-size required to move from one element to the next in
        memory. For example, a contiguous ``(3, 4)`` array of type
        ``int16`` in C-order has strides ``(8, 2)``.  This implies that
        to move from element to element in memory requires jumps of 2 bytes.
        To move from row-to-row, one needs to jump 8 bytes at a time
        (``2 * 4``).
    ctypes : ctypes object
        Class containing properties of the array needed for interaction
        with ctypes.
    base : ndarray
        If the array is a view into another array, that array is its `base`
        (unless that array is also a view).  The `base` array is where the
        array data is actually stored.

    Attributes added for Global Arrays
    ----------------------------------
    handle : int
        The Global Arrays handle.
    global_slice: tuple of integers and/or slice objects
        Represents the slice to take from this ndarray. global_slice is
        calculated first based on the shape of the array, then as slices are
        taken from it, slice arithmetic is performed. When an ndarray is
        accessed or converted to an ndarray, the global_slice is used to turn
        the ndarray into its correct shape/strides before returning to the
        caller.

    See Also
    --------
    array : Construct an array.
    zeros : Create an array, each element of which is zero.
    empty : Create an array, but leave its allocated memory unchanged (i.e.,
            it contains "garbage").
    dtype : Create a data-type.

    Notes
    -----
    There are two modes of creating an array using ``__new__``:

    1. If `buffer` is None, then only `shape`, `dtype`, and `order`
       are used.
    2. If `buffer` is an object exposing the buffer interface, then
       all keywords are interpreted.

    No ``__init__`` method is needed because the array is fully initialized
    after the ``__new__`` method.

    Examples
    --------
    These examples illustrate the low-level `ndarray` constructor.  Refer
    to the `See Also` section above for easier ways of constructing an
    ndarray.

    First mode, `buffer` is None:

    >>> np.ndarray(shape=(2,2), dtype=float, order='F')
    array([[ -1.13698227e+002,   4.25087011e-303],
           [  2.88528414e-306,   3.27025015e-309]])

    Second mode:

    >>> np.ndarray((2,), buffer=np.array([1,2,3]),
    ...            offset=np.int_().itemsize,
    ...            dtype=int) # offset = 1*itemsize, i.e. skip first element
    array([2, 3])

    """
    def __init__(self, shape, dtype=float, buffer=None, offset=0,
            strides=None, order=None, base=None):
        self._shape = shape
        self._dtype = np.dtype(dtype)
        self._order = order
        self._base = base
        if order is not None:
            raise NotImplementedError, "order parameter not supported"
        if base is None:
            self.handle = ga.create(gatypes[np.dtype(dtype)], shape)
            print_sync("created: handle=%s type=%s shape=%s distribution=%s"%(
                self.handle, self._dtype, self._shape,
                str(ga.distribution(self.handle))))
            if buffer is not None:
                local = ga.access(self.handle)
                if local is not None:
                    lo,hi = ga.distribution(self.handle)
                    a = np.ndarray(shape, dtype, buffer, offset, strides, order)
                    a = a[_lohi_slice(lo,hi)]
                    local[:] = a
                    ga.release_update(self.handle)
            self.global_slice = map(lambda x:slice(0,x,1), shape)
            self._strides = [self.itemsize]
            for size in shape[-1:0:-1]:
                self._strides = [size*self._strides[0]] + self._strides
        else:
            self.handle = base.handle
            self.global_slice = base.global_slice
            self._strides = strides
            print_debug("![%d] view: hndl=%s typ=%s shp=%s dstrbtn=%s"%(
                    me, self.handle, self._dtype, self._shape,
                    str(ga.distribution(self.handle))))

    def __del__(self):
        if self.base is None:
            if ga.initialized():
                print_sync("deleting %s %s" % (self.shape, self.handle))
                ga.destroy(self.handle)

    ################################################################
    ### ndarray methods added for Global Arrays
    ################################################################
    def owns(self):
        lo,hi = ga.distribution(self.handle)
        return np.all(hi>=0)

    def access(self):
        """Access the local array. Return None if no data is owned."""
        if self.owns():
            lo,hi = ga.distribution(self.handle)
            access_slice = None
            try:
                access_slice = util.access_slice(self.global_slice, lo, hi)
            except IndexError:
                pass
            if access_slice:
                a = ga.access(self.handle)
                print_sync("ndarray.access(%s) shape=%s" % (
                        self.handle, a.shape))
                ret = a[access_slice]
                print_debug("![%d] a[access_slice].shape=%s" % (me,ret.shape))
                return ret
        print_sync("ndarray.access None")
        return None

    def get(self):
        """Get remote copy of ndarray based on current global_slice."""
        # We must translate global_slice into a strided get
        print_debug("![%d] inside gainarray.get()" % me)
        print_debug("![%d] self.global_slice = %s" % (me,self.global_slice))
        shape = util.slices_to_shape(self.global_slice)
        print_debug("![%d] inside gainarray.get() shape=%s" % (me,shape))
        nd_buffer = np.zeros(shape, dtype=self.dtype)
        _lo = []
        _hi = []
        _skip = []
        adjust = []
        for item in self.global_slice:
            if isinstance(item, slice):
                if item.step < 0:
                    adjust.append(slice(None,None,-1))
                    length = util.slicelength(item)-1
                    _lo.append(item.step*length + item.start)
                    _hi.append(item.start+1)
                    _skip.append(-item.step)
                else:
                    adjust.append(slice(0,None,None))
                    _lo.append(item.start)
                    _hi.append(item.stop)
                    _skip.append(item.step)
            elif isinstance(item, (int,long)):
                _lo.append(item)
                _hi.append(item+1)
                _skip.append(1)
            elif item is None:
                adjust.append(None)
            else:
                raise IndexError, "invalid index item"
        print_debug("![%d] ga.strided_get(%s, %s, %s, %s, nd_buffer)" % (
                me, self.handle, _lo, _hi, _skip))
        print_debug("![%d] adjust=%s" % (me,adjust))
        ret = ga.strided_get(self.handle, _lo, _hi, _skip, nd_buffer)
        print_debug("![%d] ret.shape=%s" % (me,ret.shape))
        ret = ret[adjust]
        print_debug("![%d] adjusted ret.shape=%s" % (me,ret.shape))
        return ret

    ################################################################
    ### ndarray properties
    ################################################################

    def _get_T(self):
        raise NotImplementedError, "TODO"
    def _set_T(self):
        raise NotImplementedError, "TODO"
    T = property(_get_T, _set_T)

    def _get_data(self):
        raise NotImplementedError, "TODO"
    data = property(_get_data)

    def _get_dtype(self):
        return self._dtype
    dtype = property(_get_dtype)

    def _get_flags(self):
        raise NotImplementedError, "TODO"
    flags = property(_get_flags)

    def _get_flat(self):
        raise NotImplementedError, "TODO"
    flat = property(_get_flat)

    def _get_imag(self):
        raise NotImplementedError, "TODO"
    imag = property(_get_imag)

    def _get_real(self):
        raise NotImplementedError, "TODO"
    real = property(_get_real)

    def _get_size(self):
        return reduce(lambda x,y: x*y, self.shape)
    size = property(_get_size)

    def _get_itemsize(self):
        return self._dtype.itemsize
    itemsize = property(_get_itemsize)

    def _get_nbytes(self):
        return self.itemsize * self.size
    nbytes = property(_get_nbytes)

    def _get_ndim(self):
        return len(self._shape)
    ndim = property(_get_ndim)

    def _get_shape(self):
        return self._shape
    def _set_shape(self, value):
        raise NotImplementedError, "TODO"
    shape = property(_get_shape, _set_shape)

    def _get_strides(self):
        strides = [self.itemsize]
        for size in self.shape[-1:0:-1]:
            strides = [size*strides[0]] + strides
        return strides
    strides = property(_get_strides)

    def _get_ctypes(self):
        raise NotImplementedError, "TODO"
    ctypes = property(_get_ctypes)

    def _get_base(self):
        return self._base
    base = property(_get_base)

    def __str__(self):
        result = ""
        if 0 == me:
            result = str(self.get())
        return result

    ################################################################
    ### ndarray operator overloading
    ################################################################
    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        if isinstance(key, (str,unicode)):
            raise NotImplementedError, "str or unicode key"
        if self.ndim == 0:
            raise IndexError, "0-d arrays can't be indexed"
        try:
            iter(key)
        except:
            key = [key]
        fancy = False
        for arg in key:
            if isinstance(arg, (ndarray,np.ndarray,list,tuple)):
                fancy = True
                break
        if fancy:
            raise NotImplementedError, "TODO: fancy indexing"
        key = util.canonicalize_indices(self.shape, key)
        new_shape = util.slices_to_shape(key)
        a = ndarray(new_shape, self.dtype, base=self)
        a.global_slice = util.slice_arithmetic(self.global_slice, key)
        return a

class _UnaryOperation(object):

    def __init__(self, func):
        self.func = func

    def __call__(self, input, out=None, *args, **kwargs):
        print_sync("_UnaryOperation.__call__ %s" % self.func)
        ga.sync()
        input = asarray(input)
        if out is None:
            out = ndarray(input.shape, input.dtype)
        elif input.shape != out.shape:
            # broadcasting doesn't apply to unary operations
            raise ValueError, 'invalid return array shape'
        # get out as an ndarray first
        npout = out.access()
        if npout is None:
            print_sync("npout is None")
            print_sync("NA")
        # first opt: input and out are same object
        elif input is out:
            print_sync("same object")
            print_sync("NA")
            self.func(npout, npout, *args, **kwargs)
            ga.release_update(out.handle)
        # second opt: same distributions and same slicing
        # in practice this might not happen all that often
        elif (ga.compare_distr(input.handle, out.handle)
                and input.global_slice == out.global_slice):
            print_sync("same distributions")
            print_sync("NA")
            npin = input.access()
            self.func(npin, npout, *args, **kwargs)
            ga.release_update(out.handle)
            ga.release(input.handle)
        else:
            lo,hi = ga.distribution(out.handle)
            result = util.get_slice(out.global_slice, lo, hi)
            print_sync("local_slice=%s" % str(result))
            matching_input = input[result]
            npin = matching_input.get()
            print_sync("npin.shape=%s, npout.shape=%s" % (
                npin.shape, npout.shape))
            self.func(npin, npout, *args, **kwargs)
            ga.release_update(out.handle)
        ga.sync()
        return out
            
#..............................................................................
# Unary ufuncs
exp = _UnaryOperation(umath.exp)
conjugate = _UnaryOperation(umath.conjugate)
sin = _UnaryOperation(umath.sin)
cos = _UnaryOperation(umath.cos)
tan = _UnaryOperation(umath.tan)
arctan = _UnaryOperation(umath.arctan)
arcsinh = _UnaryOperation(umath.arcsinh)
sinh = _UnaryOperation(umath.sinh)
cosh = _UnaryOperation(umath.cosh)
tanh = _UnaryOperation(umath.tanh)
abs = absolute = _UnaryOperation(umath.absolute)
fabs = _UnaryOperation(umath.fabs)
negative = _UnaryOperation(umath.negative)
floor = _UnaryOperation(umath.floor)
ceil = _UnaryOperation(umath.ceil)
around = _UnaryOperation(np.round_)
logical_not = _UnaryOperation(umath.logical_not)
# Domained unary ufuncs .......................................................
sqrt = _UnaryOperation(umath.sqrt)
log = _UnaryOperation(umath.log)
log10 = _UnaryOperation(umath.log10)
tan = _UnaryOperation(umath.tan)
arcsin = _UnaryOperation(umath.arcsin)
arccos = _UnaryOperation(umath.arccos)
arccosh = _UnaryOperation(umath.arccosh)
arctanh = _UnaryOperation(umath.arctanh)

def zeros(shape, dtype=np.float, order='C'):
    a = ndarray(shape, dtype)
    ga.zero(a.handle)
    return a

def ones(shape, dtype=np.float, order='C'):
    a = ndarray(shape, dtype)
    ga.fill(a.handle, 1)
    return a

def fromfunction(func, shape, **kwargs):
    dtype = kwargs.pop('dtype', np.float32)
    # create the new GA (collective operation)
    a = ndarray(shape, dtype)
    # determine which part of 'a' we maintain
    local_array = ga.access(a.handle)
    if local_array is not None:
        lo,hi = ga.distribution(a.handle)
        local_shape = hi-lo
        # create a numpy indices array
        args = np.indices(local_shape, dtype=dtype)
        # modify the indices arrays based on our distribution
        for index in xrange(len(lo)):
            args[index] += lo[index]
        # call the passed function
        buf = func(*args, **kwargs)
        # now put the data into the global array
        local_array[:] = buf
    ga.sync()
    return a

def arange(start, stop=None, step=None, dtype=None):
    if step == 0:
        raise ValueError, "step size of 0 not allowed"
    if not step:
        step = 1
    if not stop:
        start,stop = 0,start
    length = 0
    if ((step < 0 and stop >= start) or (step > 0 and start >= stop)):
        length = 0
    length = math.ceil((stop-start)/step) # true division, otherwise off by one
    if dtype is None:
        if (isinstance(start, (int,long))
                and isinstance(stop, (int,long))
                and isinstance(step, (int,long))):
            dtype = np.int32
        else:
            dtype = np.float64
    a = ndarray((int(length),), dtype)
    a_local = a.access()
    if a_local is not None:
        lo,hi = ga.distribution(a.handle)
        a_local[...] = np.arange(lo[0],hi[0])
        a_local *= step
        a_local += start
    return a

def dot(first, second):
    """Dot product of two arrays.

    For 2-D arrays it is equivalent to matrix multiplication, and for 1-D arrays
    to inner product of vectors (without complex conjugation). For N dimensions
    it is a sum product over the last axis of a and the second-to-last of b:
        dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])

    """
    """
    #print "dot"
    # We cheat for now. Only supports 1D dot
    if not (isinstance(first, (gainarray, flatiter))
            or isinstance(second, (gainarray, flatiter))):
        # numpy pass through
        return np.dot(first,second)
    if isinstance(first, flatiter):
        #print "before flatten attempt first"
        first = first.base.flatten()
    if isinstance(second, flatiter):
        #print "before flatten attempt second"
        second = second.base.flatten()
    assert first.ndim == second.ndim
    assert first.ndim == 1, "TODO. Only supports 1D dot for now."
    tmp = first * second
    a = tmp.access()
    result = np.add.reduce(a)
    return MPI.COMM_WORLD.Allreduce(result,MPI.SUM)
    """
    raise NotImplementedError

def asarray(a, dtype=None, order=None):
    if isinstance(a, ndarray):
        return a
    else:
        npa = np.asarray(a)
        g_a = ndarray(npa.shape, npa.dtype, npa)
        return g_a

def print_debug(s):
    if False:
        print s

def print_sync(what):
    if False:
        ga.sync()
        if 0 == me:
            print "[0] %s" % str(what)
            for proc in xrange(1,nproc):
                data = MPI.COMM_WORLD.recv(source=proc, tag=11)
                print "[%d] %s" % (proc, str(data))
        else:
            MPI.COMM_WORLD.send(what, dest=0, tag=11)
        ga.sync()

# imports from 'numpy' module every missing attribute from 'gain' module
# replaces 'gain' docstrings from 'numpy' module
if __name__ != '__main__':
    self_module = sys.modules[__name__]
    for attr in dir(np):
        np_obj = getattr(np, attr)
        if hasattr(self_module, attr):
            if not me: print "gain override exists for: %s" % attr
            try:
                self_obj = getattr(self_module, attr)
                self_obj.__doc__ = np_obj.__doc__
            except AttributeError:
                pass
        else:
            setattr(self_module, attr, getattr(np, attr))
