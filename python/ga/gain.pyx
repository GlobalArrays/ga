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

# at what point do we distribute arrays versus leaving as np.ndarray?
SIZE_THRESHOLD = 1
DEBUG = False
DEBUG_SYNC = False

gatypes = {
np.dtype(np.int8):       ga.C_CHAR,
np.dtype(np.int32):      ga.C_INT,
np.dtype(np.int64):      ga.C_LONG,
np.dtype(np.float32):    ga.C_FLOAT,
np.dtype(np.float64):    ga.C_DBL,
np.dtype(np.float128):   ga.C_LDBL,
np.dtype(np.complex64):  ga.C_SCPL,
np.dtype(np.complex128): ga.C_DCPL,
np.dtype(np.complex256): ga.C_LDCPL,
}

class flagsobj(object):
    def __init__(self):
        self._c = True
        self._f = False
        self._o = True
        self._w = True
        self._a = True
        self._u = False
    def _get_c(self):
        return self._c
    c_contiguous = property(_get_c)
    def _get_f(self):
        return self._f
    f_contiguous = property(_get_f)
    def _get_o(self):
        return self._o
    owndata = property(_get_o)
    def _get_w(self):
        return self._w
    writeable = property(_get_w)
    def _get_a(self):
        return self._a
    aligned = property(_get_a)
    def _get_u(self):
        return self._u
    updateifcopy = property(_get_u)
    def __getitem__(self, item):
        if isinstance(item, str):
            if item == "C" or item == "C_CONTIGUOUS":
                return self._c
            if item == "F" or item == "F_CONTIGUOUS":
                return self._f
            if item == "O" or item == "OWNDATA":
                return self._o
            if item == "W" or item == "WRITEABLE":
                return self._w
            if item == "A" or item == "ALIGNED":
                return self._a
            if item == "U" or item == "UPDATEIFCOPY":
                return self._u
        raise KeyError, "Unknown flag"
    def __repr__(self):
        return """  C_CONTIGUOUS : %s
  F_CONTIGUOUS : %s
  OWNDATA : %s
  WRITEABLE : %s
  ALIGNED : %s
  UPDATEIFCOPY : %s""" % (self._c, self._f, self._o,
                          self._w, self._a, self._u)

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
    _is_real : bool
        Whether this is a 'real' view of a complex ndarray.
    _is_imag : bool
        Whether this is an 'imag' view of a complex ndarray.

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
        try:
            iter(shape)
        except:
            shape = [shape]
        self._shape = shape
        self._dtype = np.dtype(dtype)
        self._order = order
        self._base = base
        self._is_real = False
        self._is_imag = False
        if order is not None:
            raise NotImplementedError, "order parameter not supported"
        if base is None:
            self._flags = flagsobj()
            dtype_ = np.dtype(dtype)
            gatype = None
            if dtype_ in gatypes:
                gatype = gatypes[dtype_]
            else:
                gatype = ga.register_dtype(dtype_)
                gatypes[dtype_] = gatype
            self.handle = ga.create(gatype, shape)
            print_sync("created: handle=%s type=%s shape=%s distribution=%s"%(
                self.handle, self._dtype, self._shape,
                str(self.distribution())))
            if buffer is not None:
                local = ga.access(self.handle)
                if local is not None:
                    a = None
                    if isinstance(buffer, np.ndarray):
                        buffer.shape = shape
                        a = buffer
                    else:
                        a = np.ndarray(shape, dtype, buffer, offset,
                                strides, order)
                    lo,hi = self.distribution()
                    a = a[map(lambda x,y: slice(x,y), lo, hi)]
                    local[:] = a
                    self.release_update()
            self.global_slice = map(lambda x:slice(0,x,1), shape)
            self._strides = [self.itemsize]
            for size in shape[-1:0:-1]:
                self._strides = [size*self._strides[0]] + self._strides
        else:
            self._flags = base._flags
            self._flags._c = False
            self._flags._o = False
            self.handle = base.handle
            self.global_slice = base.global_slice
            self._strides = strides
            print_debug("![%d] view: hndl=%s typ=%s shp=%s dstrbtn=%s"%(
                    me, self.handle, self._dtype, self._shape,
                    str(self.distribution())))

    def __del__(self):
        if self.base is None:
            if ga.initialized():
                print_sync("deleting %s %s" % (self.shape, self.handle))
                ga.destroy(self.handle)

    ################################################################
    ### ndarray methods added for Global Arrays
    ################################################################
    def distribution(self):
        return ga.distribution(self.handle)

    def owns(self):
        lo,hi = self.distribution()
        return np.all(hi>=0)

    def access(self):
        """Access the local array. Return None if no data is owned."""
        if self.owns():
            lo,hi = self.distribution()
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
                if self._is_real:
                    ret = ret.real
                elif self._is_imag:
                    ret = ret.imag
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
        dtype = self._dtype
        if self._is_real or self._is_imag:
            dtype = np.dtype("complex%s" % (self._dtype.itemsize*2*8))
        nd_buffer = np.zeros(shape, dtype=dtype)
        _lo = []
        _hi = []
        _skip = []
        adjust = []
        need_strided = False
        for item in self.global_slice:
            if isinstance(item, slice):
                if item.step > 1 or item.step < -1:
                    need_strided = True
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
                adjust.append(slice(0,1,1))
            else:
                raise IndexError, "invalid index item"
        print_debug("![%d] ga.strided_get(%s, %s, %s, %s, nd_buffer)" % (
                me, self.handle, _lo, _hi, _skip))
        print_debug("![%d] adjust=%s" % (me,adjust))
        ret = None
        if need_strided:
            ret = ga.strided_get(self.handle, _lo, _hi, _skip, nd_buffer)
        else:
            ret = ga.get(self.handle, _lo, _hi, nd_buffer)
        print_debug("![%d] ret.shape=%s" % (me,ret.shape))
        ret = ret[adjust]
        if self._is_real:
            ret = ret.real
        elif self._is_imag:
            ret = ret.imag
        print_debug("![%d] adjusted ret.shape=%s" % (me,ret.shape))
        return ret

    def release(self):
        ga.release(self.handle)

    def release_update(self):
        ga.release_update(self.handle)

    ################################################################
    ### ndarray properties
    ################################################################

    def _get_T(self):
        raise NotImplementedError, "TODO"
    T = property(_get_T)

    def _get_data(self):
        a = self.access()
        if a is not None:
            return a.data
        return None
    data = property(_get_data)

    def _get_dtype(self):
        return self._dtype
    dtype = property(_get_dtype)

    def _get_flags(self):
        return self._flags
    flags = property(_get_flags)

    def _get_flat(self):
        raise NotImplementedError, "TODO"
    flat = property(_get_flat)

    def _get_imag(self):
        if self._dtype.kind != 'c':
            return zeros(self.shape, self.dtype)
        else:
            ret = self[:]
            ret._is_imag = True
            ret._dtype = np.dtype("float%s" % (self._dtype.itemsize/2*8))
            return ret
    def _set_imag(self, value):
        if self._dtype.kind != 'c':
            raise TypeError, "array does not have imaginary part to set"
        else:
            self._get_imag()[:] = value
    imag = property(_get_imag,_set_imag)

    def _get_real(self):
        if self._dtype.kind != 'c':
            return self
        else:
            ret = self[:]
            ret._is_real = True
            ret._dtype = np.dtype("float%s" % (self._dtype.itemsize/2*8))
            return ret
    def _set_real(self, value):
        self._get_real()[:] = value
    real = property(_get_real,_set_real)

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

    def __repr__(self):
        result = ""
        if 0 == me:
            result = repr(self.get())
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

    def __setitem__(self, key, value):
        new_self = self[key]
        value = asarray(value)
        npvalue = None
        release_value = False
        # get new_self as an ndarray first
        npself = new_self.access()
        if npself is None:
            print_sync("npself is None")
            print_sync("NA")
            ga.sync()
            return
        if isinstance(value, ndarray):
            if (ga.compare_distr(value.handle, new_self.handle)
                    and value.global_slice == new_self.global_slice):
                # opt: same distributions and same slicing
                # in practice this might not happen all that often
                print_sync("same distributions")
                print_sync("NA")
                npvalue = value.access()
                release_value = True
            else:
                lo,hi = new_self.distribution()
                result = util.get_slice(new_self.global_slice, lo, hi)
                result = util.broadcast_chomp(value.shape, result)
                print_sync("local_slice=%s" % str(result))
                matching_input = value[result]
                npvalue = matching_input.get()
                print_sync("npvalue.shape=%s, npself.shape=%s" % (
                    npvalue.shape, npself.shape))
        else:
            if value.ndim > 0:
                lo,hi = new_self.distribution()
                result = util.get_slice(new_self.global_slice, lo, hi)
                result = util.broadcast_chomp(value.shape, result)
                print_sync("local_slice=%s" % str(result))
                npvalue = value[result]
                print_sync("npvalue.shape=%s, npself.shape=%s" % (
                    npvalue.shape, npself.shape))
            else:
                npvalue = value
        npself[:] = npvalue

class _UnaryOperation(object):

    def __init__(self, func):
        self.func = func
        self.__doc__ = func.__doc__

    def __call__(self, input, out=None, *args, **kwargs):
        print_sync("_UnaryOperation.__call__ %s" % self.func)
        input = asarray(input)
        if not (isinstance(input, ndarray) or isinstance(out, ndarray)):
            print_sync("_UnaryOperation.__call__ %s pass through" % self.func)
            # no ndarray instances used, pass through immediately to numpy
            return self.func(input, out, *args, **kwargs)
        # since we have an ndarray somewhere
        ga.sync()
        if out is None:
            # input must be an ndarray given previous conditionals
            # TODO okay, is there something better than this?
            ignore = np.ones(1, dtype=input.dtype)
            out_type = self.func(ignore).dtype
            print_debug("out_type = %s" % out_type)
            out = ndarray(input.shape, out_type) # distribute
        # sanity checks
        if not isinstance(out, (ndarray, np.ndarray)):
            raise TypeError, "return arrays must be of ArrayType"
        elif input.shape != out.shape:
            # broadcasting doesn't apply to unary operations
            raise ValueError, 'invalid return array shape'
        # Now figure out what to do...
        if isinstance(out, ndarray):
            # get out as an ndarray first
            npout = out.access()
            if npout is None:
                print_sync("npout is None")
                print_sync("NA")
                ga.sync()
                return out
            npin = None
            release_in = False
            # first opt: input and out are same object
            if input is out:
                print_sync("same object")
                print_sync("NA")
                npin = npout
            elif isinstance(input, ndarray):
                # second opt: same distributions and same slicing
                # in practice this might not happen all that often
                if (ga.compare_distr(input.handle, out.handle)
                        and input.global_slice == out.global_slice):
                    print_sync("same distributions")
                    print_sync("NA")
                    npin = input.access()
                    release_in = True
                else:
                    lo,hi = out.distribution()
                    result = util.get_slice(out.global_slice, lo, hi)
                    print_sync("local_slice=%s" % str(result))
                    matching_input = input[result]
                    npin = matching_input.get()
                    print_sync("npin.shape=%s, npout.shape=%s" % (
                        npin.shape, npout.shape))
            else:
                lo,hi = out.distribution()
                result = util.get_slice(out.global_slice, lo, hi)
                print_sync("np.ndarray slice=%s" % str(result))
                npin = input[result]
                print_sync("npin.shape=%s, npout.shape=%s" % (
                        npin.shape, npout.shape))
            self.func(npin, npout, *args, **kwargs)
            if release_in: input.release()
            out.release_update()
        else:
            print_sync("out is not ndarray")
            print_sync("NA")
            # out is not an ndarray
            npin = input
            if isinstance(input, ndarray):
                # input is an ndarray, so get entire thing
                npin = input.get()
            self.func(npin, out, *args, **kwargs)
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

class _BinaryOperation(object):

    def __init__(self, func):
        self.func = func
        self.__doc__ = func.__doc__

    def __call__(self, first, second, out=None, *args, **kwargs):
        print_sync("_BinaryOperation.__call__ %s" % self.func)
        # just in case
        first = asarray(first)
        second = asarray(second)
        if not (isinstance(first, ndarray)
                or isinstance(second, ndarray)
                or isinstance(out, ndarray)):
            # no ndarray instances used, pass through immediately to numpy
            print_sync("_BinaryOperation.__call__ %s pass through" % self.func)
            return self.func(first, second, out, *args, **kwargs)
        # since we have an ndarray somewhere
        ga.sync()
        if out is None:
            # first and/or second must be ndarrays given previous conditionals
            # TODO okay, is there something better than this?
            ignore1 = np.ones(1, dtype=first.dtype)
            ignore2 = np.ones(1, dtype=second.dtype)
            out_type = self.func(ignore1,ignore2).dtype
            shape = util.broadcast_shape(first.shape, second.shape)
            print_sync("broadcast_shape = %s" % str(shape))
            out = ndarray(shape, out_type)
        # sanity checks
        if not isinstance(out, (ndarray, np.ndarray)):
            raise TypeError, "return arrays must be of ArrayType"
        # Now figure out what to do...
        if isinstance(out, ndarray):
            # get out as an ndarray first
            npout = out.access()
            if npout is None:
                print_sync("npout is None")
                print_sync("NA")
                print_sync("NA")
                print_sync("NA")
                ga.sync()
                return out
            # get matching and compatible portions of input arrays
            # broadcasting rules (may) apply
            npfirst = None
            release_first = False
            if first is out:
                # first opt: first and out are same object
                npfirst = npout
                print_sync("same object first out")
                print_sync("NA")
            elif isinstance(first, ndarray):
                if (ga.compare_distr(first.handle, out.handle)
                        and first.global_slice == out.global_slice):
                    # second opt: same distributions and same slicing
                    # in practice this might not happen all that often
                    print_sync("same distributions")
                    print_sync("NA")
                    npfirst = first.access()
                    release_first = True
                else:
                    lo,hi = out.distribution()
                    result = util.get_slice(out.global_slice, lo, hi)
                    result = util.broadcast_chomp(first.shape, result)
                    print_sync("local_slice=%s" % str(result))
                    matching_input = first[result]
                    npfirst = matching_input.get()
                    print_sync("npfirst.shape=%s, npout.shape=%s" % (
                        npfirst.shape, npout.shape))
            else:
                if first.ndim > 0:
                    lo,hi = out.distribution()
                    result = util.get_slice(out.global_slice, lo, hi)
                    result = util.broadcast_chomp(first.shape, result)
                    print_sync("local_slice=%s" % str(result))
                    npfirst = first[result]
                    print_sync("npfirst.shape=%s, npout.shape=%s" % (
                        npfirst.shape, npout.shape))
                else:
                    npfirst = first
            npsecond = None
            release_second = False
            if second is out:
                # first opt: second and out are same object
                npsecond = npout
                print_sync("same object second out")
                print_sync("NA")
            elif isinstance(second, ndarray):
                if (ga.compare_distr(second.handle, out.handle)
                        and second.global_slice == out.global_slice):
                    # second opt: same distributions and same slicing
                    # in practice this might not happen all that often
                    print_sync("same distributions")
                    print_sync("NA")
                    npsecond = second.access()
                    release_second = True
                else:
                    lo,hi = out.distribution()
                    result = util.get_slice(out.global_slice, lo, hi)
                    result = util.broadcast_chomp(second.shape, result)
                    print_sync("local_slice=%s" % str(result))
                    matching_input = second[result]
                    npsecond = matching_input.get()
                    print_sync("npsecond.shape=%s, npout.shape=%s" % (
                        npsecond.shape, npout.shape))
            else:
                if second.ndim > 0:
                    lo,hi = out.distribution()
                    result = util.get_slice(out.global_slice, lo, hi)
                    result = util.broadcast_chomp(second.shape, result)
                    print_sync("local_slice=%s" % str(result))
                    npsecond = second[result]
                    print_sync("npsecond.shape=%s, npout.shape=%s" % (
                        npsecond.shape, npout.shape))
                else:
                    npsecond = second
            self.func(npfirst, npsecond, npout, *args, **kwargs)
            if release_first: first.release()
            if release_second: second.release()
            out.release_update()
        else:
            print_sync("npout is None")
            print_sync("NA")
            print_sync("NA")
            print_sync("NA")
            # out is not an ndarray
            ndfirst = first
            if isinstance(first, ndarray):
                ndfirst = first.get()
            ndsecond = second
            if isinstance(second, ndarray):
                ndsecond = second.get()
            self.func(ndfirst, ndsecond, out, *args, **kwargs)
        ga.sync()
        return out

    def reduce(self, *args, **kwargs):
        raise NotImplementedError, "TODO, sorry"
    def accumulate(self, *args, **kwargs):
        raise NotImplementedError, "TODO, sorry"
    def outer(self, *args, **kwargs):
        raise NotImplementedError, "TODO, sorry"
    def reduceat(self, *args, **kwargs):
        raise NotImplementedError, "TODO, sorry"

# Binary ufuncs ...............................................................
add = _BinaryOperation(umath.add)
subtract = _BinaryOperation(umath.subtract)
multiply = _BinaryOperation(umath.multiply)
power = _BinaryOperation(umath.power)
arctan2 = _BinaryOperation(umath.arctan2)
equal = _BinaryOperation(umath.equal)
equal.reduce = None
not_equal = _BinaryOperation(umath.not_equal)
not_equal.reduce = None
less_equal = _BinaryOperation(umath.less_equal)
less_equal.reduce = None
greater_equal = _BinaryOperation(umath.greater_equal)
greater_equal.reduce = None
less = _BinaryOperation(umath.less)
less.reduce = None
greater = _BinaryOperation(umath.greater)
greater.reduce = None
logical_and = _BinaryOperation(umath.logical_and)
alltrue = _BinaryOperation(umath.logical_and)
logical_or = _BinaryOperation(umath.logical_or)
sometrue = logical_or.reduce
logical_xor = _BinaryOperation(umath.logical_xor)
bitwise_and = _BinaryOperation(umath.bitwise_and)
bitwise_or = _BinaryOperation(umath.bitwise_or)
bitwise_xor = _BinaryOperation(umath.bitwise_xor)
hypot = _BinaryOperation(umath.hypot)
# Domained binary ufuncs ......................................................
divide = _BinaryOperation(umath.divide)
true_divide = _BinaryOperation(umath.true_divide)
floor_divide = _BinaryOperation(umath.floor_divide)
remainder = _BinaryOperation(umath.remainder)
fmod = _BinaryOperation(umath.fmod)

def zeros(shape, dtype=np.float, order='C'):
    """zeros(shape, dtype=float, order='C')

    Return a new array of given shape and type, filled with zeros.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        The desired data-type for the array, e.g., `numpy.int8`.  Default is
        `numpy.float64`.
    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous
        (row- or column-wise) order in memory.

    Returns
    -------
    out : ndarray
        Array of zeros with the given shape, dtype, and order.

    See Also
    --------
    zeros_like : Return an array of zeros with shape and type of input.
    ones_like : Return an array of ones with shape and type of input.
    empty_like : Return an empty array with shape and type of input.
    ones : Return a new array setting values to one.
    empty : Return a new uninitialized array.

    Examples
    --------
    >>> np.zeros(5)
    array([ 0.,  0.,  0.,  0.,  0.])

    >>> np.zeros((5,), dtype=numpy.int)
    array([0, 0, 0, 0, 0])

    >>> np.zeros((2, 1))
    array([[ 0.],
           [ 0.]])

    >>> s = (2,2)
    >>> np.zeros(s)
    array([[ 0.,  0.],
           [ 0.,  0.]])

    >>> np.zeros((2,), dtype=[('x', 'i4'), ('y', 'i4')]) # custom dtype
    array([(0, 0), (0, 0)],
          dtype=[('x', '<i4'), ('y', '<i4')])

    """
    a = ndarray(shape, dtype)
    buf = a.access()
    if buf is not None:
        buf[:] = 0
        a.release_update()
    return a

def zeros_like(a, dtype=None, order='K', subok=True):
    """Return an array of zeros with the same shape and type as a given array.

    Equivalent to ``a.copy().fill(0)``.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define the parameters of
        the returned array.

    Returns
    -------
    out : ndarray
        Array of zeros with same shape and type as `a`.

    See Also
    --------
    ones_like : Return an array of ones with shape and type of input.
    empty_like : Return an empty array with shape and type of input.
    zeros : Return a new array setting values to zero.
    ones : Return a new array setting values to one.
    empty : Return a new uninitialized array.

    Examples
    --------
    >>> x = np.arange(6)
    >>> x = x.reshape((2, 3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.zeros_like(x)
    array([[0, 0, 0],
           [0, 0, 0]])

    >>> y = np.arange(3, dtype=np.float)
    >>> y
    array([ 0.,  1.,  2.])
    >>> np.zeros_like(y)
    array([ 0.,  0.,  0.])

    """
    return a.copy().fill(0)

def ones(shape, dtype=np.float, order='C'):
    """Return a new array of given shape and type, filled with ones.

    Please refer to the documentation for `zeros`.

    See Also
    --------
    zeros

    Examples
    --------
    >>> np.ones(5)
    array([ 1.,  1.,  1.,  1.,  1.])

    >>> np.ones((5,), dtype=np.int)
    array([1, 1, 1, 1, 1])

    >>> np.ones((2, 1))
    array([[ 1.],
           [ 1.]])

    >>> s = (2,2)
    >>> np.ones(s)
    array([[ 1.,  1.],
           [ 1.,  1.]])
    
    """
    a = ndarray(shape, dtype)
    buf = a.access()
    if buf is not None:
        buf[:] = 1
        a.release_update()
    return a

def ones_like(x):
    """ones_like(x[, out])

    Returns an array of ones with the same shape and type as a given array.

    Equivalent to ``a.copy().fill(1)``.

    Please refer to the documentation for `zeros_like`.

    See Also
    --------
    zeros_like

    Examples
    --------
    >>> a = np.array([[1, 2, 3], [4, 5, 6]])
    >>> np.ones_like(a)
    array([[1, 1, 1],
           [1, 1, 1]])

    """
    return x.copy().fill(1)

def empty(shape, dtype=float, order='C'):
    """empty(shape, dtype=float, order='C')

    Return a new array of given shape and type, without initializing entries.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the empty array
    dtype : data-type, optional
        Desired output data-type.
    order : {'C', 'F'}, optional
        Whether to store multi-dimensional data in C (row-major) or
        Fortran (column-major) order in memory.

    See Also
    --------
    empty_like, zeros, ones

    Notes
    -----
    `empty`, unlike `zeros`, does not set the array values to zero,
    and may therefore be marginally faster.  On the other hand, it requires
    the user to manually set all the values in the array, and should be
    used with caution.

    Examples
    --------
    >>> np.empty([2, 2])
    array([[ -9.74499359e+001,   6.69583040e-309],  #random data
           [  2.13182611e-314,   3.06959433e-309]])

    >>> np.empty([2, 2], dtype=int)
    array([[-1073741821, -1067949133],  #random data
           [  496041986,    19249760]])

    """
    return ndarray(shape, dtype)

def empty_like(a, dtype=None, order='K', subok=True):
    """    Return a new array with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define the parameters of the
        returned array.

    Returns
    -------
    out : ndarray
        Array of random data with the same shape and type as `a`.

    See Also
    --------
    ones_like : Return an array of ones with shape and type of input.
    zeros_like : Return an array of zeros with shape and type of input.
    empty : Return a new uninitialized array.
    ones : Return a new array setting values to one.
    zeros : Return a new array setting values to zero.

    Notes
    -----
    This function does *not* initialize the returned array; to do that use
    `zeros_like` or `ones_like` instead. It may be marginally faster than the
    functions that do set the array values.

    Examples
    --------
    >>> a = ([1,2,3], [4,5,6])                         # a is array-like
    >>> np.empty_like(a)
    array([[-1073741821, -1073741821,           3],    #random
           [          0,           0, -1073741821]])
    >>> a = np.array([[1., 2., 3.],[4.,5.,6.]])
    >>> np.empty_like(a)
    array([[ -2.00000715e+000,   1.48219694e-323,  -2.00000572e+000], #random
           [  4.38791518e-305,  -2.00000715e+000,   4.17269252e-309]])
    
    """
    return empty(a.shape, dtype or a.dtype)

def eye(N, M=None, k=0, dtype=float):
    """Return a 2-D array with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    N : int
      Number of rows in the output.
    M : int, optional
      Number of columns in the output. If None, defaults to `N`.
    k : int, optional
      Index of the diagonal: 0 refers to the main diagonal, a positive value
      refers to an upper diagonal, and a negative value to a lower diagonal.
    dtype : dtype, optional
      Data-type of the returned array.

    Returns
    -------
    I : ndarray (N,M)
      An array where all elements are equal to zero, except for the `k`-th
      diagonal, whose values are equal to one.

    See Also
    --------
    diag : Return a diagonal 2-D array using a 1-D array specified by the user.

    Examples
    --------
    >>> np.eye(2, dtype=int)
    array([[1, 0],
           [0, 1]])
    >>> np.eye(3, k=1)
    array([[ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  0.,  0.]])
    
    """
    if M is None:
        M = N
    a = zeros((N,M), dtype=dtype)
    nda = a.access()
    if nda is not None:
        lo,hi = a.distribution()
        indices = np.indices(nda.shape)
        indices[0] += lo[0]
        indices[1] += lo[1]-k
        bindex = (indices[0] == indices[1])
        nda[bindex] = 1
        a.release_update()
    return a

def identity(n, dtype=None):
    """Return the identity array.

    The identity array is a square array with ones on
    the main diagonal.

    Parameters
    ----------
    n : int
        Number of rows (and columns) in `n` x `n` output.
    dtype : data-type, optional
        Data-type of the output.  Defaults to ``float``.

    Returns
    -------
    out : ndarray
        `n` x `n` array with its main diagonal set to one,
        and all other elements 0.

    Examples
    --------
    >>> np.identity(3)
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])

    """
    if dtype is None:
        dtype = np.dtype(float)
    return eye(n,n,dtype=dtype)

def fromfunction(func, shape, **kwargs):
    """Construct an array by executing a function over each coordinate.

    The resulting array therefore has a value ``fn(x, y, z)`` at
    coordinate ``(x, y, z)``.

    Parameters
    ----------
    function : callable
        The function is called with N parameters, each of which
        represents the coordinates of the array varying along a
        specific axis.  For example, if `shape` were ``(2, 2)``, then
        the parameters would be two arrays, ``[[0, 0], [1, 1]]`` and
        ``[[0, 1], [0, 1]]``.  `function` must be capable of operating on
        arrays, and should return a scalar value.
    shape : (N,) tuple of ints
        Shape of the output array, which also determines the shape of
        the coordinate arrays passed to `function`.
    dtype : data-type, optional
        Data-type of the coordinate arrays passed to `function`.
        By default, `dtype` is float.

    Returns
    -------
    out : any
        The result of the call to `function` is passed back directly.
        Therefore the type and shape of `out` is completely determined by
        `function`.

    See Also
    --------
    indices, meshgrid

    Notes
    -----
    Keywords other than `shape` and `dtype` are passed to `function`.

    Examples
    --------
    >>> np.fromfunction(lambda i, j: i == j, (3, 3), dtype=int)
    array([[ True, False, False],
           [False,  True, False],
           [False, False,  True]], dtype=bool)

    >>> np.fromfunction(lambda i, j: i + j, (3, 3), dtype=int)
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4]])
    
    """
    dtype = kwargs.pop('dtype', np.float32)
    # create the new GA (collective operation)
    a = ndarray(shape, dtype)
    # determine which part of 'a' we maintain
    local_array = ga.access(a.handle)
    if local_array is not None:
        lo,hi = a.distribution()
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
    """Return evenly spaced values within a given interval.

    Values are generated within the half-open interval ``[start, stop)``
    (in other words, the interval including `start` but excluding `stop`).
    For integer arguments the function is equivalent to the Python built-in
    `range <http://docs.python.org/lib/built-in-funcs.html>`_ function,
    but returns a ndarray rather than a list.

    Parameters
    ----------
    start : number, optional
        Start of interval.  The interval includes this value.  The default
        start value is 0.
    stop : number
        End of interval.  The interval does not include this value.
    step : number, optional
        Spacing between values.  For any output `out`, this is the distance
        between two adjacent values, ``out[i+1] - out[i]``.  The default
        step size is 1.  If `step` is specified, `start` must also be given.
    dtype : dtype
        The type of the output array.  If `dtype` is not given, infer the data
        type from the other input arguments.

    Returns
    -------
    out : ndarray
        Array of evenly spaced values.

        For floating point arguments, the length of the result is
        ``ceil((stop - start)/step)``.  Because of floating point overflow,
        this rule may result in the last element of `out` being greater
        than `stop`.

    See Also
    --------
    linspace : Evenly spaced numbers with careful handling of endpoints.
    ogrid: Arrays of evenly spaced numbers in N-dimensions
    mgrid: Grid-shaped arrays of evenly spaced numbers in N-dimensions

    Examples
    --------
    >>> np.arange(3)
    array([0, 1, 2])
    >>> np.arange(3.0)
    array([ 0.,  1.,  2.])
    >>> np.arange(3,7)
    array([3, 4, 5, 6])
    >>> np.arange(3,7,2)
    array([3, 5])
    
    """
    if step == 0:
        raise ValueError, "step size of 0 not allowed"
    if not step:
        step = 1
    if not stop:
        start,stop = 0,start
    length = 0
    if ((step < 0 and stop >= start) or (step > 0 and start >= stop)):
        length = 0
    else:
        # true division, otherwise off by one
        length = math.ceil((stop-start)/step)
    # bail if threshold not met
    if length < SIZE_THRESHOLD:
        return np.arange(start,stop,step,dtype)
    if dtype is None:
        if (isinstance(start, (int,long))
                and isinstance(stop, (int,long))
                and isinstance(step, (int,long))):
            dtype = np.int64
        else:
            dtype = np.float64
    a = ndarray((int(length),), dtype)
    a_local = a.access()
    if a_local is not None:
        lo,hi = a.distribution()
        a_local[...] = np.arange(lo[0],hi[0])
        a_local *= step
        a_local += start
    return a

def linspace(start, stop, num=50, endpoint=True, retstep=False):
    """Return evenly spaced numbers over a specified interval.

    Returns `num` evenly spaced samples, calculated over the
    interval [`start`, `stop` ].

    The endpoint of the interval can optionally be excluded.

    Parameters
    ----------
    start : scalar
        The starting value of the sequence.
    stop : scalar
        The end value of the sequence, unless `endpoint` is set to False.
        In that case, the sequence consists of all but the last of ``num + 1``
        evenly spaced samples, so that `stop` is excluded.  Note that the step
        size changes when `endpoint` is False.
    num : int, optional
        Number of samples to generate. Default is 50.
    endpoint : bool, optional
        If True, `stop` is the last sample. Otherwise, it is not included.
        Default is True.
    retstep : bool, optional
        If True, return (`samples`, `step`), where `step` is the spacing
        between samples.

    Returns
    -------
    samples : ndarray
        There are `num` equally spaced samples in the closed interval
        ``[start, stop]`` or the half-open interval ``[start, stop)``
        (depending on whether `endpoint` is True or False).
    step : float (only if `retstep` is True)
        Size of spacing between samples.


    See Also
    --------
    arange : Similiar to `linspace`, but uses a step size (instead of the
             number of samples).
    logspace : Samples uniformly distributed in log space.

    Examples
    --------
    >>> np.linspace(2.0, 3.0, num=5)
        array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ])
    >>> np.linspace(2.0, 3.0, num=5, endpoint=False)
        array([ 2. ,  2.2,  2.4,  2.6,  2.8])
    >>> np.linspace(2.0, 3.0, num=5, retstep=True)
        (array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ]), 0.25)

    Graphical illustration:

    >>> import matplotlib.pyplot as plt
    >>> N = 8
    >>> y = np.zeros(N)
    >>> x1 = np.linspace(0, 10, N, endpoint=True)
    >>> x2 = np.linspace(0, 10, N, endpoint=False)
    >>> plt.plot(x1, y, 'o')
    >>> plt.plot(x2, y + 0.5, 'o')
    >>> plt.ylim([-0.5, 1])
    >>> plt.show()

    """
    # bail if threshold not met
    if num < SIZE_THRESHOLD:
        return np.linspace(start,stop,num,endpoint,retstep)
    a = ndarray(num)
    step = None
    if endpoint:
        step = (stop-start)/(num-1)
    else:
        step = (stop-start)/num
    buf = a.access()
    if buf is not None:
        lo,hi = a.distribution()
        lo,hi = lo[0],hi[0]
        buf[:] = np.arange(lo,hi)*step+start
        a.release_update()
    ga.sync()
    if retstep:
        return a,step
    return a

def logspace(start, stop, num=50, endpoint=True, base=10.0):
    """Return numbers spaced evenly on a log scale.

    In linear space, the sequence starts at ``base ** start``
    (`base` to the power of `start`) and ends with ``base ** stop``
    (see `endpoint` below).

    Parameters
    ----------
    start : float
        ``base ** start`` is the starting value of the sequence.
    stop : float
        ``base ** stop`` is the final value of the sequence, unless `endpoint`
        is False.  In that case, ``num + 1`` values are spaced over the
        interval in log-space, of which all but the last (a sequence of
        length ``num``) are returned.
    num : integer, optional
        Number of samples to generate.  Default is 50.
    endpoint : boolean, optional
        If true, `stop` is the last sample. Otherwise, it is not included.
        Default is True.
    base : float, optional
        The base of the log space. The step size between the elements in
        ``ln(samples) / ln(base)`` (or ``log_base(samples)``) is uniform.
        Default is 10.0.

    Returns
    -------
    samples : ndarray
        `num` samples, equally spaced on a log scale.

    See Also
    --------
    arange : Similiar to linspace, with the step size specified instead of the
             number of samples. Note that, when used with a float endpoint, the
             endpoint may or may not be included.
    linspace : Similar to logspace, but with the samples uniformly distributed
               in linear space, instead of log space.

    Notes
    -----
    Logspace is equivalent to the code

    >>> y = linspace(start, stop, num=num, endpoint=endpoint)
    >>> power(base, y)

    Examples
    --------
    >>> np.logspace(2.0, 3.0, num=4)
        array([  100.        ,   215.443469  ,   464.15888336,  1000.        ])
    >>> np.logspace(2.0, 3.0, num=4, endpoint=False)
        array([ 100.        ,  177.827941  ,  316.22776602,  562.34132519])
    >>> np.logspace(2.0, 3.0, num=4, base=2.0)
        array([ 4.        ,  5.0396842 ,  6.34960421,  8.        ])

    Graphical illustration:

    >>> import matplotlib.pyplot as plt
    >>> N = 10
    >>> x1 = np.logspace(0.1, 1, N, endpoint=True)
    >>> x2 = np.logspace(0.1, 1, N, endpoint=False)
    >>> y = np.zeros(N)
    >>> plt.plot(x1, y, 'o')
    >>> plt.plot(x2, y + 0.5, 'o')
    >>> plt.ylim([-0.5, 1])
    >>> plt.show()
    
    """
    # bail if threshold not met
    if num < SIZE_THRESHOLD:
        return np.logspace(start,stop,num,endpoint,base)
    a = ndarray(num)
    step = None
    if endpoint:
        step = (stop-start)/(num-1)
    else:
        step = (stop-start)/num
    buf = a.access()
    if buf is not None:
        lo,hi = a.distribution()
        lo,hi = lo[0],hi[0]
        buf[:] = base**(np.arange(lo,hi)*step+start)
        a.release_update()
    ga.sync()
    return a

def dot(a, b, out=None):
    """dot(a, b)

    Dot product of two arrays.

    For 2-D arrays it is equivalent to matrix multiplication, and for 1-D
    arrays to inner product of vectors (without complex conjugation). For
    N dimensions it is a sum product over the last axis of `a` and
    the second-to-last of `b`::

        dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])

    Parameters
    ----------
    a : array_like
        First argument.
    b : array_like
        Second argument.

    Returns
    -------
    output : ndarray
        Returns the dot product of `a` and `b`.  If `a` and `b` are both
        scalars or both 1-D arrays then a scalar is returned; otherwise
        an array is returned.

    Raises
    ------
    ValueError
        If the last dimension of `a` is not the same size as
        the second-to-last dimension of `b`.

    See Also
    --------
    vdot : Complex-conjugating dot product.
    tensordot : Sum products over arbitrary axes.

    Examples
    --------
    >>> np.dot(3, 4)
    12

    Neither argument is complex-conjugated:

    >>> np.dot([2j, 3j], [2j, 3j])
    (-13+0j)

    For 2-D arrays it's the matrix product:

    >>> a = [[1, 0], [0, 1]]
    >>> b = [[4, 1], [2, 2]]
    >>> np.dot(a, b)
    array([[4, 1],
           [2, 2]])

    >>> a = np.arange(3*4*5*6).reshape((3,4,5,6))
    >>> b = np.arange(3*4*5*6)[::-1].reshape((5,4,6,3))
    >>> np.dot(a, b)[2,3,2,1,2,2]
    499128
    >>> sum(a[2,3,2,:] * b[1,2,:,2])
    499128

    """
    if not (isinstance(a, ndarray) or isinstance(b, ndarray)):
        # numpy pass through
        return np.dot(a,b)
    a = asarray(a)
    b = asarray(b)
    if a.ndim == 1:
        if len(a) != len(b):
            raise ValueError, "objects are not aligned"
        tmp = multiply(a,b)
        a = tmp.access()
        local_sum = None
        if a is None:
            local_sum = np.add.redcue(np.asarray([0], dtype=tmp.dtype))
        else:
            local_sum = np.add.reduce(a)
        return ga.gop_add(local_sum)
    elif a.ndim == 2:
        if a.shape[1] != b.shape[0]:
            raise ValueError, "objects are not aligned"
        out = zeros((a.shape[0],b.shape[1]), a.dtype)
        # use GA gemm if certain conditions apply
        valid_types = [np.dtype(np.float32),
                np.dtype(np.float64),
                np.dtype(np.float128),
                np.dtype(np.complex64),
                np.dtype(np.complex128)]
        if (a.base is None and b.base is None
                and a.dtype == b.dtype and a.dtype in valid_types):
            ga.gemm(False, False, a.shape[0], b.shape[1], b.shape[0],
                    1, a.handle, b.handle, 1, out.handle)
            return out
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

def asarray(a, dtype=None, order=None):
    if isinstance(a, ndarray):
        return a
    else:
        npa = np.asarray(a, dtype=dtype)
        if np.size(npa) > SIZE_THRESHOLD:
            g_a = ndarray(npa.shape, npa.dtype, npa)
            return g_a # distributed using Global Arrays ndarray
        else:
            return npa # scalar or zero rank array

def diag(v, k=0):
    """Extract a diagonal or construct a diagonal array.

    Parameters
    ----------
    v : array_like
        If `v` is a 2-D array, return a copy of its `k`-th diagonal.
        If `v` is a 1-D array, return a 2-D array with `v` on the `k`-th
        diagonal.
    k : int, optional
        Diagonal in question. The default is 0. Use `k>0` for diagonals
        above the main diagonal, and `k<0` for diagonals below the main
        diagonal.

    Returns
    -------
    out : ndarray
        The extracted diagonal or constructed diagonal array.

    See Also
    --------
    diagonal : Return specified diagonals.
    diagflat : Create a 2-D array with the flattened input as a diagonal.
    trace : Sum along diagonals.
    triu : Upper triangle of an array.
    tril : Lower triange of an array.

    Examples
    --------
    >>> x = np.arange(9).reshape((3,3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])

    >>> np.diag(x)
    array([0, 4, 8])
    >>> np.diag(x, k=1)
    array([1, 5])
    >>> np.diag(x, k=-1)
    array([3, 7])

    >>> np.diag(np.diag(x))
    array([[0, 0, 0],
           [0, 4, 0],
           [0, 0, 8]])

    """
    v = asarray(v)
    if isinstance(v, ndarray):
        raise NotImplementedError, "TODO"
        # the following isn't right.
        # We want to scatter the values from the given diagonal into a brand
        # new distributed array, but how to compute the indices for the
        # scatter operation?  Or should we "access" the newly created array
        # and "gather" values from the given diagonal?
        #if v.ndim == 1:
        #    k_fabs = math.fabs(k)
        #    N = k_fabs + len(v)
        #    a = zeros((N,N), dtype=v.dtype)
        #    ndv = v.access()
        #    if ndv is not None:
        #        lo,hi = v.distribution()
        #        count = hi[0]-lo[0]
        #        indices = np.ndarray(count*2,dtype=int)
        #        if k >= 0:
        #            indices[0::2] = np.arange(count)+lo[0]
        #            indices[1::2] = np.arange(count)+lo[0]+k
        #        else:
        #            indices[0::2] = np.arange(count)+lo[0]+k_fabs
        #            indices[1::2] = np.arange(count)+lo[0]
        #        a.scatter(
        #    return a
        #elif v.ndim == 2:
        #    pass
        #else:
        #    raise ValueError, "Input must be 1- or 2-d."
    else:
        return np.diag(v,k)

def print_debug(s):
    if DEBUG:
        print s

def print_sync(what):
    if DEBUG_SYNC:
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
if __name__ != '__main__':
    self_module = sys.modules[__name__]
    for attr in dir(np):
        np_obj = getattr(np, attr)
        if hasattr(self_module, attr):
            if not me: print "gain override exists for: %s" % attr
        else:
            setattr(self_module, attr, getattr(np, attr))
