import math
import sys

from mpi4py import MPI
from ga import ga
import util

import numpy as np
cimport numpy as np

# because it's just useful to have around
me = ga.nodeid()
nproc = ga.nnodes()

# at what point do we distribute arrays versus leaving as np.ndarray?
cdef int SIZE_THRESHOLD = 1
cpdef int get_size_threshold():
    global SIZE_THRESHOLD
    return SIZE_THRESHOLD
def set_size_threshold(int threshold):
    global SIZE_THRESHOLD
    SIZE_THRESHOLD = threshold
cdef bint should_distribute(shape):
    the_shape = shape
    try:
        iter(shape)
    except:
        the_shape = [shape]
    if len(the_shape) == 0:
        return False
    try:
        return np.multiply.reduce(shape) >= get_size_threshold()
    except TypeError:
        # cannot reduce on a scalar, so compare directly
        return shape >= get_size_threshold()
cdef bint is_distributed(thing):
    return isinstance(thing, (ndarray,flatiter))
cdef bint is_array(thing):
    return isinstance(thing, (ndarray,flatiter,np.ndarray,np.flatiter))
cdef _get_shape(thing):
    try:
        return thing.shape # an ndarray
    except AttributeError:
        return (len(thing),) # a flatiter
cdef _get_dtype(thing):
    try:
        return thing.dtype # an ndarray
    except AttributeError:
        return thing.base.dtype # a flatiter

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

cdef inline sync():
    ga.sync()

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

ga_cache1 = {}
ga_cache2 = {}
ga_cache3 = {}

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
    def __new__(cls, shape, dtype=float, buffer=None, offset=0,
            strides=None, order=None, base=None):
        if base is None and not should_distribute(shape):
            return np.ndarray(shape, dtype, buffer, offset, strides, order)
        return super(ndarray, cls).__new__(cls)

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
            dtype_ = self._dtype
            gatype = None
            if dtype_ in gatypes:
                gatype = gatypes[dtype_]
            else:
                gatype = ga.register_dtype(dtype_)
                gatypes[dtype_] = gatype
            #self.handle = ga.create(gatype, shape)
            if (self.shape,self.dtype.num) in ga_cache3:
                self.handle = ga_cache3.pop((self.shape,self.dtype.num))
                #print "acquired from cache", self.shape, self.dtype.num
            elif (self.shape,self.dtype.num) in ga_cache2:
                self.handle = ga_cache2.pop((self.shape,self.dtype.num))
                #print "acquired from cache", self.shape, self.dtype.num
            elif (self.shape,self.dtype.num) in ga_cache1:
                self.handle = ga_cache1.pop((self.shape,self.dtype.num))
                #print "acquired from cache", self.shape, self.dtype.num
            else:
                #print "cache miss"
                self.handle = ga.create(gatype, shape)
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
                    local[:] = a[ga.zip(*self.distribution())]
                    #lo,hi = self.distribution()
                    #a = a[map(lambda x,y: slice(x,y), lo, hi)]
                    #local[:] = a
                    self.release_update()
            #self.global_slice = map(lambda x:slice(0,x,1), shape)
            self.global_slice = [slice(0,x,1) for x in shape]
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

    def __del__(self):
        if self._base is None:
            if ga.initialized():
                #ga.destroy(self.handle)
                if (self.shape,self.dtype.num) in ga_cache3:
                    ga.destroy(self.handle)
                    #print "destroying", self.shape, self.dtype
                elif (self.shape,self.dtype.num) in ga_cache2:
                    ga_cache3[(self.shape,self.dtype.num)] = self.handle
                    #print "destroying", self.shape, self.dtype
                elif (self.shape,self.dtype.num) in ga_cache1:
                    ga_cache2[(self.shape,self.dtype.num)] = self.handle
                    #print "destroying", self.shape, self.dtype
                else:
                    #print "caching", self.shape, self.dtype.num
                    ga_cache1[(self.shape,self.dtype.num)] = self.handle

    ################################################################
    ### ndarray methods added for Global Arrays
    ################################################################
    def distribution(self):
        """Return the bounds of the distribution.

        This operation is local.

        """
        return ga.distribution(self.handle)

    def owns(self):
        """Return True if this process owns some of the data.

        This operation is local.

        """
        lo,hi = self.distribution()
        return np.all(hi>=0)

    def access(self, global_slice=None):
        """Access the local array. Return None if no data is owned.
        
        This operation is local.
        
        """
        if global_slice is None:
            global_slice = self.global_slice
        if self.owns():
            lo,hi = self.distribution()
            access_slice = None
            try:
                access_slice = util.access_slice(global_slice, lo, hi)
            except IndexError:
                pass
            if access_slice:
                a = ga.access(self.handle)
                ret = a[access_slice]
                if self._is_real:
                    ret = ret.real
                elif self._is_imag:
                    ret = ret.imag
                return ret
        return None

    def get(self, key=None):
        """Similar to the __getitem__ built-in, but one-sided (not collective.)

        We sometimes want the semantics of "slicing" an ndarray and then
        immediately calling ga.get() to fetch the result.  We can't use the
        __getitem__ built-in because it is a collective operation.  For
        example, during a ufunc we need to get() the corresponding pieces of
        the arrays and that is where this function is handy.

        This operation is one-sided.

        """
        # first, use the key to create a new global_slice
        # TODO we *might* save a tiny bit of time if we assume the key is
        # already in its canonical form
        # NOTE: ga.get() et al expect either a contiguous 1D array or a
        # buffer with the same shape as the requested region (contiguous or
        # not, but no striding). Since the array may have had dimensions added
        # (via None) or removed, we can't simply create a buffer of the
        # current shape.  Instead, we createa  1D buffer, then reshape it
        # after the call to ga.get() or ga.strided_get().
        global_slice = self.global_slice
        if key is not None:
            key = util.canonicalize_indices(self.shape, key)
            global_slice = util.slice_arithmetic(self.global_slice, key)
        # We must translate global_slice into a strided get
        shape = util.slices_to_shape(global_slice)
        size = np.prod(shape)
        dtype = self._dtype
        if self._is_real or self._is_imag:
            dtype = np.dtype("complex%s" % (self._dtype.itemsize*2*8))
        nd_buffer = np.zeros(size, dtype=dtype)
        _lo = []
        _hi = []
        _skip = []
        adjust = []
        need_strided = False
        for item in global_slice:
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
            elif item is None:
                adjust.append(None)
            else:
                # assumes item is int, long, np.int64, etc
                _lo.append(item)
                _hi.append(item+1)
                _skip.append(1)
        ret = None
        if need_strided:
            ret = ga.strided_get(self.handle, _lo, _hi, _skip, nd_buffer)
        else:
            ret = ga.get(self.handle, _lo, _hi, nd_buffer)
        nd_buffer.shape = shape
        if ret.ndim > 0:
            ret = ret[adjust]
        if self._is_real:
            ret = ret.real
        elif self._is_imag:
            ret = ret.imag
        return ret
        # TODO not sure whether we need to convert 0d arrays to scalars
        #if ret.ndim == 0:
        #    # convert single item to np.generic (scalar)
        #    return ret.dtype.type(ret)

    def allget(self, key=None):
        """Like get(), but when all processes need the same piece.
        
        This operation is collective.
        
        """
        # TODO it's not clear whether this approach is better than having all
        # P processors ga.get() the same piece.
        if not me:
            result = self.get(key)
            return MPI.COMM_WORLD.bcast(result)
        else:
            return MPI.COMM_WORLD.bcast()

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
        return flatiter(self)
    def _set_flat(self, value):
        raise NotImplementedError, "TODO"
    flat = property(_get_flat,_set_flat)

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
        return tuple(self._shape)
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

    ################################################################
    ### ndarray methods
    ################################################################
    def all(self, axis=None, out=None):
        """Returns True if all elements evaluate to True.

        Refer to `numpy.all` for full documentation.

        See Also
        --------
        numpy.all : equivalent function

        """
        raise NotImplementedError

    def any(self, axis=None, out=None):
        """    Returns True if any of the elements of `a` evaluate to True.

        Refer to `numpy.any` for full documentation.

        See Also
        --------
        numpy.any : equivalent function

        """
        raise NotImplementedError

    def argmax(self, axis=None, out=None):
        """Return indices of the maximum values along the given axis.

        Refer to `numpy.argmax` for full documentation.

        See Also
        --------
        numpy.argmax : equivalent function

        """
        raise NotImplementedError

    def argmin(self, axis=None, out=None):
        """Return indices of the minimum values along the given axis of `a`.

        Refer to `numpy.argmin` for detailed documentation.

        See Also
        --------
        numpy.argmin : equivalent function

        """
        raise NotImplementedError

    def argsort(self, axis=-1, kind='quicksort', order=None):
        """Returns the indices that would sort this array.

        Refer to `numpy.argsort` for full documentation.

        See Also
        --------
        numpy.argsort : equivalent function

        """
        raise NotImplementedError

    def astype(self, t):
        """Copy of the array, cast to a specified type.

        Parameters
        ----------
        t : string or dtype
            Typecode or data-type to which the array is cast.

        Examples
        --------
        >>> x = np.array([1, 2, 2.5])
        >>> x
        array([ 1. ,  2. ,  2.5])

        >>> x.astype(int)
        array([1, 2, 2])

        """
        raise NotImplementedError

    def byteswap(self, inplace=False):
        """Swap the bytes of the array elements

        Toggle between low-endian and big-endian data representation by
        returning a byteswapped array, optionally swapped in-place.

        Parameters
        ----------
        inplace: bool, optional
            If ``True``, swap bytes in-place, default is ``False``.

        Returns
        -------
        out: ndarray
            The byteswapped array. If `inplace` is ``True``, this is
            a view to self.

        Examples
        --------
        >>> A = np.array([1, 256, 8755], dtype=np.int16)
        >>> map(hex, A)
        ['0x1', '0x100', '0x2233']
        >>> A.byteswap(True)
        array([  256,     1, 13090], dtype=int16)
        >>> map(hex, A)
        ['0x100', '0x1', '0x3322']

        Arrays of strings are not swapped

        >>> A = np.array(['ceg', 'fac'])
        >>> A.byteswap()
        array(['ceg', 'fac'],
              dtype='|S3')
            
        """
        raise NotImplementedError

    def choose(self, choices, out=None, mode='raise'):
        """Use an index array to construct a new array from a set of choices.

        Refer to `numpy.choose` for full documentation.

        See Also
        --------
        numpy.choose : equivalent function

        """
        raise NotImplementedError

    def clip(self, a_min, a_max, out=None):
        """Use an index array to construct a new array from a set of choices.

        Refer to `numpy.choose` for full documentation.

        See Also
        --------
        numpy.choose : equivalent function

        """
        raise NotImplementedError

    def compress(self, condition, axis=None, out=None):
        """Return selected slices of this array along given axis.

        Refer to `numpy.compress` for full documentation.

        See Also
        --------
        numpy.compress : equivalent function

        """
        return NotImplementedError

    def conj(self):
        """Complex-conjugate all elements.

        Refer to `numpy.conjugate` for full documentation.

        See Also
        --------
        numpy.conjugate : equivalent function

        """
        raise NotImplementedError

    def conjugate(self):
        """Return the complex conjugate, element-wise.

        Refer to `numpy.conjugate` for full documentation.

        See Also
        --------
        numpy.conjugate : equivalent function

        """
        raise NotImplementedError

    def copy(self, order='C'):
        """Return a copy of the array.

        Parameters
        ----------
        order : {'C', 'F', 'A'}, optional
            By default, the result is stored in C-contiguous (row-major) order in
            memory.  If `order` is `F`, the result has 'Fortran' (column-major)
            order.  If order is 'A' ('Any'), then the result has the same order
            as the input.

        Examples
        --------
        >>> x = np.array([[1,2,3],[4,5,6]], order='F')

        >>> y = x.copy()

        >>> x.fill(0)

        >>> x
        array([[0, 0, 0],
               [0, 0, 0]])

        >>> y
        array([[1, 2, 3],
               [4, 5, 6]])

        >>> y.flags['C_CONTIGUOUS']
        True

        """
        # TODO we can optimize a copy if new and old ndarray instances align
        the_copy = ndarray(self.shape, dtype=self.dtype)
        if should_distribute(the_copy.size):
            local = the_copy.access()
            if local is not None:
                lo,hi = the_copy.distribution()
                local[:] = self.get(ga.zip(lo,hi))
                the_copy.release_update()
        else:
            # case where the copy is not distributed but the original was
            the_copy[:] = self.allget()
        return the_copy

    def cumprod(self, axis=None, dtype=None, out=None):
        """Return the cumulative product of the elements along the given axis.

        Refer to `numpy.cumprod` for full documentation.

        See Also
        --------
        numpy.cumprod : equivalent function

        """
        raise NotImplementedError

    def cumsum(self, axis=None, dtype=None, out=None):
        """Return the cumulative sum of the elements along the given axis.

        Refer to `numpy.cumsum` for full documentation.

        See Also
        --------
        numpy.cumsum : equivalent function

        """
        raise NotImplementedError

    def diagonal(self, offset=0, axis1=0, axis2=1):
        """Return specified diagonals.

        Refer to `numpy.diagonal` for full documentation.

        See Also
        --------
        numpy.diagonal : equivalent function

        """
        raise NotImplementedError

    def dot(self):
        raise NotImplementedError

    def dump(self, file):
        """Dump a pickle of the array to the specified file.

        The array can be read back with pickle.load or numpy.load.

        Parameters
        ----------
        file : str
            A string naming the dump file.

        """
        raise NotImplementedError

    def dumps(self):
        """Returns the pickle of the array as a string.

        pickle.loads or numpy.loads will convert the string back to an array.

        Parameters
        ----------
        None

        """
        raise NotImplementedError

    def fill(self, value):
        """Fill the array with a scalar value.

        Parameters
        ----------
        value : scalar
            All elements of `a` will be assigned this value.

        Examples
        --------
        >>> a = np.array([1, 2])
        >>> a.fill(0)
        >>> a
        array([0, 0])
        >>> a = np.empty(2)
        >>> a.fill(1)
        >>> a
        array([ 1.,  1.])

        """
        raise NotImplementedError

    def flatten(self, order='C'):
        """Return a copy of the array collapsed into one dimension.

        Parameters
        ----------
        order : {'C', 'F'}, optional
            Whether to flatten in C (row-major) or Fortran (column-major) order.
            The default is 'C'.

        Returns
        -------
        y : ndarray
            A copy of the input array, flattened to one dimension.

        See Also
        --------
        ravel : Return a flattened array.
        flat : A 1-D flat iterator over the array.

        Examples
        --------
        >>> a = np.array([[1,2], [3,4]])
        >>> a.flatten()
        array([1, 2, 3, 4])
        >>> a.flatten('F')
        array([1, 3, 2, 4])

        """
        raise NotImplementedError

    def getfield(self, dtype, offset):
        """Returns a field of the given array as a certain type.

        A field is a view of the array data with each itemsize determined
        by the given type and the offset into the current array, i.e. from
        ``offset * dtype.itemsize`` to ``(offset+1) * dtype.itemsize``.

        Parameters
        ----------
        dtype : str
            String denoting the data type of the field.
        offset : int
            Number of `dtype.itemsize`'s to skip before beginning the element view.

        Examples
        --------
        >>> x = np.diag([1.+1.j]*2)
        >>> x
        array([[ 1.+1.j,  0.+0.j],
               [ 0.+0.j,  1.+1.j]])
        >>> x.dtype
        dtype('complex128')

        >>> x.getfield('complex64', 0) # Note how this != x
        array([[ 0.+1.875j,  0.+0.j   ],
               [ 0.+0.j   ,  0.+1.875j]], dtype=complex64)

        >>> x.getfield('complex64',1) # Note how different this is than x
        array([[ 0. +5.87173204e-39j,  0. +0.00000000e+00j],
               [ 0. +0.00000000e+00j,  0. +5.87173204e-39j]], dtype=complex64)

        >>> x.getfield('complex128', 0) # == x
        array([[ 1.+1.j,  0.+0.j],
               [ 0.+0.j,  1.+1.j]])

        If the argument dtype is the same as x.dtype, then offset != 0 raises
        a ValueError:

        >>> x.getfield('complex128', 1)
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
        ValueError: Need 0 <= offset <= 0 for requested type but received offset = 1

        >>> x.getfield('float64', 0)
        array([[ 1.,  0.],
               [ 0.,  1.]])

        >>> x.getfield('float64', 1)
        array([[  1.77658241e-307,   0.00000000e+000],
               [  0.00000000e+000,   1.77658241e-307]])

        """
        raise NotImplementedError

    def item(self, *args):
        """Copy an element of an array to a standard Python scalar and return it.

        Parameters
        ----------
        \*args : Arguments (variable number and type)

            * none: in this case, the method only works for arrays
              with one element (`a.size == 1`), which element is
              copied into a standard Python scalar object and returned.

            * int_type: this argument is interpreted as a flat index into
              the array, specifying which element to copy and return.

            * tuple of int_types: functions as does a single int_type argument,
              except that the argument is interpreted as an nd-index into the
              array.

        Returns
        -------
        z : Standard Python scalar object
            A copy of the specified element of the array as a suitable
            Python scalar

        Notes
        -----
        When the data type of `a` is longdouble or clongdouble, item() returns
        a scalar array object because there is no available Python scalar that
        would not lose information. Void arrays return a buffer object for item(),
        unless fields are defined, in which case a tuple is returned.

        `item` is very similar to a[args], except, instead of an array scalar,
        a standard Python scalar is returned. This can be useful for speeding up
        access to elements of the array and doing arithmetic on elements of the
        array using Python's optimized math.

        Examples
        --------
        >>> x = np.random.randint(9, size=(3, 3))
        >>> x
        array([[3, 1, 7],
               [2, 8, 3],
               [8, 5, 3]])
        >>> x.item(3)
        2
        >>> x.item(7)
        5
        >>> x.item((0, 1))
        1
        >>> x.item((2, 2))
        3

        """
        raise NotImplementedError

    def itemset(self, *args):
        """Insert scalar into an array (scalar is cast to array's dtype, if possible)

        There must be at least 1 argument, and define the last argument
        as *item*.  Then, ``a.itemset(*args)`` is equivalent to but faster
        than ``a[args] = item``.  The item should be a scalar value and `args`
        must select a single item in the array `a`.

        Parameters
        ----------
        \*args : Arguments
            If one argument: a scalar, only used in case `a` is of size 1.
            If two arguments: the last argument is the value to be set
            and must be a scalar, the first argument specifies a single array
            element location. It is either an int or a tuple.

        Notes
        -----
        Compared to indexing syntax, `itemset` provides some speed increase
        for placing a scalar into a particular location in an `ndarray`,
        if you must do this.  However, generally this is discouraged:
        among other problems, it complicates the appearance of the code.
        Also, when using `itemset` (and `item`) inside a loop, be sure
        to assign the methods to a local variable to avoid the attribute
        look-up at each loop iteration.

        Examples
        --------
        >>> x = np.random.randint(9, size=(3, 3))
        >>> x
        array([[3, 1, 7],
               [2, 8, 3],
               [8, 5, 3]])
        >>> x.itemset(4, 0)
        >>> x.itemset((2, 2), 9)
        >>> x
        array([[3, 1, 7],
               [2, 0, 3],
               [8, 5, 9]])

        """
        raise NotImplementedError

    def max(self, axis=None, out=None):
        """Return the maximum along a given axis.

        Refer to `numpy.amax` for full documentation.

        See Also
        --------
        numpy.amax : equivalent function

        """
        raise NotImplementedError

    def mean(self, axis=None, dtype=None, out=None):
        """Returns the average of the array elements along given axis.

        Refer to `numpy.mean` for full documentation.

        See Also
        --------
        numpy.mean : equivalent function

        """
        raise NotImplementedError

    def min(self, axis=None, out=None):
        """Return the minimum along a given axis.

        Refer to `numpy.amin` for full documentation.

        See Also
        --------
        numpy.amin : equivalent function

        """
        raise NotImplementedError

    def newbyteorder(self, new_order='S'):
        """Return the array with the same data viewed with a different byte order.

        Equivalent to::

            arr.view(arr.dtype.newbytorder(new_order))

        Changes are also made in all fields and sub-arrays of the array data
        type.



        Parameters
        ----------
        new_order : string, optional
            Byte order to force; a value from the byte order specifications
            above. `new_order` codes can be any of::

             * 'S' - swap dtype from current to opposite endian
             * {'<', 'L'} - little endian
             * {'>', 'B'} - big endian
             * {'=', 'N'} - native order
             * {'|', 'I'} - ignore (no change to byte order)

            The default value ('S') results in swapping the current
            byte order. The code does a case-insensitive check on the first
            letter of `new_order` for the alternatives above.  For example,
            any of 'B' or 'b' or 'biggish' are valid to specify big-endian.


        Returns
        -------
        new_arr : array
            New array object with the dtype reflecting given change to the
            byte order.

        """
        raise NotImplementedError

    def nonzero(self):
        """Return the indices of the elements that are non-zero.

        Refer to `numpy.nonzero` for full documentation.

        See Also
        --------
        numpy.nonzero : equivalent function

        """
        raise NotImplementedError

    def prod(self, axis=None, dtype=None, out=None):
        """Return the product of the array elements over the given axis

        Refer to `numpy.prod` for full documentation.

        See Also
        --------
        numpy.prod : equivalent function

        """
        raise NotImplementedError

    def ptp(self, axis=None, out=None):
        """Peak to peak (maximum - minimum) value along a given axis.

        Refer to `numpy.ptp` for full documentation.

        See Also
        --------
        numpy.ptp : equivalent function

        """
        raise NotImplementedError

    def put(self, indices, values, mode='raise'):
        """Set ``a.flat[n] = values[n]`` for all `n` in indices.

        Refer to `numpy.put` for full documentation.

        See Also
        --------
        numpy.put : equivalent function
        
        """
        raise NotImplementedError

    def ravel(self, order=None):
        """Return a flattened array.

        Refer to `numpy.ravel` for full documentation.

        See Also
        --------
        numpy.ravel : equivalent function

        ndarray.flat : a flat iterator on the array.

        """
        raise NotImplementedError

    def repeat(self, repeats, axis=None):
        """Repeat elements of an array.

        Refer to `numpy.repeat` for full documentation.

        See Also
        --------
        numpy.repeat : equivalent function

        """
        raise NotImplementedError

    def reshape(self, shape, order='C'):
        """Returns an array containing the same data with a new shape.

        Refer to `numpy.reshape` for full documentation.

        See Also
        --------
        numpy.reshape : equivalent function

        """
        raise NotImplementedError

    def resize(self, new_shape, refcheck=True):
        """Change shape and size of array in-place.

        Parameters
        ----------
        new_shape : tuple of ints, or `n` ints
            Shape of resized array.
        refcheck : bool, optional
            If False, reference count will not be checked. Default is True.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `a` does not own its own data or references or views to it exist,
            and the data memory must be changed.

        SystemError
            If the `order` keyword argument is specified. This behaviour is a
            bug in NumPy.

        See Also
        --------
        resize : Return a new array with the specified shape.

        Notes
        -----
        This reallocates space for the data area if necessary.

        Only contiguous arrays (data elements consecutive in memory) can be
        resized.

        The purpose of the reference count check is to make sure you
        do not use this array as a buffer for another Python object and then
        reallocate the memory. However, reference counts can increase in
        other ways so if you are sure that you have not shared the memory
        for this array with another Python object, then you may safely set
        `refcheck` to False.

        Examples
        --------
        Shrinking an array: array is flattened (in the order that the data are
        stored in memory), resized, and reshaped:

        >>> a = np.array([[0, 1], [2, 3]], order='C')
        >>> a.resize((2, 1))
        >>> a
        array([[0],
               [1]])

        >>> a = np.array([[0, 1], [2, 3]], order='F')
        >>> a.resize((2, 1))
        >>> a
        array([[0],
               [2]])

        Enlarging an array: as above, but missing entries are filled with zeros:

        >>> b = np.array([[0, 1], [2, 3]])
        >>> b.resize(2, 3) # new_shape parameter doesn't have to be a tuple
        >>> b
        array([[0, 1, 2],
               [3, 0, 0]])

        Referencing an array prevents resizing...

        >>> c = a
        >>> a.resize((1, 1))
        Traceback (most recent call last):
        ...
        ValueError: cannot resize an array that has been referenced ...

        Unless `refcheck` is False:

        >>> a.resize((1, 1), refcheck=False)
        >>> a
        array([[0]])
        >>> c
        array([[0]])

        """
        raise NotImplementedError

    def round(self, decimals=0, out=None):
        """Return `a` with each element rounded to the given number of decimals.

        Refer to `numpy.around` for full documentation.

        See Also
        --------
        numpy.around : equivalent function

        """
        raise NotImplementedError

    def searchsorted(self, v, side='left'):
        """Find indices where elements of v should be inserted in a to maintain order.

        For full documentation, see `numpy.searchsorted`

        See Also
        --------
        numpy.searchsorted : equivalent function

        """
        raise NotImplementedError

    def setfield(self, val, dtype, offset=0):
        """Put a value into a specified place in a field defined by a data-type.

        Place `val` into `a`'s field defined by `dtype` and beginning `offset`
        bytes into the field.

        Parameters
        ----------
        val : object
            Value to be placed in field.
        dtype : dtype object
            Data-type of the field in which to place `val`.
        offset : int, optional
            The number of bytes into the field at which to place `val`.

        Returns
        -------
        None

        See Also
        --------
        getfield

        Examples
        --------
        >>> x = np.eye(3)
        >>> x.getfield(np.float64)
        array([[ 1.,  0.,  0.],
               [ 0.,  1.,  0.],
               [ 0.,  0.,  1.]])
        >>> x.setfield(3, np.int32)
        >>> x.getfield(np.int32)
        array([[3, 3, 3],
               [3, 3, 3],
               [3, 3, 3]])
        >>> x
        array([[  1.00000000e+000,   1.48219694e-323,   1.48219694e-323],
               [  1.48219694e-323,   1.00000000e+000,   1.48219694e-323],
               [  1.48219694e-323,   1.48219694e-323,   1.00000000e+000]])
        >>> x.setfield(np.eye(3), np.int32)
        >>> x
        array([[ 1.,  0.,  0.],
               [ 0.,  1.,  0.],
               [ 0.,  0.,  1.]])

        """
        raise NotImplementedError

    def setflags(self, write=None, align=None, uic=None):
        """Set array flags WRITEABLE, ALIGNED, and UPDATEIFCOPY, respectively.

        These Boolean-valued flags affect how numpy interprets the memory
        area used by `a` (see Notes below). The ALIGNED flag can only
        be set to True if the data is actually aligned according to the type.
        The UPDATEIFCOPY flag can never be set to True. The flag WRITEABLE
        can only be set to True if the array owns its own memory, or the
        ultimate owner of the memory exposes a writeable buffer interface,
        or is a string. (The exception for string is made so that unpickling
        can be done without copying memory.)

        Parameters
        ----------
        write : bool, optional
            Describes whether or not `a` can be written to.
        align : bool, optional
            Describes whether or not `a` is aligned properly for its type.
        uic : bool, optional
            Describes whether or not `a` is a copy of another "base" array.

        Notes
        -----
        Array flags provide information about how the memory area used
        for the array is to be interpreted. There are 6 Boolean flags
        in use, only three of which can be changed by the user:
        UPDATEIFCOPY, WRITEABLE, and ALIGNED.

        WRITEABLE (W) the data area can be written to;

        ALIGNED (A) the data and strides are aligned appropriately for the hardware
        (as determined by the compiler);

        UPDATEIFCOPY (U) this array is a copy of some other array (referenced
        by .base). When this array is deallocated, the base array will be
        updated with the contents of this array.

        All flags can be accessed using their first (upper case) letter as well
        as the full name.

        Examples
        --------
        >>> y
        array([[3, 1, 7],
               [2, 0, 0],
               [8, 5, 9]])
        >>> y.flags
          C_CONTIGUOUS : True
          F_CONTIGUOUS : False
          OWNDATA : True
          WRITEABLE : True
          ALIGNED : True
          UPDATEIFCOPY : False
        >>> y.setflags(write=0, align=0)
        >>> y.flags
          C_CONTIGUOUS : True
          F_CONTIGUOUS : False
          OWNDATA : True
          WRITEABLE : False
          ALIGNED : False
          UPDATEIFCOPY : False
        >>> y.setflags(uic=1)
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
        ValueError: cannot set UPDATEIFCOPY flag to True

        """
        raise NotImplementedError

    def sort(self, axis=-1, kind='quicksort', order=None):
        """Sort an array, in-place.

        Parameters
        ----------
        axis : int, optional
            Axis along which to sort. Default is -1, which means sort along the
            last axis.
        kind : {'quicksort', 'mergesort', 'heapsort'}, optional
            Sorting algorithm. Default is 'quicksort'.
        order : list, optional
            When `a` is an array with fields defined, this argument specifies
            which fields to compare first, second, etc.  Not all fields need be
            specified.

        See Also
        --------
        numpy.sort : Return a sorted copy of an array.
        argsort : Indirect sort.
        lexsort : Indirect stable sort on multiple keys.
        searchsorted : Find elements in sorted array.

        Notes
        -----
        See ``sort`` for notes on the different sorting algorithms.

        Examples
        --------
        >>> a = np.array([[1,4], [3,1]])
        >>> a.sort(axis=1)
        >>> a
        array([[1, 4],
               [1, 3]])
        >>> a.sort(axis=0)
        >>> a
        array([[1, 3],
               [1, 4]])

        Use the `order` keyword to specify a field to use when sorting a
        structured array:

        >>> a = np.array([('a', 2), ('c', 1)], dtype=[('x', 'S1'), ('y', int)])
        >>> a.sort(order='y')
        >>> a
        array([('c', 1), ('a', 2)],
              dtype=[('x', '|S1'), ('y', '<i4')])

        """
        raise NotImplementedError

    def squeeze(self):
        """Remove single-dimensional entries from the shape of `a`.

        Refer to `numpy.squeeze` for full documentation.

        See Also
        --------
        numpy.squeeze : equivalent function

        """
        raise NotImplementedError

    def std(self, axis=None, dtype=None, out=None, ddof=0):
        """Remove single-dimensional entries from the shape of `a`.

        Refer to `numpy.squeeze` for full documentation.

        See Also
        --------
        numpy.squeeze : equivalent function

        """
        raise NotImplementedError

    def sum(self, axis=None, dtype=None, out=None):
        """Return the sum of the array elements over the given axis.

        Refer to `numpy.sum` for full documentation.

        See Also
        --------
        numpy.sum : equivalent function

        """
        if axis is None:
            local = self.access()
            if local is not None:
                value = np.sum(local)
                value = MPI.COMM_WORLD.allreduce(value, MPI.SUM)
            else:
                value = MPI.COMM_WORLD.allreduce(0, MPI.SUM)
            return value
        else:
            raise NotImplementedError

    def swapaxes(self, axis1, axis2):
        """Return a view of the array with `axis1` and `axis2` interchanged.

        Refer to `numpy.swapaxes` for full documentation.

        See Also
        --------
        numpy.swapaxes : equivalent function

        """
        raise NotImplementedError

    def take(self, indices, axis=None, out=None, mode='raise'):
        """Return an array formed from the elements of `a` at the given indices.

        Refer to `numpy.take` for full documentation.

        See Also
        --------
        numpy.take : equivalent function

        """
        raise NotImplementedError

    def tofile(self, fid, sep="", format="%s"):
        """Write array to a file as text or binary (default).

        Data is always written in 'C' order, independent of the order of `a`.
        The data produced by this method can be recovered using the function
        fromfile().

        Parameters
        ----------
        fid : file or str
            An open file object, or a string containing a filename.
        sep : str
            Separator between array items for text output.
            If "" (empty), a binary file is written, equivalent to
            ``file.write(a.tostring())``.
        format : str
            Format string for text file output.
            Each entry in the array is formatted to text by first converting
            it to the closest Python type, and then using "format" % item.

        Notes
        -----
        This is a convenience function for quick storage of array data.
        Information on endianness and precision is lost, so this method is not a
        good choice for files intended to archive data or transport data between
        machines with different endianness. Some of these problems can be overcome
        by outputting the data as text files, at the expense of speed and file
        size.

        """
        raise NotImplementedError

    def tolist(self):
        """Return the array as a (possibly nested) list.

        Return a copy of the array data as a (nested) Python list.
        Data items are converted to the nearest compatible Python type.

        Parameters
        ----------
        none

        Returns
        -------
        y : list
            The possibly nested list of array elements.

        Notes
        -----
        The array may be recreated, ``a = np.array(a.tolist())``.

        Examples
        --------
        >>> a = np.array([1, 2])
        >>> a.tolist()
        [1, 2]
        >>> a = np.array([[1, 2], [3, 4]])
        >>> list(a)
        [array([1, 2]), array([3, 4])]
        >>> a.tolist()
        [[1, 2], [3, 4]]

        """
        raise NotImplementedError

    def tostring(self, order='C'):
        """Construct a Python string containing the raw data bytes in the array.

        Constructs a Python string showing a copy of the raw contents of
        data memory. The string can be produced in either 'C' or 'Fortran',
        or 'Any' order (the default is 'C'-order). 'Any' order means C-order
        unless the F_CONTIGUOUS flag in the array is set, in which case it
        means 'Fortran' order.

        Parameters
        ----------
        order : {'C', 'F', None}, optional
            Order of the data for multidimensional arrays:
            C, Fortran, or the same as for the original array.

        Returns
        -------
        s : str
            A Python string exhibiting a copy of `a`'s raw data.

        Examples
        --------
        >>> x = np.array([[0, 1], [2, 3]])
        >>> x.tostring()
        '\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00'
        >>> x.tostring('C') == x.tostring()
        True
        >>> x.tostring('F')
        '\x00\x00\x00\x00\x02\x00\x00\x00\x01\x00\x00\x00\x03\x00\x00\x00'

        """
        raise NotImplementedError

    def trace(self, offset=0, axis1=0, axis2=1, dtype=None, out=None):
        """Return the sum along diagonals of the array.

        Refer to `numpy.trace` for full documentation.

        See Also
        --------
        numpy.trace : equivalent function

        """
        raise NotImplementedError

    def transpose(*axes):
        """Returns a view of the array with axes transposed.

        For a 1-D array, this has no effect. (To change between column and
        row vectors, first cast the 1-D array into a matrix object.)
        For a 2-D array, this is the usual matrix transpose.
        For an n-D array, if axes are given, their order indicates how the
        axes are permuted (see Examples). If axes are not provided and
        ``a.shape = (i[0], i[1], ... i[n-2], i[n-1])``, then
        ``a.transpose().shape = (i[n-1], i[n-2], ... i[1], i[0])``.

        Parameters
        ----------
        axes : None, tuple of ints, or `n` ints

         * None or no argument: reverses the order of the axes.

         * tuple of ints: `i` in the `j`-th place in the tuple means `a`'s
           `i`-th axis becomes `a.transpose()`'s `j`-th axis.

         * `n` ints: same as an n-tuple of the same ints (this form is
           intended simply as a "convenience" alternative to the tuple form)

        Returns
        -------
        out : ndarray
            View of `a`, with axes suitably permuted.

        See Also
        --------
        ndarray.T : Array property returning the array transposed.

        Examples
        --------
        >>> a = np.array([[1, 2], [3, 4]])
        >>> a
        array([[1, 2],
               [3, 4]])
        >>> a.transpose()
        array([[1, 3],
               [2, 4]])
        >>> a.transpose((1, 0))
        array([[1, 3],
               [2, 4]])
        >>> a.transpose(1, 0)
        array([[1, 3],
               [2, 4]])

        """
        raise NotImplementedError

    def var(self, axis=None, dtype=None, out=None, ddof=0):
        """Returns the variance of the array elements, along given axis.

        Refer to `numpy.var` for full documentation.

        See Also
        --------
        numpy.var : equivalent function

        """
        raise NotImplementedError

    def view(self, dtype=None, type=None):
        """New view of array with the same data.

        Parameters
        ----------
        dtype : data-type, optional
            Data-type descriptor of the returned view, e.g., float32 or int16.
            The default, None, results in the view having the same data-type
            as `a`.
        type : Python type, optional
            Type of the returned view, e.g., ndarray or matrix.  Again, the
            default None results in type preservation.

        Notes
        -----
        ``a.view()`` is used two different ways:

        ``a.view(some_dtype)`` or ``a.view(dtype=some_dtype)`` constructs a view
        of the array's memory with a different data-type.  This can cause a
        reinterpretation of the bytes of memory.

        ``a.view(ndarray_subclass)`` or ``a.view(type=ndarray_subclass)`` just
        returns an instance of `ndarray_subclass` that looks at the same array
        (same shape, dtype, etc.)  This does not cause a reinterpretation of the
        memory.


        Examples
        --------
        >>> x = np.array([(1, 2)], dtype=[('a', np.int8), ('b', np.int8)])

        Viewing array data using a different type and dtype:

        >>> y = x.view(dtype=np.int16, type=np.matrix)
        >>> y
        matrix([[513]], dtype=int16)
        >>> print type(y)
        <class 'numpy.matrixlib.defmatrix.matrix'>

        Creating a view on a structured array so it can be used in calculations

        >>> x = np.array([(1, 2),(3,4)], dtype=[('a', np.int8), ('b', np.int8)])
        >>> xv = x.view(dtype=np.int8).reshape(-1,2)
        >>> xv
        array([[1, 2],
               [3, 4]], dtype=int8)
        >>> xv.mean(0)
        array([ 2.,  3.])

        Making changes to the view changes the underlying array

        >>> xv[0,1] = 20
        >>> print x
        [(1, 20) (3, 4)]

        Using a view to convert an array to a record array:

        >>> z = x.view(np.recarray)
        >>> z.a
        array([1], dtype=int8)

        Views share data:

        >>> x[0] = (9, 10)
        >>> z[0]
        (9, 10)

        """
        raise NotImplementedError

    ################################################################
    ### ndarray operator overloading
    ################################################################
    def __abs__(self):
        return abs(self)

    def __add__(self, y):
        return add(self,y)

    def __and__(self, y):
        return logical_and(self,y)

    def __array__(self, dtype):
        raise NotImplementedError

    def __array_finalize__(self, *args, **kwargs):
        raise NotImplementedError

    def __array_interface__(self, *args, **kwargs):
        raise NotImplementedError

    def __array_prepare__(self, *args, **kwargs):
        raise NotImplementedError

    def __array_priority__(self, *args, **kwargs):
        raise NotImplementedError

    def __array_struct__(self, *args, **kwargs):
        raise NotImplementedError

    def __array_wrap__(self, *args, **kwargs):
        raise NotImplementedError

    def __contains__(self, y):
        raise NotImplementedError

    def __copy__(self, order=None):
        """Return a copy of the array.

    Parameters
    ----------
    order : {'C', 'F', 'A'}, optional
        If order is 'C' (False) then the result is contiguous (default).
        If order is 'Fortran' (True) then the result has fortran order.
        If order is 'Any' (None) then the result has fortran order
        only if the array already is in fortran order.

        """
        raise NotImplementedError

    def __deepcopy__(self):
        raise NotImplementedError

    #def __delattr__

    def __delitem__(self, *args, **kwargs):
        raise ValueError, "cannot delete array elements"

    #def __delslice__

    def __div__(self, y):
        return divide(self,y)

    def __divmod__(self, y):
        t = mod(self,y)
        s = subtract(self,t)
        s = divide(s,y,s)
        return s,t

    def __eq__(self, y):
        return equal(self,y)

    def __float__(self):
        raise NotImplementedError

    def __floordiv__(self,y):
        return floor_divide(self,y)

    #def __format__

    def __ge__(self,y):
        return greater_equal(self,y)

    #def __getattribute__

    def __getitem__(self, key):
        # THIS IS A COLLECTIVE OPERATION
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
        if a.ndim == 0:
            a = a.allget()
            return a.dtype.type(a) # convert single item to np.generic (scalar)
        return a

    #def __getslice__

    def __gt__(self,y):
        return greater(self,y)

    #def __hash__
    #def __hex__

    def __iadd__(self,y):
        add(self,y,self)

    def __iand__(self,y):
        logical_and(self,y,self)

    def __idiv__(self,y):
        divide(self,y,self)

    def __ifloordiv__(self,y):
        floor_divide(self,y,self)

    def __ilshift__(self,y):
        left_shift(self,y,self)

    def __imod__(self,y):
        mod(self,y,self)

    def __imul__(self,y):
        multiply(self,y,self)

    #def __index__

    def __int__(self, *args, **kwargs):
        raise NotImplementedError

    def __invert__(self):
        return invert(self)

    def __ior__(self,y):
        logical_or(self,y,self)

    def __ipow__(self,y):
        power(self,y,self)

    def __irshift__(self,y):
        right_shift(self,y,self)

    def __isub__(self,y):
        subtract(self,y,self)

    def __iter__(self, *args, **kwargs):
        raise NotImplementedError

    def __itruediv__(self,y):
        true_divide(self,y,self)

    def __ixor__(self,y):
        logical_xor(self,y,self)

    def __le__(self,y):
        return less_equal(self,y)

    def __len__(self):
        return self.shape[0]

    def __long__(self, *args, **kwargs):
        raise NotImplementedError

    def __lshift__(self,y):
        return left_shift(self,y)

    def __lt__(self,y):
        return less(self,y)

    def __mod__(self,y):
        return mod(self,y)

    def __mul__(self,y):
        return multiply(self,y)

    def __ne__(self,y):
        return not_equal(self,y)

    def __neg__(self):
        return negative(self)

    def __nonzero__(self):
        raise ValueError, ("The truth value of an array with more than one"
            " element is ambiguous. Use a.any() or a.all()")

    def __oct__(self, *args, **kwargs):
        raise NotImplementedError

    def __or__(self,y):
        return logical_or(self,y)

    def __pos__(self):
        return self.copy()

    def __pow__(self,y):
        return power(self,y)

    def __radd__(self,y):
        return add(y,self)

    def __rand__(self,y):
        return logical_and(y,self)

    def __rdiv__(self,y):
        return divide(y,self)

    def __rdivmod__(self,y):
        t = mod(y,self)
        s = subtract(y,t)
        s = divide(s,self,s)
        return s,t

    def __reduce__(self, *args, **kwargs):
        raise NotImplementedError

    def __reduce_ex__(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        result = ""
        if 0 == me:
            result = repr(self.get())
        return result

    def __rfloordiv__(self,y):
        return floor_divide(y,self)

    def __rlshift__(self,y):
        return left_shift(y,self)

    def __rmod__(self,y):
        return mod(y,self)

    def __rmul__(self,y):
        return multiply(y,self)

    def __ror__(self,y):
        return logical_or(y,self)

    def __rpow__(self,y):
        return power(y,self)

    def __rrshift__(self,y):
        return right_shift(y,self)

    def __rshift__(self,y):
        return right_shift(self,y)

    def __rsub__(self,y):
        return subtract(y,self)

    def __rtruediv__(self,y):
        return true_divide(y,self)

    def __rxor__(self,y):
        return logical_xor(y,self)

    #def __setattr__

    def __setitem__(self, key, value):
        # THIS IS A COLLECTIVE OPERATION
        sync()
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
        global_slice = util.slice_arithmetic(self.global_slice, key)
        value = asarray(value)
        npvalue = None
        release_value = False
        # access based on new global_slice as an ndarray first
        npself = self.access(global_slice)
        if npself is not None:
            if isinstance(value, ndarray):
                if (ga.compare_distr(value.handle, self.handle)
                        and value.global_slice == global_slice):
                    # opt: same distributions and same slicing
                    # in practice this might not happen all that often
                    npvalue = value.access()
                    release_value = True
                else:
                    lo,hi = self.distribution()
                    result = util.get_slice(global_slice, lo, hi)
                    result = util.broadcast_chomp(value.shape, result)
                    npvalue = value.get(result)
            elif isinstance(value, flatiter):
                raise NotImplementedError
            elif value.ndim > 0:
                    lo,hi = self.distribution()
                    result = util.get_slice(global_slice, lo, hi)
                    result = util.broadcast_chomp(value.shape, result)
                    npvalue = value[result]
            else:
                npvalue = value
            npself[:] = npvalue
        sync()

    #def __setslice__
    #def __setstate__
    #def __sizeof__
    
    def __str__(self):
        result = ""
        if 0 == me:
            result = str(self.get())
        return result

    def __sub__(self,y):
        return subtract(self,y)

    #def __subclasshook__

    def __truediv__(self,y):
        return true_divide(self,y)

    def __xor__(self,y):
        return logical_xor(self,y)

def _npin_piece_based_on_out(input, out, shape=None):
    # opt: same distributions and same slicing
    #   we can use local data exclusively
    #   in practice this might not happen all that often
    if (isinstance(input, ndarray)
            and ga.compare_distr(input.handle, out.handle)
            and input.global_slice == out.global_slice):
        return input.access(),True
    # no opt: requires copy of remote data
    elif shape is None or len(shape) > 0:
        lo,hi = out.distribution()
        result = util.get_slice(out.global_slice, lo, hi)
        if shape is not None:
            result = util.broadcast_chomp(shape, result)
        if is_distributed(input):
            return input.get(result),False
        else:
            return input[result],False
    else:
        return input,False

class ufunc(object):
    """Functions that operate element by element on whole arrays.

    A detailed explanation of ufuncs can be found in the "ufuncs.rst"
    file in the NumPy reference guide.

    Unary ufuncs:
    =============

    op(X, out=None)
    Apply op to X elementwise

    Parameters
    ----------
    X : array_like
        Input array.
    out : array_like
        An array to store the output. Must be the same shape as `X`.

    Returns
    -------
    r : array_like
        `r` will have the same shape as `X`; if out is provided, `r`
        will be equal to out.

    Binary ufuncs:
    ==============

    op(X, Y, out=None)
    Apply `op` to `X` and `Y` elementwise. May "broadcast" to make
    the shapes of `X` and `Y` congruent.

    The broadcasting rules are:

    * Dimensions of length 1 may be prepended to either array.
    * Arrays may be repeated along dimensions of length 1.

    Parameters
    ----------
    X : array_like
        First input array.
    Y : array_like
        Second input array.
    out : array_like
        An array to store the output. Must be the same shape as the
        output would have.

    Returns
    -------
    r : array_like
        The return value; if out is provided, `r` will be equal to out.

    """
    def __init__(self, func):
        self.func = func
        self.__doc__ = func.__doc__

    def _get_identity(self):
        return self.func.identity
    identity = property(_get_identity)

    def _get_nargs(self):
        return self.func.nargs
    nargs = property(_get_nargs)

    def _get_nin(self):
        return self.func.nin
    nin = property(_get_nin)

    def _get_nout(self):
        return self.func.nout
    nout = property(_get_nout)

    def _get_ntypes(self):
        return self.func.ntypes
    ntypes = property(_get_ntypes)

    def _get_signature(self):
        return self.func.signature
    signature = property(_get_signature)

    def _get_types(self):
        return self.func.types
    types = property(_get_types)

    def __call__(self, *args, **kwargs):
        if self.func.nin == 1:
            return self._unary_call(*args, **kwargs)
        elif self.func.nin == 2:
            return self._binary_call(*args, **kwargs)
        else:
            raise ValueError, "only unary and binary ufuncs supported"

    def _unary_call(self, input, out=None, *args, **kwargs):
        input = asarray(input)
        input_shape = _get_shape(input)
        input_dtype = _get_dtype(input)
        if not (is_distributed(input) or is_distributed(out)):
            # no ndarray instances used, pass through immediately to numpy
            return self.func(input, out, *args, **kwargs)
        if out is None:
            # input must be an ndarray given previous conditionals
            # TODO okay, is there something better than this?
            ignore = np.ones(1, dtype=input_dtype)
            out_type = self.func(ignore).dtype
            out = ndarray(input_shape, out_type)
        # sanity checks
        if not is_array(out):
            raise TypeError, "return arrays must be of ArrayType"
        out_shape = _get_shape(out)
        if input_shape != out_shape:
            # broadcasting doesn't apply to unary operations
            raise ValueError, 'invalid return array shape'
        # Now figure out what to do...
        if isinstance(out, ndarray):
            sync()
            # get out as an np.ndarray first
            npout = out.access()
            if npout is not None: # this proc owns data
                if input is out:
                    npin,release_in = npout,False
                else:
                    npin,release_in = _npin_piece_based_on_out(input,out)
                self.func(npin, npout, *args, **kwargs)
                if release_in:
                    input.release()
                out.release_update()
            #sync()
        elif isinstance(out, flatiter):
            sync()
            # first opt: input and out are same object
            #   we call _unary_call over again with the bases
            #   NOT SURE THAT THIS IS ACTUALLY OPTIMAL -- NEED TO TEST
            if input is out:
                self._unary_call(out.base, out.base, *args, **kwargs)
                return out.copy() # differs from NumPy (should be view)
            else:
                npout = out.access()
                if npout is not None: # this proc 'owns' data
                    if is_distributed(input):
                        npin = input.get(out._range)
                    else:
                        npin = input[out._range]
                    self.func(npin, npout, *args, **kwargs)
                    out.release_update()
            #sync()
        else:
            sync()
            # out is not distributed
            npin = input
            if is_distributed(input):
                npin = input.allget()
            self.func(npin, out, *args, **kwargs)
            #sync() # I don't think we need this one
        return out

    def _binary_call(self, first, second, out=None, *args, **kwargs):
        first_isscalar = np.isscalar(first)
        second_isscalar = np.isscalar(second)
        # just in case
        first = asarray(first)
        second = asarray(second)
        if not (is_distributed(first)
                or is_distributed(second)
                or is_distributed(out)):
            # no ndarray instances used, pass through immediately to numpy
            return self.func(first, second, out, *args, **kwargs)
        first_dtype = _get_dtype(first)
        second_dtype = _get_dtype(second)
        first_shape = _get_shape(first)
        second_shape = _get_shape(second)
        if out is None:
            # first and/or second must be ndarrays given previous conditionals
            # TODO okay, is there something better than this?
            dtype = None
            if first_isscalar:
                if second_isscalar:
                    dtype = np.find_common_type([],[first_dtype,second_dtype])
                else:
                    dtype = np.find_common_type([second_dtype],[first_dtype])
            else:
                if second_isscalar:
                    dtype = np.find_common_type([first_dtype],[second_dtype])
                else:
                    dtype = np.find_common_type([first_dtype,second_dtype],[])
            shape = util.broadcast_shape(first_shape, second_shape)
            out = ndarray(shape, dtype)
        # sanity checks
        if not is_array(out):
            raise TypeError, "return arrays must be of ArrayType"
        # Now figure out what to do...
        if isinstance(out, ndarray):
            sync()
            # get out as an np.ndarray first
            npout = out.access()
            if npout is not None: # this proc owns data
                # get matching and compatible portions of input arrays
                # broadcasting rules (may) apply
                if first is out:
                    npfirst,release_first = npout,False
                else:
                    npfirst,release_first = _npin_piece_based_on_out(
                            first,out,first_shape)
                if second is first:
                    # zeroth opt: first and second are same object, so do the
                    # same thing for second that we did for first
                    npsecond,release_second = npfirst,False
                elif second is out:
                    npsecond,release_second = npout,False
                else:
                    npsecond,release_second = _npin_piece_based_on_out(
                            second,out,second_shape)
                self.func(npfirst, npsecond, npout, *args, **kwargs)
                if release_first:
                    first.release()
                if release_second:
                    second.release()
                out.release_update()
            #sync()
        elif isinstance(out, flatiter):
            sync()
            # first op: first and second and out are same object
            if first is second is out:
                self._binary_call(out.base,out.base,out.base,*args,**kwargs)
                return out.copy()
            else:
                npout = out.access()
                if npout is not None: # this proc 'owns' data
                    if is_distributed(first):
                        npfirst = first.get(out._range)
                    else:
                        npfirst = first[out._range]
                    if second is first:
                        npsecond = npfirst
                    elif is_distributed(second):
                        npsecond = second.get(out._range)
                    else:
                        npsecond = second[out._range]
                    self.func(npfirst, npsecond, npout, *args, **kwargs)
                    out.release_update()
            #sync()
        else:
            sync()
            # out is not distributed
            ndfirst = first
            if is_distributed(first):
                ndfirst = first.allget()
            ndsecond = second
            if second is first:
                ndsecond = ndfirst
            elif is_distributed(second):
                ndsecond = second.allget()
            self.func(ndfirst, ndsecond, out, *args, **kwargs)
            #sync() # I don't think we need this one
        return out

    def reduce(self, a, axis=0, dtype=None, out=None, *args, **kwargs):
        """reduce(a, axis=0, dtype=None, out=None)

    Reduces `a`'s dimension by one, by applying ufunc along one axis.

    Let :math:`a.shape = (N_0, ..., N_i, ..., N_{M-1})`.  Then
    :math:`ufunc.reduce(a, axis=i)[k_0, ..,k_{i-1}, k_{i+1}, .., k_{M-1}]` =
    the result of iterating `j` over :math:`range(N_i)`, cumulatively applying
    ufunc to each :math:`a[k_0, ..,k_{i-1}, j, k_{i+1}, .., k_{M-1}]`.
    For a one-dimensional array, reduce produces results equivalent to:
    ::

     r = op.identity # op = ufunc
     for i in xrange(len(A)):
       r = op(r, A[i])
     return r

    For example, add.reduce() is equivalent to sum().

    Parameters
    ----------
    a : array_like
        The array to act on.
    axis : int, optional
        The axis along which to apply the reduction.
    dtype : data-type code, optional
        The type used to represent the intermediate results. Defaults
        to the data-type of the output array if this is provided, or
        the data-type of the input array if no output array is provided.
    out : ndarray, optional
        A location into which the result is stored. If not provided, a
        freshly-allocated array is returned.

    Returns
    -------
    r : ndarray
        The reduced array. If `out` was supplied, `r` is a reference to it.

    Examples
    --------
    >>> np.multiply.reduce([2,3,5])
    30

    A multi-dimensional array example:

    >>> X = np.arange(8).reshape((2,2,2))
    >>> X
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])
    >>> np.add.reduce(X, 0)
    array([[ 4,  6],
           [ 8, 10]])
    >>> np.add.reduce(X) # confirm: default axis value is 0
    array([[ 4,  6],
           [ 8, 10]])
    >>> np.add.reduce(X, 1)
    array([[ 2,  4],
           [10, 12]])
    >>> np.add.reduce(X, 2)
    array([[ 1,  5],
           [ 9, 13]])

        """
        if self.func.nin != 2:
            raise ValueError, "reduce only supported for binary functions"
        a = asarray(a)
        if not (isinstance(a, ndarray) or isinstance(out, ndarray)):
            # no ndarray instances used, pass through immediately to numpy
            return self.func.reduce(a, axis, dtype, out, *args, **kwargs)
        if axis < 0:
            axis += a.ndim
        if a.ndim < axis < 0:
            raise ValueError, "axis not in array"
        if out is None:
            shape = list(a.shape)
            del shape[axis]
            if dtype is None:
                dtype = a.dtype
            out = ndarray(shape, dtype=dtype)
        if out.ndim == 0:
            # optimize the 1d reduction
            nda = a.access()
            value = self.func.identity
            if nda is not None:
                value = self.func.reduce(nda)
            everything = MPI.COMM_WORLD.allgather(value)
            self.func.reduce(everything, out=out)
        else:
            slicer = [slice(0,None,None)]*a.ndim
            axis_iterator = iter(xrange(a.shape[axis]))
            # copy first loop iteration to 'out'
            slicer[axis] = axis_iterator.next()
            out[:] = a[slicer]
            # remaining loop iterations are appropriately reduced
            for i in axis_iterator:
                slicer[axis] = i
                ai = a[slicer]
                self.__call__(out,ai,out)
        return out

    def accumulate(self, a, axis=0, dtype=None, out=None, *args, **kwargs):
        if self.func.nin != 2:
            raise ValueError, "accumulate only supported for binary functions"
        a = asarray(a)
        if not (isinstance(a, ndarray) or isinstance(out, ndarray)):
            # no ndarray instances used, pass through immediately to numpy
            return self.func.accumulate(a, axis, dtype, out, *args, **kwargs)
        if axis < 0:
            axis += a.ndim
        if a.ndim < axis < 0:
            raise ValueError, "axis not in array"
        if out is None:
            if dtype is None:
                dtype = a.dtype
            out = ndarray(a.shape, dtype=dtype)
        if out.ndim == 1:
            # optimize the 1d accumulate
            if isinstance(out, ndarray):
                ndout = out.access()
                if ndout is not None:
                    lo,hi = util.calc_distribution_lohi(
                            out.global_slice, *out.distribution())
                    lo,hi = lo[0],hi[0]
                    piece = a[lo:hi]
                    if isinstance(piece, ndarray):
                        piece = piece.get()
                    self.func.accumulate(piece, out=ndout)
                    # probably more efficient to use allgather and exchange last
                    # values among all procs. We also need ordering information,
                    # so we exchange the 'lo' value.
                    everything = MPI.COMM_WORLD.allgather((ndout[-1],lo))
                    reduction = self.func.identity
                    for lvalue,llo in everything:
                        if lvalue is not None and llo < lo:
                            reduction = self.func(reduction,lvalue)
                    self.func(ndout,reduction,ndout)
                else:
                    everything = MPI.COMM_WORLD.allgather((None,None))
            else:
                raise NotImplementedError
        else:
            slicer_i = [slice(0,None,None)]*a.ndim
            slicer_i_1 = [slice(0,None,None)]*a.ndim
            axis_iterator = iter(xrange(a.shape[axis]))
            # copy first loop iteration to 'out'
            slicer_i[axis] = axis_iterator.next()
            out[slicer_i] = a[slicer_i]
            # remaining loop iterations are appropriately accumulated
            for i in axis_iterator:
                slicer_i[axis] = i
                slicer_i_1[axis] = i-1
                x = out[slicer_i_1]
                y = a[slicer_i]
                z = out[slicer_i]
                self.__call__(x,y,z)
                #self.__call__(out[slicer_i_1],a[slicer_i],out[slicer_i])
        return out

    def outer(self, *args, **kwargs):
        if self.func.nin != 2:
            raise ValueError, "outer product only supported for binary functions"
        raise NotImplementedError

    def reduceat(self, *args, **kwargs):
        if self.func.nin != 2:
            raise ValueError, "reduceat only supported for binary functions"
        raise NotImplementedError

# unary ufuncs
abs = ufunc(np.abs)
absolute = abs
arccos = ufunc(np.arccos)
arccosh = ufunc(np.arccosh)
arcsin = ufunc(np.arcsin)
arcsinh = ufunc(np.arcsinh)
arctan = ufunc(np.arctan)
arctanh = ufunc(np.arctanh)
bitwise_not = ufunc(np.bitwise_not)
ceil = ufunc(np.ceil)
conj = ufunc(np.conj)
conjugate = ufunc(np.conjugate)
cos = ufunc(np.cos)
cosh = ufunc(np.cosh)
deg2rad = ufunc(np.deg2rad)
degrees = ufunc(np.degrees)
exp = ufunc(np.exp)
exp2 = ufunc(np.exp2)
expm1 = ufunc(np.expm1)
fabs = ufunc(np.fabs)
floor = ufunc(np.floor)
frexp = ufunc(np.frexp)
invert = ufunc(np.invert)
isfinite = ufunc(np.isfinite)
isinf = ufunc(np.isinf)
isnan = ufunc(np.isnan)
log = ufunc(np.log)
log10 = ufunc(np.log10)
log1p = ufunc(np.log1p)
logical_not = ufunc(np.logical_not)
modf = ufunc(np.modf)
negative = ufunc(np.negative)
rad2deg = ufunc(np.rad2deg)
radians = ufunc(np.radians)
reciprocal = ufunc(np.reciprocal)
rint = ufunc(np.rint)
sign = ufunc(np.sign)
signbit = ufunc(np.signbit)
sin = ufunc(np.sin)
sinh = ufunc(np.sinh)
spacing = ufunc(np.spacing)
sqrt = ufunc(np.sqrt)
square = ufunc(np.square)
tan = ufunc(np.tan)
tanh = ufunc(np.tanh)
trunc = ufunc(np.trunc)
# binary ufuncs
add = ufunc(np.add)
arctan2 = ufunc(np.arctan2)
bitwise_and = ufunc(np.bitwise_and)
bitwise_or = ufunc(np.bitwise_or)
bitwise_xor = ufunc(np.bitwise_xor)
copysign = ufunc(np.copysign)
divide = ufunc(np.divide)
equal = ufunc(np.equal)
floor_divide = ufunc(np.floor_divide)
fmax = ufunc(np.fmax)
fmin = ufunc(np.fmin)
fmod = ufunc(np.fmod)
greater = ufunc(np.greater)
greater_equal = ufunc(np.greater_equal)
hypot = ufunc(np.hypot)
ldexp = ufunc(np.ldexp)
left_shift = ufunc(np.left_shift)
less = ufunc(np.less)
less_equal = ufunc(np.less_equal)
logaddexp = ufunc(np.logaddexp)
logaddexp2 = ufunc(np.logaddexp2)
logical_and = ufunc(np.logical_and)
logical_or = ufunc(np.logical_or)
logical_xor = ufunc(np.logical_xor)
maximum = ufunc(np.maximum)
minimum = ufunc(np.minimum)
mod = ufunc(np.mod)
multiply = ufunc(np.multiply)
nextafter = ufunc(np.nextafter)
not_equal = ufunc(np.not_equal)
power = ufunc(np.power)
remainder = ufunc(np.remainder)
right_shift = ufunc(np.right_shift)
subtract = ufunc(np.subtract)
true_divide = ufunc(np.true_divide)

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
    if not should_distribute(shape):
        return np.zeros(shape, dtype, order)
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
    if not should_distribute(shape):
        return np.ones(shape, dtype, order)
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
    if not should_distribute((N,M)):
        return np.eye(N,M,k,dtype)
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
    if not should_distribute(shape):
        return np.fromfunction(func, shape, **kwargs)
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
    #sync()
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
        length = int(math.ceil((stop-start)/step))
    # bail if threshold not met
    if not should_distribute(length):
        return np.arange(start,stop,step,dtype)
    if dtype is None:
        if (isinstance(start, (int,long))
                and isinstance(stop, (int,long))
                and isinstance(step, (int,long))):
            dtype = np.int64
        else:
            dtype = np.float64
    a = ndarray(length, dtype)
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
    if not should_distribute(num):
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
    #sync()
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
    if not should_distribute(num):
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
    #sync()
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
    a = asarray(a)
    b = asarray(b)
    if not (is_distributed(a) or is_distributed(b)):
        # numpy pass through
        return np.dot(a,b)
    # working with flatiter instances can be expensive, try this opt
    if (isinstance(a,flatiter)
            and isinstance(b,flatiter)
            and a._base is b._base):
        return (a._base * b._base).sum()
    if ((isinstance(a,flatiter) or a.ndim == 1)
            and (isinstance(b,flatiter) or b.ndim == 1)):
        if len(a) != len(b):
            raise ValueError, "objects are not aligned"
        tmp = multiply(a,b)
        ndtmp = tmp.access()
        local_sum = None
        if ndtmp is None:
            local_sum = np.add.reduce(np.asarray([0], dtype=tmp.dtype))
        else:
            local_sum = np.add.reduce(ndtmp)
        return ga.gop_add(local_sum)
    elif a.ndim == 2 and b.ndim == 2:
        if a.shape[1] != b.shape[0]:
            raise ValueError, "objects are not aligned"
        # use GA gemm if certain conditions apply
        valid_types = [np.dtype(np.float32),
                np.dtype(np.float64),
                np.dtype(np.float128),
                np.dtype(np.complex64),
                np.dtype(np.complex128)]
        if (a.base is None and b.base is None
                and a.dtype == b.dtype and a.dtype in valid_types):
            out = zeros((a.shape[0],b.shape[1]), a.dtype)
            ga.gemm(False, False, a.shape[0], b.shape[1], b.shape[0],
                    1, a.handle, b.handle, 1, out.handle)
            return out
        else:
            raise NotImplementedError
    elif isinstance(a,(ndarray,flatiter)) and isinstance(b,(ndarray,flatiter)):
        if a.shape[1] != b.shape[0]:
            raise ValueError, "objects are not aligned"
        raise NotImplementedError, "arbitrary dot"
    else:
        # assume we have a scalar somewhere, so just multiply
        return multiply(a,b)

def asarray(a, dtype=None, order=None):
    if isinstance(a, (ndarray,flatiter,np.ndarray,np.generic)):
        # we return ga.gain.ndarray instances for obvious reasons, but we
        # also return numpy.ndarray instances because they already exist in
        # whole on all procs -- no need to distribute pieces
        return a
    else:
        npa = np.asarray(a, dtype=dtype)
        if should_distribute(npa.size):
            g_a = ndarray(npa.shape, npa.dtype, npa)
            return g_a # distributed using Global Arrays ndarray
        else:
            return npa # possibly a scalar or zero rank array

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

class flatiter(object):
    """Flat iterator object to iterate over arrays.

    A `flatiter` iterator is returned by ``x.flat`` for any array `x`.
    It allows iterating over the array as if it were a 1-D array,
    either in a for-loop or by calling its `next` method.

    Iteration is done in C-contiguous style, with the last index varying the
    fastest. The iterator can also be indexed using basic slicing or
    advanced indexing.

    See Also
    --------
    ndarray.flat : Return a flat iterator over an array.
    ndarray.flatten : Returns a flattened copy of an array.

    Notes
    -----
    A `flatiter` iterator can not be constructed directly from Python code
    by calling the `flatiter` constructor.

    Examples
    --------
    >>> x = np.arange(6).reshape(2, 3)
    >>> fl = x.flat
    >>> type(fl)
    <type 'numpy.flatiter'>
    >>> for item in fl:
    ...     print item
    ...
    0
    1
    2
    3
    4
    5

    >>> fl[2:4]
    array([2, 3])

    """
    def __init__(self, base):
        self._base = base
        self._index = 0
        self._len = np.multiply.reduce(self._base.shape)
        self__len = self._len
        count = self__len//nproc
        if me*count < self__len:
            if (me+1)*count > self__len:
                self._range = slice(me*count,self__len,1)
            else:
                self._range = slice(me*count,(me+1)*count,1)
        else:
            self._range = None
    
    def _get_base(self):
        return self._base
    base = property(_get_base)

    def _get_coords(self):
        return np.unravel_index(self._index, self._base.shape)
    coords = property(_get_coords)

    def copy(self):
        """Get a copy of the iterator as a 1-D array.

        Examples
        --------
        >>> x = np.arange(6).reshape(2, 3)
        >>> x
        array([[0, 1, 2],
               [3, 4, 5]])
        >>> fl = x.flat
        >>> fl.copy()
        array([0, 1, 2, 3, 4, 5])

        """
        return self._base.flatten()
    
    def _get_index(self):
        return self._index
    index = property(_get_index)

    def get(self, key=None):
        # THIS OPERATION IS ONE-SIDED
        # we expect key to be a single value or a slice, but might also be an
        # iterable of length 1
        if key is None:
            # TODO this doesn't feel right -- communicate and then a local copy
            # operation? The answer is correct, but seems like too much extra
            # work.
            return self._base.get().flatten()
        try:
            key = key[0]
        except:
            pass
        if isinstance(key, slice):
            # get shape of global_slice
            shape = []
            offsets = []
            for gs in self._base.global_slice:
                if gs is None:
                    pass
                elif isinstance(gs, slice):
                    shape.append(util.slicelength(gs))
                    offsets.append(0)
                else:
                    shape.append(1)
                    offsets.append(gs)
            # create index coordinates
            i = (np.indices(shape).reshape(len(shape),-1).T + offsets)[key]
            return ga.gather(self._base.handle, i)
        else:
            # assumes int,long,etc
            try:
                index = np.unravel_index(key, self._base.shape)
                return self._base.get(index)
            except:
                raise IndexError, "unsupported iterator index (%s)" % str(key)

    def __getitem__(self, key):
        # THIS OPERATION IS COLLECTIVE
        # we expect key to be a single value or a slice, but might also be an
        # iterable of length 1
        try:
            key = key[0]
        except:
            pass
        if isinstance(key, slice):
            # get shape of global_slice
            shape = []
            offsets = []
            for gs in self._base.global_slice:
                if gs is None:
                    pass
                elif isinstance(gs, slice):
                    shape.append(util.slicelength(gs))
                    offsets.append(0)
                else:
                    shape.append(1)
                    offsets.append(gs)
            # create index coordinates
            i = (np.indices(shape).reshape(len(shape),-1).T + offsets)[key]
            # TODO optimize the gather since this is a collective
            return ga.gather(self._base.handle, i)
        else:
            # assumes int,long,etc
            try:
                index = np.unravel_index(key, self._base.shape)
                return self._base[index]
            except:
                raise IndexError, "unsupported iterator index (%s)" % str(key)

    def __len__(self):
        return self._len

    def __setitem__(self, key, value):
        raise NotImplementedError

    def next(self):
        if self._index < self._len:
            tmp = self.base[self.coords]
            self._index += 1
            return tmp
        else:
            raise StopIteration

    def access(self):
        """Return a copy of a 'local' portion."""
        if self._range_values is not None:
            raise ValueError, "call release or release_update before access"
        if self._range is None:
            self._range_values = None
        else:
            self._range_values = self[self._range]
        return self._range_values

    def release(self):
        self._range_values = None

    def release_update(self):
        if self._range is not None and self._range_values is not None:
            self[self._range] = self._range_values
        self._range_values = None

def clip(a, a_min, a_max, out=None):
    """Clip (limit) the values in an array.

    Given an interval, values outside the interval are clipped to
    the interval edges.  For example, if an interval of ``[0, 1]``
    is specified, values smaller than 0 become 0, and values larger
    than 1 become 1.

    Parameters
    ----------
    a : array_like
        Array containing elements to clip.
    a_min : scalar or array_like
        Minimum value.
    a_max : scalar or array_like
        Maximum value.  If `a_min` or `a_max` are array_like, then they will
        be broadcasted to the shape of `a`.
    out : ndarray, optional
        The results will be placed in this array. It may be the input
        array for in-place clipping.  `out` must be of the right shape
        to hold the output.  Its type is preserved.

    Returns
    -------
    clipped_array : ndarray
        An array with the elements of `a`, but where values
        < `a_min` are replaced with `a_min`, and those > `a_max`
        with `a_max`.

    See Also
    --------
    numpy.doc.ufuncs : Section "Output arguments"

    Examples
    --------
    >>> a = np.arange(10)
    >>> np.clip(a, 1, 8)
    array([1, 1, 2, 3, 4, 5, 6, 7, 8, 8])
    >>> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> np.clip(a, 3, 6, out=a)
    array([3, 3, 3, 3, 4, 5, 6, 6, 6, 6])
    >>> a = np.arange(10)
    >>> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> np.clip(a, [3,4,1,1,1,4,4,4,4,4], 8)
    array([3, 4, 2, 3, 4, 5, 6, 7, 8, 8])

    """
    # just in case
    a = asarray(a)
    a_min = asarray(a_min)
    a_max = asarray(a_max)
    if not (is_distributed(a)
            or is_distributed(a_min)
            or is_distributed(a_max)
            or is_distributed(out)):
        # no ndarray instances used, pass through immediately to numpy
        return np.clip(a, a_min, a_max, out)
    a_shape = _get_shape(a)
    a_min_shape = _get_shape(a_min)
    a_max_shape = _get_shape(a_max)
    if out is None:
        out = ndarray(a_shape, _get_dtype(a))
    # sanity checks
    if not is_array(out):
        raise TypeError, "output must be an array"
    if out.shape != a.shape:
        raise ValueError, ("clip: Output array must have thesame shape as "
                "the input.")
    # Now figure out what to do...
    if isinstance(out, ndarray):
        sync()
        # get out as an np.ndarray first
        npout = out.access()
        if npout is not None: # this proc owns data
            # get matching and compatible portions of input arrays
            # broadcasting rules (may) apply
            if a is out:
                npa,release_a = npout,False
            else:
                npa,release_a = _npin_piece_based_on_out(a,out,a_shape)
            if a_min is out:
                npa_min,release_a_min = npout,False
            elif a_min is a:
                npa_min,release_a_min = npa,False
            else:
                npa_min,release_a_min = _npin_piece_based_on_out(
                        a_min,out,a_min_shape)
            if a_max is out:
                npa_max,release_a_max = npout,False
            elif a_max is a:
                npa_max,release_a_max = npa,False
            elif a_max is a_min:
                npa_max,release_a_max = npa_min,False
            else:
                npa_max,release_a_max = _npin_piece_based_on_out(
                        a_max,out,a_max_shape)
            np.clip(npa, npa_min, npa_max, npout)
            if release_a:
                a.release()
            if release_a_min:
                a_min.release()
            if release_a_max:
                a_max.release()
            out.release_update()
        #sync()
    elif isinstance(out, flatiter):
        raise NotImplementedError, "flatiter version of clip"
        #sync()
        ## first op: first and second and out are same object
        #if first is second is out:
        #    self._binary_call(out.base,out.base,out.base,*args,**kwargs)
        #    return out.copy()
        #else:
        #    npout = out.access()
        #    if npout is not None: # this proc 'owns' data
        #        if is_distributed(first):
        #            npfirst = first.get(out._range)
        #        else:
        #            npfirst = first[out._range]
        #        if second is first:
        #            npsecond = npfirst
        #        elif is_distributed(second):
        #            npsecond = second.get(out._range)
        #        else:
        #            npsecond = second[out._range]
        #        self.func(npfirst, npsecond, npout, *args, **kwargs)
        #        out.release_update()
        #sync()
    else:
        sync()
        # out is not distributed
        nda = a
        if is_distributed(a):
            nda = a.allget()
        nda_min = a_min
        if a is a_min:
            nda_min = nda
        elif is_distributed(a_min):
            nda_min = a_min.allget()
        nda_max = a_max
        if a is a_max:
            nda_max = nda
        elif a_max is a_min:
            nda_max = nda_min
        elif is_distributed(a_max):
            nda_max = a_max.allget()
        np.clip(a, nda_min, nda_max, out)
        #sync() # I don't think we need this one
    return out

DEBUG = False
DEBUG_SYNC = False
def set_debug(val):
    global DEBUG
    DEBUG = val

def set_debug_sync(val):
    global DEBUG_SYNC
    DEBUG_SYNC = val

def print_debug(s):
    if DEBUG:
        print s

def print_sync(what):
    if DEBUG_SYNC:
        sync()
        if 0 == me:
            print "[0] %s" % str(what)
            for proc in xrange(1,nproc):
                data = MPI.COMM_WORLD.recv(source=proc, tag=11)
                print "[%d] %s" % (proc, str(data))
        else:
            MPI.COMM_WORLD.send(what, dest=0, tag=11)
        sync()
