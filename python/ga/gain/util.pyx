"""Contains index- and slice-related operations needed for bookkeeping."""

def is_canonical_slice(sliceobj):
    """Return True if the slice instance does not contain None."""
    return (sliceobj.start is not None
            and sliceobj.stop is not None
            and sliceobj.step is not None)

def canonicalize_indices(shape, indices):
    """Make getitem indices friendly to slice arithmetic.
    
    Replaces Ellipsis with slice(0,dim_max,1).
    Removes all None instances from slice instances with the result of 
    running slice.indices(dim_max).

    Raises IndexError in all cases where we cannot canonicalize.

    Returns the new slices list, always as a list, even if len() == 1.

    Parameters
    ----------
    shape : tuple of ints
        This is the base shape of the ndarray, without PseudoIndex (None),
        slice instances, or Ellipsis.
    indices : iterable of ints, None, slice instances, or Ellipsis
        Represents the slice to be taken from the ndarray with the given shape
    
    Examples
    --------
    >>> from numpy.lib.index_tricks import s_
    >>> canonicalize_indices((10,20,30), 1)
    [1, slice(0, 20, 1), slice(0, 30, 1)]
    >>> canonicalize_indices((10,20,30,40), s_[1,...,:5:-1])
    [1, slice(0, 20, 1), slice(0, 30, 1), slice(39, 5, -1)]
    >>> canonicalize_indices((10,20,30,40), s_[1,...,:5:-1,None])
    [1, slice(0, 20, 1), slice(0, 30, 1), slice(39, 5, -1), None]

    """
    # convert the indices into an iterable i.e. if a single int is passed
    try:
        iter(indices)
    except TypeError:
        indices = [indices]
    shape_iter = iter(shape)
    canonicalized_indices = []
    count_real_indices = 0
    count_ellipsis = 0
    # count number of real indices
    for index in indices:
        if isinstance(index,(int,long,slice)):
            count_real_indices += 1
    if count_real_indices > len(shape):
        raise IndexError, "invalid index"
    # now iterate through the indices and process
    for index in indices:
        if index is None:
            # PseudoIndex, we don't change these
            canonicalized_indices.append(None)
        elif index is Ellipsis:
            # Replace first Ellipsis with as many full slices as needed.
            # Additional Ellipsis are removed without warning/error.
            count_ellipsis += 1
            if 1 == count_ellipsis:
                for j in xrange(len(shape)-count_real_indices):
                    dim_max = shape_iter.next()
                    canonicalized_indices.append(slice(0,dim_max,1))
        elif isinstance(index, slice):
            dim_max = shape_iter.next()
            if is_canonical_slice(index):
                canonicalized_indices.append(index)
            else:
                canonicalized_indices.append(slice(*index.indices(dim_max)))
        else:
            # assumes an int, long, np.int64, etc
            dim_max = shape_iter.next()
            if index < 0:
                index += dim_max
            if index >= dim_max or index < 0:
                raise IndexError, "invalid index"
            canonicalized_indices.append(index)
    # ran out of indices; fill remaining with full slices
    for dim_max in shape_iter:
        canonicalized_indices.append(slice(0,dim_max,1))
    return canonicalized_indices

def slice_of_a_slice(original_slice, slice_operand):
    """Return a new slice representing a slice of a slice.

    Assumes slices are already in their canonical forms.

    In other words, if one were to slice an ndarray multiple times, return the
    slice or int index which would produce the equivalent result.

    Parameters
    ----------
    original_slice : slice or None
        If original_slice is None, None is returned.
        If original_slice is a slice instance, it is sliced further based on
        the given slice_operand.
    slice_operand : slice, int, long, 
        You can further slice a slice using either a slice, int, or long.
    
    Examples
    --------
    >>> from numpy.lib.index_tricks import s_
    >>> slice_of_a_slice(s_[3:18:2], s_[5:-1:-1])
    slice(13, 1, -2)
    >>> slice_of_a_slice(s_[1:18:3], s_[3])
    10

    """
    if isinstance(slice_operand, slice):
        if original_slice is None:
            if (slice_operand.start != 0
                    or slice_operand.stop != 1
                    or slice_operand.step != 1):
                raise IndexError, "%s,%s" % (original_slice, slice_operand)
            return None
        elif isinstance(original_slice, slice):
            start = ((slice_operand.start*original_slice.step)
                    + original_slice.start)
            stop = ((slice_operand.stop*original_slice.step)
                    + original_slice.start)
            step = slice_operand.step * original_slice.step
            return slice(start,stop,step)
    else:
        # assumes slice_operand is int, long, etc
        if original_slice is None:
            if slice_operand not in [0,1]:
                raise IndexError
            return None
        elif isinstance(original_slice, slice):
            shifted = (slice_operand*original_slice.step) + original_slice.start
            if (original_slice.step < 0
                    and shifted <= original_slice.stop
                    or original_slice.step > 0
                    and shifted >= original_slice.stop):
                raise IndexError
            return shifted
        else:
            raise IndexError

def slice_arithmetic(original_ops, ops):
    """Take slices of slices.
    
    Calls slice_of_a_slice as appropriate, skipping any int/long values in the
    original_ops.

    Notes
    -----
    Assumes inputs are in their canonical forms already.

    Parameters
    ----------
    original_ops : iterable of slice instances, None, int/long
        Represents the slicing already performed on an ndarray.
    ops : iterable of slice instances, None, int/long
        Represents how to (further) slice the ndarray.
    
    Returns
    -------
    A list representing the new slice.

    Examples
    --------
    >>> from numpy.lib.index_tricks import s_
    >>> slice_arithmetic(s_[3:38:2,5:15:1,None,57,30:-1:-1], s_[1,2,1,4])
    [5, 7, None, 57, 26]
    >>> slice_arithmetic(s_[3:38:2,5:15:1,None,57,30:-1:-1], s_[1,2,1,4:10:2])
    [5, 7, None, 57, slice(26, 20, -2)]

    """
    # First of all, the number of non-None values in ops should match that
    # of the number of non-integer values in original_ops.
    count_original_ops = 0
    for op in original_ops:
        if isinstance(op,slice) or op is None:
        #if not isinstance(op, (int,long)):
            count_original_ops += 1
    count_ops = 0
    for op in ops:
        if op is not None:
            count_ops += 1
    if count_original_ops != count_ops:
        raise ValueError, "incompatible shapes"
    new_op = []
    idx = 0
    for op in ops:
        while (idx < len(original_ops)
                and isinstance(original_ops[idx], (int,long))):
            new_op.append(original_ops[idx])
            idx += 1
        if op is None:
            new_op.append(op)
        else:
            original_op = original_ops[idx]
            new_op.append(slice_of_a_slice(original_op, op))
            idx += 1
    while (idx < len(original_ops)
            and isinstance(original_ops[idx], (int,long))):
        new_op.append(original_ops[idx])
        idx += 1
    return new_op

def calc_index_lohi(global_slice, lo, hi):
    """Calculate global_slice range for the subarray denoted by lo and hi.
    
    returns a list of slice objects representing the piece of the subarray
    that the lo and hi bounds maintain after applying bounds to global_slice.

    raises IndexError if the lo and hi bounds don't get a piece of the subarray
    
    Examples
    --------
    >>> slices = [slice(1,10,2), slice(10,1,-3), 2, 10]
    >>> calc_index_lohi(slices, [4,4,0,5], [20,9,3,20])
    [slice(5, 10, 2), slice(7, 3, -3), 2, 10]

    """
    # None as a value in the global_slice is otherwise ignored.
    item_key = []
    idx = 0
    for op in global_slice:
        if isinstance(op, (int,long)):
            if not (lo[idx] <= op < hi[idx]):
                raise IndexError, "lo/hi out of bounds"
            item_key.append(op)
            idx += 1
        elif isinstance(op, slice):
            item_key.append(subindex(op, lo[idx], hi[idx]))
            idx += 1
        elif op is None:
            pass
        else:
            raise ValueError, "global_slice contained unknown object"
    return item_key

def subindex(sliceobj, lo, hi):
    """Return a slice modified to fit between lo and hi.

    Notes
    -----
    Assumes the sliceobj is a "canonical" slice (no None instances, start and
    stop are positive and only step is allowed to be negative).

    Raises
    ------
    IndexError when a slice is out of the bounds of lo,hi 

    Returns
    -------
    A new slice between the values of lo and hi.

    Examples
    --------
    >>> subindex(slice(1,10,2), 4, 20)
    slice(5, 10, 2)
    >>> subindex(slice(1,10,2), 1, 20)
    slice(1, 10, 2)
    >>> subindex(slice(1,10,2), 4, 10)
    slice(5, 10, 2)
    >>> subindex(slice(1,10,2), 4, 9)
    slice(5, 9, 2)

    """
    new_start = 0
    new_stop = 0
    if sliceobj.step > 0:
        if sliceobj.start >= hi:
            raise IndexError, "start >= hi (out of bounds)"
        elif sliceobj.start >= lo: # start < hi is implied
            new_start = sliceobj.start
        else: # start < lo < hi is implied
            guess = (lo-sliceobj.start)//sliceobj.step
            new_start = guess*sliceobj.step + sliceobj.start
            while new_start < lo:
                guess += 1
                new_start = guess*sliceobj.step + sliceobj.start
        if sliceobj.stop <= lo:
            raise IndexError, "stop <= lo (out of bounds)"
        elif sliceobj.stop <= hi: # lo < stop is implied
            new_stop = sliceobj.stop
        else: # lo < hi < stop is implied
            new_stop = hi # this should be good enough
    else:
        if sliceobj.start < lo:
            raise IndexError, "negative step, start < lo (out of bounds)"
        elif sliceobj.start < hi:
            new_start = sliceobj.start
        else: # start >= hi >= lo
            guess = (hi-sliceobj.start)//sliceobj.step
            new_start = guess*sliceobj.step + sliceobj.start
            while new_start >= hi:
                guess += 1
                new_start = guess*sliceobj.step + sliceobj.start
        if sliceobj.stop >= hi:
            raise IndexError, "negative step, stop >= hi (out of bounds)"
        elif sliceobj.stop >= (lo-1):
            new_stop = sliceobj.stop
        else:
            new_stop = lo-1 # this should be good enough
    if length(new_start, new_stop, sliceobj.step) <= 0:
        raise IndexError, "slice arithmetic resulted in 0 length"
    return slice(new_start, new_stop, sliceobj.step)

def calc_distribution_lohi(global_slice, lo, hi):
    """Return lo,hi distribution based on current global_slice."""
    result = calc_index_lohi(global_slice, lo, hi)
    lo = []
    hi = []
    for gop,op in zip(global_slice,result):
        if isinstance(op, slice):
            lo.append((op.start-gop.start)/op.step)
            hi.append(((op.start+(op.step*(slicelength(op)-1)))-gop.start)/op.step)
            hi[-1]+=1
    return lo,hi

def access_slice(global_slice, lo, hi):
    """Converts from global_slice to a local slice appropriate for ga.access().

    Examples
    --------

    """
    result = calc_index_lohi(global_slice, lo, hi)
    # None as a value in the global_slice is otherwise ignored.
    item_key = []
    idx = 0
    for op in result:
        if isinstance(op, (int,long)):
            item_key.append(op-lo[idx])
            idx += 1
        elif isinstance(op, slice):
            start = op.start-lo[idx]
            stop = op.stop-lo[idx]
            # if start or stop are negative after translation replace with None
            if start < 0: start = None
            if stop < 0: stop = None
            item_key.append(slice(start, stop, op.step))
            idx += 1
        elif op is None:
            pass
        else:
            raise ValueError, "global_slice contained unknown object"
    return item_key

def slicelength(sliceobj):
    """Returns the length of the given slice instance.

    Examples
    -------
    >>> slicelength(slice(2,19,1))
    17
    >>> slicelength(slice(2,19,2))
    9

    """
    return length(sliceobj.start, sliceobj.stop, sliceobj.step)

def length(start, stop, step):
    result = None
    if (step < 0 and stop >= start) or (step > 0 and start >= stop):
        result = 0
    elif step < 0:
        result = (stop - start + 1) / (step) + 1
    else:
        result = (stop - start - 1) / (step) + 1
    return result

def slices_to_shape(items):
    try:
        iter(items)
    except:
        items = [items]
    new_shape = []
    for item in items:
        if item is None:
            new_shape.append(1)
        elif isinstance(item, slice):
            new_shape.append(slicelength(item))
        else:
            # assume int/long etc
            # we don't count int/long as part of shape
            pass
    return new_shape

def get_slice(global_slice, lo, hi):
    """Converts from global_slice to a local slice appropriate for ga.get().

    Examples
    --------
    TODO

    """
    restricted_slice = calc_index_lohi(global_slice, lo, hi)
    # None as a value in the global_slice is otherwise ignored.
    item_key = []
    idx = 0
    for rs,gs in zip(restricted_slice,global_slice):
        if isinstance(gs, (int,long)):
            idx += 1
        elif isinstance(gs, slice):
            offset = length(gs.start, rs.start, gs.step)
            slicelen = slicelength(rs)
            item_key.append(slice(offset, offset+slicelen, 1))
            idx += 1
        elif gs is None:
            pass
        else:
            raise ValueError, "global_slice contained unknown object"
    return item_key

def broadcast_shape(first, second):
    """Return the broadcasted version of shapes first and second."""
    def worker(x,y):
        if x is None: x = 1
        if y is None: y = 1
        if x != 1 and y != 1 and x != y:
            raise ValueError, "shape mismatch:" \
                    " objects cannot be broadcast to a single shape"
        return max(x,y)
    return tuple(reversed(map(worker, reversed(first), reversed(second))))

def broadcast_chomp(smaller_key, larger_key):
    """Return a key appropriate for the given shape."""
    new_key = []
    for s,l in zip(reversed(smaller_key),reversed(larger_key)):
        if s == 1:
            new_key.append(slice(0,1,1))
        else:
            new_key.append(l)
    new_key.reverse()
    return new_key

def transpose(global_slice, axes):
    """Swap the axes of the given global_slice.

    Returns the inverse of the transpose in addition to the new global_slice.

    Parameters
    ----------
    global_slice: tuple of integers and/or slice objects
        The slice to take from an ndarray.
    axes : list of ints
        `i` in the `j`-th place in the tuple means `a`'s `i`-th axis becomes
        `a.transpose()`'s `j`-th axis.
    
    Examples
    --------
    >>> transpose([slice(2,34,1), 4, None, slice(10,2,-1)], (1,2,0))
    ([2, 0, 1], [None, 4, slice(10, 2, -1), slice(2, 34, 1)])
    >>> transpose([1, 2, 3, slice(4,14,4)], [0])
    ([0], [1, 2, 3, slice(4, 14, 4)])
    >>> transpose([slice(0,10,1),slice(0,20,1)], [1,0])
    ([1, 0], [slice(0, 20, 1), slice(0, 10, 1)])

    """
    # create mapping for global_slice indices to actual indices, used later
    s = []
    for i,val in enumerate(global_slice):
        if val is None or isinstance(val, slice):
            s.append(i)
    if len(s) != len(axes):
        raise ValueError, "axes don't match array"
    # create the inverse of the given axes
    inverse = [None]*len(axes)
    for i,val in enumerate(axes):
        if val >= len(s):
            raise ValueError, "invalid axis for this array"
        inverse[val] = i
    # create the transpose of the global_slice now that we have mapping s
    ret = [None]*len(global_slice)
    count = 0
    for val in axes:
        while not (global_slice[count] is None
                or isinstance(global_slice[count], slice)):
            ret[count] = global_slice[count]
            count += 1
        ret[count] = global_slice[s[val]]
        count += 1
    return inverse,ret
    
class slh(object):
    def __init__(self, val, lo, hi):
        self.val = val
        self.lo = lo
        self.hi = hi

def transpose_lohi(global_slice, lo, hi, axes):
    """Reorder lo/hi based on the given axes and global_slice."""
    # create mapping for global_slice indices to actual indices, used later
    s = []
    for i,val in enumerate(global_slice):
        if val is None or isinstance(val, slice):
            s.append(i)
    if len(s) != len(axes):
        raise ValueError, "axes don't match array"
    # create a new list of slh instances
    ilo = iter(lo)
    ihi = iter(hi)
    things = [None]*len(global_slice)
    for i,val in enumerate(global_slice):
        if val is None:
            things[i] = slh(None,None,None)
        else:
            things[i] = slh(val,ilo.next(),ihi.next())
    # create the transpose of the global_slice now that we have mapping s
    ret = [None]*len(global_slice)
    count = 0
    for val in axes:
        while not (things[count].val is None
                or isinstance(things[count].val, slice)):
            ret[count] = things[count]
            count += 1
        ret[count] = things[s[val]]
        count += 1
    # create the return lo/hi values
    rlo = [None]*len(lo)
    rhi = [None]*len(hi)
    count = 0
    for val in ret:
        if val.val is None:
            pass
        else:
            rlo[count] = val.lo
            rhi[count] = val.hi
            count += 1
    return rlo,rhi

def unravel_index(x,dims):
    """Like np.unravel_index, but 'x' can be an integer array.

    Yeah, I know, numpy 1.6.0 has this already, but we're based on 1.5.1.
    I copied the code and modified it from 1.5.1.

    """
    import numpy as _nx
    x = _nx.asarray(x)
    if x.ndim == 0:
        return _nx.unravel_index(x,dims)
    max = _nx.prod(dims)-1
    if _nx.any(x>max) or _nx.any(x<0):
        raise ValueError("Invalid index, must be 0 <= x <= number of elements.")
    idx = _nx.empty_like(dims)

    # Take dimensions
    # [a,b,c,d]
    # Reverse and drop first element
    # [d,c,b]
    # Prepend [1]
    # [1,d,c,b]
    # Calculate cumulative product
    # [1,d,dc,dcb]
    # Reverse
    # [dcb,dc,d,1]
    dim_prod = _nx.cumprod([1] + list(dims)[:0:-1])[::-1]
    # Indices become [x/dcb % a, x/dc % b, x/d % c, x/1 % d]
    return tuple(x[:,None]//dim_prod % dims)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
