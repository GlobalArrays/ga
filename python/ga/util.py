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
        elif isinstance(index, (int,long)):
            dim_max = shape_iter.next()
            if index < 0:
                index += dim_max
            if index >= dim_max or index < 0:
                raise IndexError, "invalid index"
            canonicalized_indices.append(index)
        else:
            raise IndexError, ("each subindex must be either a slice, "
                    "an integer, Ellipsis, or newaxis")
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
    if isinstance(slice_operand, (int,long)):
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
    elif isinstance(slice_operand, slice):
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
        if not isinstance(op, (int,long)):
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
        while isinstance(original_ops[idx], (int,long)):
            new_op.append(original_ops[idx])
            idx += 1
        if op is None:
            new_op.append(op)
        else:
            original_op = original_ops[idx]
            new_op.append(slice_of_a_slice(original_op, op))
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
            if not (lo[idx] <= op < hi):
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
    if sliceobj.step > 0:
        new_start = 0
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
        new_stop = 0
        if sliceobj.stop <= lo:
            raise IndexError, "stop <= lo (out of bounds)"
        elif sliceobj.stop <= hi: # lo < stop is implied
            new_stop = sliceobj.stop
        else: # lo < hi < stop is implied
            new_stop = hi # this should be good enough
        return slice(new_start, new_stop, sliceobj.step)
    else:
        new_start = 0 
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
        new_stop = 0
        if sliceobj.stop >= hi:
            raise IndexError, "negative step, stop >= hi (out of bounds)"
        elif sliceobj.stop >= (lo-1):
            new_stop = sliceobj.stop
        else:
            new_stop = lo-1 # this should be good enough
        return slice(new_start, new_stop, sliceobj.step)

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
            item_key.append(slice(op.start-lo[idx], op.stop-lo[idx], op.step))
            idx += 1
        elif op is None:
            pass
        else:
            raise ValueError, "global_slice contained unknown object"
    return item_key

def slicelength(sliceobj):
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
        elif isinstance(item, (int,long)):
            # we don't count int/long as part of shape
            pass
        else:
            raise IndexError, "invalid index"
    return new_shape

def get_slice(global_slice, lo, hi):
    """Converts from global_slice to a local slice appropriate for ga.get().

    Examples
    --------
    TODO

    """
    restricted_slice = calc_index_lohi(global_slice, lo, hi)
    #print "![?] in get_slice(%s,%s,%s) restricted_slice=%s" % (
    #        global_slice, lo, hi, restricted_slice)
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
    #print "![?] get_slice(%s,%s,%s)=%s" % (global_slice, lo, hi, item_key)
    return item_key

if __name__ == '__main__':
    import doctest
    doctest.testmod()
