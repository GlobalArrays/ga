from numpy.lib.index_tricks import s_
from ga.util import canonicalize_indices

print canonicalize_indices((10,20,30), 1)
# [1, slice(0, 20, 1), slice(0, 30, 1)]

print canonicalize_indices((10,20,30,40), s_[1,...,:5:-1])
# [1, slice(0, 20, 1), slice(0, 30, 1), slice(39, 5, -1)]

print canonicalize_indices((10,20,30,40,50), s_[1,...,:5:-1,None])
# [1, slice(0, 20, 1), slice(0, 30, 1), slice(39, 5, -1), None]

print canonicalize_indices((10,20,30,40,50), s_[-1,...,:5:-1,None])
# [9, slice(0, 20, 1), slice(0, 30, 1), slice(39, 5, -1), None]

print canonicalize_indices((10,20,30,40,50), s_[-10,...,:5:-1,None])
# [0, slice(0, 20, 1), slice(0, 30, 1), slice(39, 5, -1), None]

try:
    print canonicalize_indices((10,20,30,40,50), s_[-11,...,:5:-1,None])
    print "did not catch IndexError"
except IndexError:
    print "caught IndexError OK"

