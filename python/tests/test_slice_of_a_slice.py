from numpy.lib.index_tricks import s_
from ga.util import slice_of_a_slice

try:
    print slice_of_a_slice(s_[3:38:2], s_[::-1])
    print "did not catch TypeError"
except TypeError:
    print "caught TypeError"

print ">>> slice_of_a_slice(s_[3:18:2], s_[5:-1:-1])"
print slice_of_a_slice(s_[3:18:2], s_[5:-1:-1])

print ">>> slice_of_a_slice(s_[1:18:3], s_[3])"
print slice_of_a_slice(s_[1:18:3], s_[3])


