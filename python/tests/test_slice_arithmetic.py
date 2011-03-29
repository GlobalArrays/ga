from numpy.lib.index_tricks import s_
from ga.util import slice_arithmetic as slice_arithmetic

try:
    print slice_arithmetic(s_[3:38:2,5:15,None,1,::-1], [s_[::-1]])
    print "did NOT catch ValueError"
except ValueError, e:
    print "caught ValueError %s" % e

print "slice_arithmetic(s_[3:38:2,5:15:1,None,57,30:-1:-1], s_[1,2,1,4])"
print slice_arithmetic(s_[3:38:2,5:15:1,None,57,30:-1:-1], s_[1,2,1,4])

print "slice_arithmetic(s_[3:38:2,5:15:1,None,57,30:-1:-1], s_[1,2,1,4:10:2])"
print slice_arithmetic(s_[3:38:2,5:15:1,None,57,30:-1:-1], s_[1,2,1,4:10:2])
