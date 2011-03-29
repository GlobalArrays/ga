from numpy.lib.index_tricks import s_
from ga.util import calc_index_lohi, subindex, global_to_local_slice

print "subindex(slice(1,10,2), 4, 20)"
print subindex(slice(1,10,2), 4, 20)

print "subindex(slice(1,10,2), 1, 20)"
print subindex(slice(1,10,2), 1, 20)

print "subindex(slice(1,10,2), 4, 10)"
print subindex(slice(1,10,2), 4, 10)

print "subindex(slice(1,10,2), 4, 9)"
print subindex(slice(1,10,2), 4, 9)

print "subindex(slice(10,1,-3), 4, 9)"
print subindex(slice(10,1,-3), 4, 9)

slices = [slice(1,10,2), slice(10,1,-3), 2, 10]
print calc_index_lohi(slices, [4,4,0,5], [20,9,3,20])

slices = [slice(1,10,2), slice(10,1,-3), 2, 10]
print global_to_local_slice(slices, [4,4,0,5], [20,9,3,20])
