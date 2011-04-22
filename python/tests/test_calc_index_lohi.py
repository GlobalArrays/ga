from numpy.lib.index_tricks import s_
from ga.util import calc_index_lohi, subindex, calc_distribution_lohi

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
lo,hi = calc_distribution_lohi(slices, [4,4,0,5], [20,9,3,20])
print zip(lo,hi)
