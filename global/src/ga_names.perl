#!/usr/local/bin/perl
#
#
# Finds all C GA routines and prepares include file that defines
# uppercase names for Cray machines (fortran-C name convention)
#
#
# 	finds references to GA routines that end with _ (C-versions)
# 	uses associative array to assure uniqueness
# 	strips the names of trailing _ and transforms to uppercase
# 	JN 09.08.94
#


#
# form associative array of GA names that end with _ in all the argument files
#

while (<>){
    if (/.*(ga_.*)_\(.*/){
	$garoutine = $1;
	$found{$garoutine} = 1;
    }
}

#
# now sort them up and print definition entries
#

foreach $symbol (sort keys(%found)){
    $upper = $symbol;
    $symbol = $symbol . '_';
    $upper =~ tr/a-z/A-Z/;
    #print "#define $upper \t\t $symbol";
    printf("#define  %-24s  %s\n", $symbol,  $upper);
}
