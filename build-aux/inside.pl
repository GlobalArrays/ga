#!/usr/bin/perl

$numargs = $#ARGV + 1;
if ($numargs != 2) {
    print "Usage: wrapped.txt naked.txt\n";
    exit 1;
}

# Read each input file as a string (rather than a list).
local $/=undef;

open WRAPPED, "$ARGV[0]" or die "Couldn't open wrapped text file: $!";
$wrapped_lines = <WRAPPED>;
close WRAPPED;

open NAKED, "$ARGV[1]" or die "Couldn't open naked text file: $!";
$naked_lines = <NAKED>;
close NAKED;

# Multiline match of the original output within the wrapped output.
if ($wrapped_lines =~ /$naked_lines/) {
    print "Found as substring\n";
    exit 0;
}
elsif ($wrapped_lines eq $naked_lines) {
    print "Found equal\n";
    exit 0;
}
else {
    print "Not found\n";
    exit 1;
}
