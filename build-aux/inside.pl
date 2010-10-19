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

# Replace newlines, + from wrapped and naked lines.
$wrapped_lines =~ tr/\n+/ /;
$naked_lines =~ tr/\n+/ /;

# Remove whitespace from beginning of wrapped and naked lines.
$wrapped_lines =~ s/^\s+//;
$naked_lines =~ s/^\s+//;

# Remove whitespace from end of wrapped and naked lines.
$wrapped_lines =~ s/\s+$//;
$naked_lines =~ s/\s+$//;

# If either wrapped_lines or naked_lines are empty, this is an error.
# It is assumed that the particular version string which created the input
# files should generate SOMETHING.
unless ($wrapped_lines) {
    exit 1;
}
unless ($naked_lines) {
    exit 1;
}

# Can the naked lines be found within the wrapped lines?
if ($wrapped_lines =~ /$naked_lines/) {
    #print "Found as substring\n";
    exit 0;
}
# Are the naked lines exactly the same as the wrapped lines?
elsif ($wrapped_lines eq $naked_lines) {
    #print "Found equal\n";
    exit 0;
}
else {
    #print "Not found\n";
    exit 1;
}
