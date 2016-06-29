#!/usr/bin/env perl
while (<STDIN>) {
  $line = $_;
  $line =~ s/PAPI/WAPI/g;
  $line =~ s/pnga_/wnga_/g;
  if (!($line =~ /typedef/)) {
    print "$line";
  }
}
