#!/usr/bin/env perl
while (<STDIN>) {
  $line = $_;
  if (/^#/) {
    print "$_";
  }
}
