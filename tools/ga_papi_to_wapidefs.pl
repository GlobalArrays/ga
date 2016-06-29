#!/usr/bin/env perl
while (<STDIN>) {
  $line = $_;
  if ($line =~ /.*p(nga_.*) *\(/) {
    $line = "#define w$1 p$1\n";
  }
  if ($line =~ /PAPI/) {
    $line =~ s/PAPI/WAPIDEFS/g;
  }
  if ($line =~ /^ /) {
    $line = "";
  }
  if ($line =~ /^\/\*/) {
    $line = "";
  }
  if ($line =~ /^ \*\\/) {
    $line = "";
  }
  if ($line =~ /typedef/) {
    $line = "";
  }
  if ($line ne "") {
    print "$line";
  }
}
