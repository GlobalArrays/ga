#!/bin/csh

# port 'machine-list' file.proto

if ($#argv != 2) then
  echo usage $0 '"machine-list" file.proto'
  exit 1
endif

if (! -e "$2") then
  echo Prototype file "$2" does not exist
  exit 1
endif

set output = `echo $2 | sed -e 's/\.proto//'`

set cdir = `pwd`

echo Porting $2 to $output for $1

echo $1  > port.$$
cat  $2 >> port.$$

awk -f port.awk port.$$ | \
      sed "s,TOP_LEVEL_DIRECTORY,$cdir,g" > $output

/bin/rm -f port.$$
