#!/bin/sh
#
# This script extracts the version from global/src/gacommon.h, which is the master
# location for this information.
#
filename="global/src/gacommon.h"

if [ ! -f $filename ]; then
    echo "version.sh: error: global/src/gacommon.h does not exist" 1>&2
    exit 1
fi
MAJOR=`egrep '^#define +GA_VERSION_MAJOR +[0-9]+$' $filename`
MINOR=`egrep '^#define +GA_VERSION_MINOR +[0-9]+$' $filename`
PATCH=`egrep '^#define +GA_VERSION_PATCH +[0-9]+$' $filename`
if [ -z "$MAJOR" -o -z "$MINOR" -o -z "$PATCH" ]; then
    echo "version.sh: error: could not extract version from $filename" 1>&2
    exit 1
fi
MAJOR=`echo $MAJOR | awk '{ print $3 }'`
MINOR=`echo $MINOR | awk '{ print $3 }'`
PATCH=`echo $PATCH | awk '{ print $3 }'`
echo $MAJOR.$MINOR.$PATCH | tr -d '\n'

