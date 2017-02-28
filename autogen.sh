#! /bin/sh

set -e
set -x

if [ -z "$1" ]; then
    AUTOTOOLS_DIR="`pwd`/autotools"
else
    AUTOTOOLS_DIR="$1"
fi

# this is where updated Autotools will be for Linux
# we force the use of specific Autotools versions
if [ ! -d "$AUTOTOOLS_DIR" ]; then
    sh ./travis/install-autotools.sh "$AUTOTOOLS_DIR"
fi

export PATH="$AUTOTOOLS_DIR/bin":$PATH

autoreconf=${AUTORECONF:-autoreconf}
$autoreconf ${autoreconf_args:-"-vif"}
