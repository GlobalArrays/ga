#! /bin/sh

set -e
set -x

if [ -z "$1" ]; then
    AUTOTOOLS_DIR="`pwd`/autotools"
else
    AUTOTOOLS_DIR="$1"
fi

# This is where updated Autotools will be for Linux.
# We force the use of specific Autotools versions.
sh ./travis/install-autotools.sh "$AUTOTOOLS_DIR"

export PATH="${AUTOTOOLS_DIR}/bin":$PATH
export M4="${AUTOTOOLS_DIR}/bin/m4"
export AUTOCONF="${AUTOTOOLS_DIR}/bin/autoconf"
export AUTOHEADER="${AUTOTOOLS_DIR}/bin/autoheader"
export AUTOM4TE="${AUTOTOOLS_DIR}/bin/autom4te"
export AUTORECONF="${AUTOTOOLS_DIR}/bin/autoreconf"
export AUTOSCAN="${AUTOTOOLS_DIR}/bin/autoscan"
export AUTOUPDATE="${AUTOTOOLS_DIR}/bin/autoupdate"
export IFNAMES="${AUTOTOOLS_DIR}/bin/ifnames"
export AUTOMAKE="${AUTOTOOLS_DIR}/bin/automake"
export ACLOCAL="${AUTOTOOLS_DIR}/bin/aclocal"

autoreconf -vif

# patch to configure script for PGF90 and -lnuma
for conffile in configure comex/configure armci/configure
do
    # check whether patch is needed
    if grep lnuma $conffile > /dev/null
    then
        echo "patch already applied to $conffile"
    else
        echo "patching $conffile"
        # OSX sed doesn't do in-place easily, the following should work anywhere
        rm -f $conffile.tmp
        sed '/cmdline.*ignore/a\
ac_f77_v_output=`echo $ac_f77_v_output | sed "s/ -lnuma//g"`\
' $conffile > $conffile.tmp
        mv $conffile.tmp $conffile
        # sed might change file permissions
        chmod ug+xr $conffile
    fi
done

# overwrite config.guess and config.sub with latest
for dir in build-aux comex/build-aux armci/build-aux
do
    cp $AUTOTOOLS_DIR/bin/config.guess $dir
    cp $AUTOTOOLS_DIR/bin/config.sub $dir
    # ensure these are executable scripts
    chmod ug+xr $dir/config.guess
    chmod ug+xr $dir/config.sub
done

# patch ltmain.sh for special intel -mkl flag
for dir in build-aux comex/build-aux armci/build-aux
do
    rm -f $dir/ltmain.sh.tmp
    sed 's/\([m6][t4]|\)/mkl*|-\1/' $dir/ltmain.sh > $dir/ltmain.sh.tmp
    mv $dir/ltmain.sh.tmp $dir/ltmain.sh
    # sed might change file permissions
    chmod ug+xr $dir/ltmain.sh
done

