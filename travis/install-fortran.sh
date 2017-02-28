#!/bin/sh

set -e
set -x

os=`uname`
TRAVIS_ROOT="$1"

# this is where updated Autotools will be for Linux
export PATH=$TRAVIS_ROOT/bin:$PATH

case "$os" in
    Darwin)
        echo "Mac"
        brew update
        brew install gfortran
    ;;

    Linux)
        echo "Linux"
        sudo apt-get install gfortran -y
        ;;
esac

echo "using gfortran at `which gfortran`"
