#!/bin/bash

# usage:  ./make.sh
# use val/call-grind to profile

set -e  # exit immediately if anything fails

rm -rf ~/.cmake  # otherwise env/bin/plugins sometimes gets out of sync

rootdir=/groups/mousebrainmicro/mousebrainmicro/Software/barycentric5
installdir=$rootdir/env

mkdir -p $installdir/build
export ND_DIR=$installdir/build/nd
export ND_INCLUDE_DIRS=$installdir/include/nd
export PATH=/usr/local/cmake-3.10.1/bin:$PATH

# SL7
export CC=/usr/local/gcc-6.1.0/bin/gcc
export CXX=/usr/local/gcc-6.1.0/bin/g++
export LD_LIBRARY_PATH=/usr/local/gcc-6.1.0/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cmake-3.6.1/bin:$PATH

rm -rf $installdir/build/mltk-bary
mkdir $installdir/build/mltk-bary
cd $installdir/build/mltk-bary
release='-DCMAKE_BUILD_TYPE=Release'
#release='-DCMAKE_BUILD_TYPE=Debug -DCMAKE_VERBOSE_MAKEFILE=ON'
cmake $release -DCMAKE_INSTALL_PREFIX=$installdir $rootdir/src/mltk-bary/
make
make install
