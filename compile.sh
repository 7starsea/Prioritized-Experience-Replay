#!/bin/bash

build_dir=build
[ -d $build_dir ] || mkdir $build_dir
cd $build_dir
    cmake -G "Sublime Text 2 - Unix Makefiles" ../util_cpp/
    make -j
cd ..


