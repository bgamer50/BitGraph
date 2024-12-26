#!/bin/bash

if [[ $1 = "release" ]]; then
    echo "Performing release build!"
    VERBOSE=1 cmake -GNinja .
    ninja -j4 all
    echo "Build Done"
else
    echo "Performing debug build!"
    VERBOSE=1 cmake -DCMAKE_BUILD_TYPE=Debug -GNinja .
    ninja -j4 all
    echo "Build Done"
fi