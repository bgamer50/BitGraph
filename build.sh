#!/bin/bash

VERBOSE=1 cmake -GNinja .
ninja -j12 all
echo "Build Done"