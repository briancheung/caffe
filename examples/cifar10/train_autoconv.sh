#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_autoconv_solver.prototxt

