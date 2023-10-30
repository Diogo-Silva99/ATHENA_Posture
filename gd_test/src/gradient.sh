#!/bin/bash

cd ~/catkin_ws/src/gd_test/src

python3 grad.py --screen & pid=$!
wait $pid
