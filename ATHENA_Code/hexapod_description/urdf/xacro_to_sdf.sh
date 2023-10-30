#!/usr/bin/env bash

echo "Cleaning sdf and urdf"
rm -rf model.sdf model.urdf
rosrun xacro xacro hexapod.xacro > model.urdf
gz sdf -p model.urdf > model.sdf
echo "Generated SDF"
