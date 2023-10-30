#!/bin/bash

# Set up the ROS environment
source /home/diogo/catkin_ws/devel/setup.bash
cd ~/catkin_ws/src/gd_test/launch
names=("Rampa_3Deg.world" "Rampa_6Deg.world" "Rampa_9Deg.world" "Rampa_12Deg.world" "Rampa_15Deg.world")
cd /home/diogo/Desktop/Stability

read number < environment.txt


cd ~/catkin_ws/src/gd_test/launch

substitute="${names[number-1]}"

new_line="    <arg name=\"world_name\" value=\"\$(find posture)/\worlds/$substitute\"/>"
echo $new_line
sed -i "10s|.*|$new_line|" gd.launch


cd ~/catkin_ws/src/gd_test/src
roslaunch gd_test gd.launch --screen & pid=$!
wait $pid
