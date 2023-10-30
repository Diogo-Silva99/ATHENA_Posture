#!/bin/bash
cd /home/diogo/Desktop/Stability
export FROM_SCRIPT=1
fileiteration="iteration.txt"

if [ -f $fileiteration ]; then
  current_iteration=$(<$fileiteration)
else
  current_iteration=1
fi

# cd ~/catkin_ws/src/gd_test/src
# python3 initial_conditions.py --screen & pid=$!
# wait $pid

iterations=1000
filename="status.txt"
total_iterations="total_ite.txt"
echo "$iterations" > $total_iterations

cd ~/catkin_ws/src/gd_test/src
for i in $(seq $current_iteration $iterations)
do
  echo "Running iteration $i"
  # Changing directory to check status.txt
  cd /home/diogo/Desktop/Stability
  if grep -q False $filename; then
    cd ~/catkin_ws/src/gd_test/src
    python3 Q_Learning.py --screen & pid=$!
    cd /home/diogo/Desktop/Stability
    echo "$i" > $fileiteration
    cd ~/catkin_ws/src/gd_test/src
    wait $pid
  else
    break
  fi
done

cd /home/diogo/Desktop/Stability

if grep -wq $iterations iteration.txt; then
  echo "1" > $fileiteration
fi

echo "Program ended!"
