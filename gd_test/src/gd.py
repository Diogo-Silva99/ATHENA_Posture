#! /usr/bin/env python3

import os
import json
import subprocess
import numpy as np

if __name__ == '__main__':
    path = '/home/diogo/Desktop/Stability/'
    os.chdir(path)
    # Read parameters
    fd = open('initial_conditions.json')
    params = json.load(fd)
    tolerance = params['tolerance']
    alpha = params['learn_rate']
    d = params['d']
    # Send the damping coefficient to parameters.json
    output = {'damping_coefficient': params['d']}
    fn = 'd.json'
    with open(fn, 'w') as outfile:
        json.dump(output, outfile)

    # Run gradient.py
    subprocess.run(['/home/diogo/catkin_ws/src/gd_test/src/gradient.sh'])
    # Get the gradient value
    fd = open('gradient.json')
    params = json.load(fd)
    grad = params['gradient']

    diff = - alpha * grad
    if np.all(np.abs(diff) <= tolerance):
        fn = open('status.txt', 'w')
        fn.write('True \n')
        fn.close()
    else:
        d += diff
        # Save the output to control the loop
        output = {'tolerance': tolerance, 'learn_rate': alpha, 'd': d}
        fn = 'initial_conditions.json'
        with open(fn, 'w') as outfile:
            json.dump(output, outfile)
