#! /usr/bin/env python3

import os
import json
import subprocess
import numpy as np

if __name__ == '__main__':
    dx = 0.5
    path = '/home/diogo/Desktop/Stability/'
    os.chdir(path)

    # Read the desired damping coefficient from .json file
    fd = open('d.json')
    params = json.load(fd)
    d = params['damping_coefficient']

    ds = [d - dx, d, d + dx]

    fn = 'parameters.json'
    for id, d_ in enumerate(ds):
        # Generate parameters.json
        output = {'damping_coefficient': d_, 'iteration': id}
        with open(fn, 'w') as outfile:
            json.dump(output, outfile)
        print('################################################')
        print('################################################')
        print('################################################')
        print('Simulation ready to run. Iteration number: ', id)
        print('################################################')
        print('################################################')
        print('################################################')
        subprocess.run(['/home/diogo/catkin_ws/src/gd_test/src/iterations.sh'])

    # Gather results from simulations
    print('Ready to gather results')
    output = []
    for i in range(3):
        fn = 'output_' + str(i) + '.json'
        jf = open(fn)
        data = json.load(jf)
        output.append(data['result'])

    grad = {'gradient': np.gradient(np.array(output))[1]}

    # Save result for further analysis
    fn = 'gradient.json'
    with open(fn, 'w') as outfile:
        json.dump(grad, outfile)
