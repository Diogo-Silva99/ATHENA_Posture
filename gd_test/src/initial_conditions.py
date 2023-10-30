#! /usr/bin/env python3

import os
import json

if __name__ == '__main__':
    path = '/home/diogo/Desktop/Stability/'
    os.chdir(path)
    print('Setting the initial conditions for the algorithm ...')
    start = 5.0
    tolerance = 0.8
    learn_rate = 0.5

    input = {'tolerance': tolerance, 'learn_rate': learn_rate, 'd': start}
    fn = 'initial_conditions.json'
    with open(fn, 'w') as outfile:
        json.dump(input, outfile)
    print('Saving the boolean variable to control the loop ...')
    fn = open('status.txt', 'w')
    fn.write('False \n')
    fn.close()
    print('All set. Ready to begin.')
