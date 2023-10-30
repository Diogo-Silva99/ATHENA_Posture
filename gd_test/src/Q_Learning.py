#! /usr/bin/env python3

import os
import json
import subprocess
import numpy as np
import random

class qLearning:
    def __init__(self, num_states, num_actions, alpha, gamma, explore_rate):
        self.num_actions = num_actions
        self.num_states = num_states
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.explore_rate = explore_rate
        self.q_table = np.zeros((num_actions,num_states))
        #self.q_table = self.q_table.tolist()

    def choose_action(self, state_q):
        if np.random.rand() < self.explore_rate:
            # Explore (randomly choose an action)
            action = np.random.choice(self.num_actions)
        else:
            # Exploit (choosse the action with the highest Q-Value)
            action = np.argmax(self.q_table[:, state_q])
        print(action)
        return action

    def update_q_table(self, state_q, action, reward, next_state):
        max_q_value = np.max(self.q_table[:, next_state])
        current_q_value = self.q_table[action, state_q]
        self.q_table[action, state_q] += self.alpha * (reward + self.gamma * max_q_value - current_q_value)
        return self.q_table[action, state_q]

    def save_q_table(self, filename):
        np.savetxt(filename, self.q_table)

    def load_q_table(self, filename):
        self.q_table = np.loadtxt(filename)

    def final_q_table(self, filename):
        np.savetxt(filename, self.q_table)

if __name__ == '__main__':
    path = '/home/diogo/Desktop/Stability/'
    os.chdir(path)

    with open ('iteration.txt', 'r') as file:
        file_contents = file.read()
        print('##############')
        print('##############')
        print(file_contents)
        print('##############')
        print('##############')

    with open('environment.txt', 'w') as file:
        state_env = random.randint(1, 5)
        file.write(str(state_env))

    if int(file_contents) == 1:
        initial_kh = 0
        fm_kh = 'kh_value.json'
        with open(fm_kh, 'w') as outfile:
            json.dump(initial_kh, outfile)
        ep = 1
        actuation_RMS = 0
        height_RMS = 0
        roll_RMS = 0
        pitch_RMS = 0
        state_q = int(state_env) - 1

        state = [actuation_RMS, height_RMS, roll_RMS, pitch_RMS]
        input = {'Actuation_RMS': actuation_RMS, 'Height_RMS': height_RMS, 'Roll_RMS': roll_RMS, 'Pitch_RMS': pitch_RMS}
        fn = 'state_values.json'
        with open(fn, 'w') as outfile:
            json.dump(input, outfile)

    else:

        ep = int(file_contents)
        '''fd = open('state_values_update.json')
        params = json.load(fd)
        actuation_RMS = params['Actuation_RMS']
        height_RMS = params['Height_RMS']
        roll_RMS = params['Roll_RMS']
        pitch_RMS = params['Pitch_RMS']
        state = [actuation_RMS, height_RMS, roll_RMS, pitch_RMS]'''
        state_q = int(state_env) - 1

    with open('total_ite.txt', 'r') as f:
        value = int(f.read().strip())

    epochs = value
    #epochs = 1000
    alpha = 0.8
    gamma = 0.99
    explore_rate = 0.3
    k_h = [0, 0.2, 0.4, 0.6, 0.8, 1]
    states = ['3Deg', '6Deg', '9Deg', '12Deg', '15Deg']
    num_states = 5
    num_actions = 6
    if int(file_contents) == 1:
        q_learning = qLearning(num_states, num_actions, alpha, gamma, explore_rate)
        print(q_learning.q_table)
        Done = False
    else:
        q_learning = qLearning(num_states, num_actions, alpha, gamma, explore_rate)
        q_learning.load_q_table('q_table.txt')
        print(q_learning.q_table)
        Done = False
    print(ep)
    print("INICIAR CICLO")
    for e in range(epochs):
        if Done == False:
            action = q_learning.choose_action(state_q)
            fm_kh = 'kh_value.json'
            input = {'kh_value': k_h[action]}
            with open(fm_kh, 'w') as outfile:
                json.dump(input, outfile)

            old_state = state_q
            print("ESTADO UTILIZADO NA SIMULAÇÃO")
            print("ESTADO UTILIZADO NA SIMULAÇÃO")
            print(state_q)
            print(states[state_q])
            print("ESTADO UTILIZADO NA SIMULAÇÃO")
            print("ESTADO UTILIZADO NA SIMULAÇÃO")
            print("K_H UTILIZADO NA SIMULAÇÃO")
            print("K_H UTILIZADO NA SIMULAÇÃO")
            print(action)
            print(k_h[action])
            print("K_H UTILIZADO NA SIMULAÇÃO")
            print("K_H UTILIZADO NA SIMULAÇÃO")
            subprocess.run(['/home/diogo/catkin_ws/src/gd_test/src/iterations.sh'])
            #output file is generated
            #fn = open('output.json')
            #params_fn = json.load(fn)
            fst_va = open('state_values_update.json')
            fst_va_params = json.load(fst_va)
            actuation_RMS = fst_va_params['Actuation_RMS']
            height_RMS = fst_va_params['Height_RMS']
            roll_RMS = fst_va_params['Roll_RMS']
            pitch_RMS = fst_va_params['Pitch_RMS']
            RMS_Values = [actuation_RMS, height_RMS, roll_RMS, pitch_RMS]

            reward_va = open('reward_value.json')
            reward_param = json.load(reward_va)

            reward = reward_param['reward']

            next_state = random.randint(0, 4)
            print("K_H UTILIZADO")
            print(k_h[action])
            print("ESTADO USADO")
            print(states[state_q])
            print("ESTADO USADO")
            print(reward)
            print(next_state)
            print("NEXT_STATE")
            print("NEXT_STATE")
            print("NEXT_STATE")

            q_learning.update_q_table(state_q, action, reward, next_state)

            # q_table[state, action] = new_value
            #update = q_learning.update_q_value(state, action, reward, next_state)
            print(q_learning.q_table)
            #q_learn_table = 'Q_table.json'
            #input_q = {'q_table': q_learning.q_table}
            q_learning.save_q_table('q_table.txt')
            print("Episodio Atual")
            print(ep)
            print("Episodios Totais")
            print(epochs)
            if (ep) == epochs:
                best_actions = []
                for i in range(num_states):
                    best_action = np.argmax(q_learning.q_table[:,i])
                    print("State {}: Optimal Action {}".format(states[i], k_h[best_action]))
                q_learning.final_q_table('final_q_table.txt')
            break

                #best_actions = np.argmax(q_learning.q_table)
                #print("Best action: k_h = ", k_h[best_action])
