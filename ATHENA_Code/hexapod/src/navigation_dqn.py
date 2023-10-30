#! /usr/bin/env python3

import os
import sys
import torch
import rospy
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from environment import MazeEnvironment


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Exploit:
    def __init__(self, env, dqn_path):
        self.env = env
        self.state_dim = self.env.obs_space
        self.action_dim = self.env.action_space

        self.model = torch.load(dqn_path)#DQN(self.action_dim, self.state_dim)

        self.log_ = []
        self.logs = pd.DataFrame(self.log_, columns=['reward'])
        self.action_logs = pd.DataFrame(self.log_, columns=['episode', 'action', 'cot'])

    def exploit(self, max_episodes=100):
        rewards_history, losses_history = [0.0], [0.0]
        for ep in range(max_episodes):
            next_obs, done = self.env.reset()
            while not done:
                obs = next_obs.copy()
                action = self.get_action(obs)

                next_obs, reward, done = self.env.step()
                rewards_history[ep] += reward

                actions = [ep, action]
                action_row = pd.DataFrame([actions], columns=['episode', 'action', 'cot'])
                frames = [self.action_logs, action_row]
                self.action_logs = pd.concat(frames)

            if ep < max_episodes:
                rewards_history.append(0.0)
            print("Episode: %04d, Reward: %03d" % (ep, rewards_history[ep]))

            row_ = [rewards_history[ep]]
            row = pd.DataFrame([row_], columns=['reward'])
            frames = [self.logs, row]
            self.logs = pd.concat(frames)

        print('Saving model ...')
        results_path = path + '/gamma_' + str(self.gamma) + '.pth'
        torch.save(self.model, results_path)

        return self.logs, self.action_logs

    def get_action(self, obs):
        with torch.no_grad():
            return self.model(torch.tensor(obs).to(device).unsqueeze(0)).max(1)[1].cpu()


class DQN(nn.Module):
    def __init__(self, action_dim, state_dim, hidden_layer=50):
        super(DQN, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_layer).to(device)
        self.l2 = nn.Linear(hidden_layer, action_dim).to(device)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

class Agent:
    def __init__(self, env, gamma, exploit, lr=0.007, batch_size=64):
        self.env = env
        self.state_dim = self.env.obs_space
        self.action_dim = self.env.action_space

        self.path = path + 'model.pth'

        self.log_ = []

        if exploit is False:
            self.model = DQN(self.action_dim, self.state_dim)
            self.memory = ReplayMemory(10000)
            self.optimizer = optim.Adam(self.model.parameters(), lr)

            self.gamma = gamma
            self.batch_size = batch_size


            self.logs = pd.DataFrame(self.log_, columns=['reward'])
            self.action_logs = pd.DataFrame(self.log_, columns=['episode', 'action', 'cot'])
        else:
            self.model = torch.load(self.path)
            self.logs_exploit = pd.DataFrame(self.log_, columns=['reward'])
            self.action_logs_exploit = pd.DataFrame(self.log_, columns=['episode', 'action', 'cot'])


    def train(self, max_episodes=1000):
        rewards_history, losses_history = [0.0], [0.0]
        for ep in range(max_episodes):
            eps = 0.1 * np.power(0.99, ep)
            next_obs, done = self.env.reset()
            while not done:
                obs = next_obs.copy()
                action = self.get_action(obs, eps)

                next_obs, reward, done, cot = self.env.step(action[0])
                self.memory.push((obs, action[0], next_obs, reward, int(done)))
                rewards_history[ep] += reward
                loss = self.learn()
                #losses_history[ep] += loss
                actions = [ep, action, cot]
                action_row = pd.DataFrame([actions], columns=['episode', 'action', 'cot'])
                frames = [self.action_logs, action_row]
                self.action_logs = pd.concat(frames)

            if ep < max_episodes:
                rewards_history.append(0.0)
                losses_history.append(0.0)
            rospy.loginfo("Episode: %04d, Reward: %03d" % (ep, rewards_history[ep]))

            row_ = [rewards_history[ep]]
            row = pd.DataFrame([row_], columns=['reward'])
            frames = [self.logs, row]
            self.logs = pd.concat(frames)

        rospy.loginfo('Saving model ...')
        results_path = path + 'model.pth'
        torch.save(self.model, results_path)

        return self.logs, self.action_logs

    def get_action(self, obs, eps):
        if random.random() > eps:
            with torch.no_grad():
                return self.model(torch.tensor(obs).to(device).unsqueeze(0)).max(1)[1].cpu().detach().numpy()
        else:
            return np.array([random.randrange(self.action_dim)])

    def learn(self):
        if len(self.memory) < self.batch_size:
            pass
        else:
            transitions = self.memory.sample(self.batch_size)
            batch_obs, batch_action, batch_next_obs, batch_reward, batch_done = zip(*transitions)

            batch_obs = torch.tensor(np.array(batch_obs)).to(device)
            batch_action = torch.tensor(np.array(batch_action)).to(device)
            batch_reward = torch.tensor(np.array(batch_reward)).to(device)
            batch_next_obs = torch.tensor(np.array(batch_next_obs)).to(device)
            batch_done = torch.tensor(np.array(batch_done)).to(device)

            current_q_values = self.model(batch_obs).gather(1, batch_action.unsqueeze(1)).squeeze(1)
            max_next_q_values = self.model(batch_next_obs).max(1)[1]
            expected_q_values = batch_reward + (1 - batch_done) * (self.gamma * max_next_q_values.detach())
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(current_q_values.float(), expected_q_values.float())
            loss.backward()
            self.optimizer.step()

    def exploit(self, max_episodes=15):
        rewards_history, losses_history = [0.0], [0.0]
        for ep in range(max_episodes):
            next_obs, done = self.env.reset()
            while not done:
                obs = next_obs.copy()
                #action = self.action(obs)
                action = 0
                next_obs, reward, done, cot = self.env.step(action)
                rewards_history[ep] += reward
                actions = [ep, action, cot]
                action_row = pd.DataFrame([actions], columns=['episode', 'action', 'cot'])
                frames = [self.action_logs_exploit, action_row]
                self.action_logs_exploit = pd.concat(frames)

            if ep < max_episodes:
                rewards_history.append(0.0)
            row_ = [rewards_history[ep]]
            row = pd.DataFrame([row_], columns=['reward'])
            frames = [self.logs_exploit, row]
            self.logs_exploit = pd.concat(frames)
            rospy.loginfo("Exploit, Episode: %04d, Reward: %03d" % (ep, rewards_history[ep]))

            row_ = [rewards_history[ep]]
            row = pd.DataFrame([row_], columns=['reward'])
            frames = [self.logs_exploit, row]
            self.logs = pd.concat(frames)

        rospy.loginfo('Saving model ...')
        results_path = path + '/exploit.pth'
        torch.save(self.model, results_path)

        return self.logs_exploit, self.action_logs_exploit

    def action(self, obs):
        with torch.no_grad():
            return self.model(torch.tensor(obs).to(device).unsqueeze(0)).max(1)[1].cpu()



if __name__ == '__main__':
    path = '/home/diogo/Desktop/Testes' #+ datetime.now().strftime("%Y_%m_%d__%I_%M_%S_%p")
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    # Use CUDA if available
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    torch.set_default_dtype(torch.float64)

    rospy.loginfo('Instantiating Environment')
    env = MazeEnvironment()

    exploit = False
    agent = Agent(env, 0.999, exploit)

    rospy.loginfo('Walker Ready')
    try:
        if exploit is False:
            data, actions = agent.train()

            pd.DataFrame(data).to_csv(path + '/results.csv')
            pd.DataFrame(actions).to_csv(path + '/actions.csv')

            rospy.loginfo('Ready to exploit model ...')
        else:
            rospy.loginfo('Exploiting ...')
            data, actions = agent.exploit()
            pd.DataFrame(data).to_csv(path + '/results_exploit.csv')
            pd.DataFrame(actions).to_csv(path + '/actions.csv')


    except rospy.ROSInterruptException:
        rospy.logerr("ROS Interrupt Exception!")
        pass

    env.pause_()
    rospy.signal_shutdown('Simulation ended!')
