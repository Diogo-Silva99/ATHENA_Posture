#! /usr/bin/env python3

import numpy as np

class Kinematics:
    def __init__(self, id):
        self.l1 = 0.063875
        self.l2 = 0.080
        self.l3 = 0.114308 + 0.04

        self._id = id

        self.torso_width = 0.2
        self.torso_depth = 0.12

        if id == 0 or id == 3:
            self.a = self.torso_depth / 2
        else:
            self.a = np.sqrt(np.power(self.torso_width / 2, 2) + np.power(self.torso_depth / 2, 2))


        ro = np.arctan((self.torso_width / 2) / (self.torso_depth / 2))
        self.tau = [0, ro, np.pi - ro, np.pi, np.pi + ro, -ro]

    def fk(self, arg):
        return np.array([np.cos(self.tau[self._id] + arg[0]) * (self.l3 * np.cos(arg[1] + arg[2]) + self.l2 * np.cos(arg[1]) + self.l1) + self.a * np.cos(self.tau[self._id]), np.sin(self.tau[self._id] + arg[0]) * (self.l3 * np.cos(arg[1] + arg[2]) + self.l2 * np.cos(arg[1]) + self.l1) + self.a * np.sin(self.tau[self._id]), self.l3 * np.sin(arg[1] + arg[2]) + self.l2 * np.sin(arg[1]), 1])

    '''def fk(self, arg):
        return np.array([np.cos(arg[0]) * (self.l3 * np.cos(arg[1] + arg[2]) + self.l2 * np.cos(arg[1]) + self.l1), np.sin(arg[0]) * (self.l3 * np.cos(arg[1] + arg[2]) + self.l2 * np.cos(arg[1]) + self.l1), self.l3 * np.sin(arg[1] + arg[2]) + self.l2 * np.sin(arg[1]), 1])'''

    def m_04(self, arg):
        # Transformation matrix between the COM and the foot
        m = np.eye(4, dtype=float)
        m[0, 0] = np.cos(self.tau[self._id] + arg[0]) * np.cos(arg[1] + arg[2])
        m[0, 1] = - np.cos(self.tau[self._id] + arg[0]) * np.sin(arg[1] + arg[2])
        m[0, 2] = np.sin(self.tau[self._id] + arg[0])
        m[0, 3] = np.cos(self.tau[self._id] + arg[0]) * (self.l3 * np.cos(arg[1] + arg[2]) + self.l2 * np.cos(arg[1]) + self.l1) + self.a * np.cos(self.tau[self._id])

        m[1, 0] = np.sin(self.tau[self._id] + arg[0]) * np.cos(arg[1] + arg[2])
        m[1, 1] = - np.sin(self.tau[self._id] + arg[0]) * np.sin(arg[1] + arg[2])
        m[1, 2] = - np.cos(self.tau[self._id] + arg[0])
        m[1, 3] = np.sin(self.tau[self._id] + arg[0]) * (self.l3 * np.cos(arg[1] + arg[2]) + self.l2 * np.cos(arg[1]) + self.l1) + self.a * np.sin(self.tau[self._id])

        m[2, 0] = np.sin(arg[1] + arg[2])
        m[2, 1] = np.cos(arg[1] + arg[2])
        m[2, 2] = 0
        m[2, 3] = self.l3 * np.sin(arg[1] + arg[2]) + self.l2 * np.sin(arg[1])
        return m

    def transformation_matrix(self, arg):
        '''
        :param arg: [alpha, theta, a, d]
        :return: transformation matrix
        '''
        m = np.zeros((4, 4), dtype=float)
        m[0, 0] = np.cos(arg[1])
        m[0, 1] = - np.sin(arg[1]) * np.cos(arg[0])
        m[0, 2] = np.sin(arg[1]) * np.sin(arg[0])
        m[0, 3] = arg[2] * np.cos(arg[1])
        m[1, 0] = np.sin(arg[1])
        m[1, 1] = np.cos(arg[1]) * np.cos(arg[0])
        m[1, 2] = - np.cos(arg[1]) * np.sin(arg[0])
        m[1, 3] = arg[2] * np.sin(arg[1])
        m[2, 1] = np.sin(arg[0])
        m[2, 2] = np.cos(arg[0])
        m[2, 3] = arg[3]
        m[3, 3] = 1
        return m

    def tc(self):
        return self.transformation_matrix([0.0, self.tau[self._id], self.a, 0.0])

    '''def ctr(self, arg):
        return np.array([self.l1 * np.cos(self.tau[self._id] + arg[0]) + self.a * np.cos(self.tau[self._id]), self.l1 * np.sin(self.tau[self._id] + arg[0]) + self.a * np.sin(self.tau[self._id]), 0, 1])

    def fti(self, arg):
        return np.array([(self.l2 * np.cos(arg[1]) + self.l1) * np.cos(self.tau[self._id] + arg[0]) + self.a * np.cos(self.tau[self._id]), (self.l2 * np.cos(arg[1]) + self.l1) * np.sin(self.tau[self._id] + arg[0]) + self.a * np.sin(self.tau[self._id]), self.l2 * np.sin(arg[1]), 1])
    '''

    def tc_ang(self, foot):
        m_torso_tc = self.tc()
        foot_tc = np.linalg.inv(m_torso_tc).dot(foot.T)
        tc_position = foot_tc[: 3]
        return tc_position

    def ik(self, foot):
        m_torso_tc = self.tc()
        foot_tc = np.linalg.inv(m_torso_tc).dot(foot.T)
        # Coordenadas do Pé com Tc como referência

        # Using the TC joint as reference
        theta_1 = np.arctan2(foot_tc[1], foot_tc[0])

        # Get the CTr position but using the body as reference
        ctr = np.array([self.l1 * np.cos(theta_1 + self.tau[self._id]) + self.a * np.cos(self.tau[self._id]), self.l1 * np.sin(theta_1 + self.tau[self._id]) + self.a * np.sin(self.tau[self._id]), 0])
        # So using the ctr and foot_vec that is wrt body makes sense
        foot_vec = np.delete(foot, [3])

        # Auxiliary vector
        r = np.linalg.norm(foot_vec - ctr)
        beta = np.arccos(np.clip((- np.power(r, 2) + np.power(self.l2, 2) + np.power(self.l3, 2)) / (2 * self.l2 * self.l3), -1.0, 1.0))

        theta_3 = - (np.pi - beta)

        beta = np.arccos(np.clip((- np.power(self.l3, 2) + np.power(self.l2, 2) + np.power(r, 2)) / (2 * r * self.l2), -1.0, 1.0))
        gamma = np.arcsin(np.clip((ctr[2] - foot_vec[2]) / r, -1.0, 1.0))
        theta_2 = beta - gamma
        return np.array([theta_1, theta_2, theta_3])


    '''def ik(self, foot, torso, rotation):
        if len(foot) < 4:
            foot = np.concatenate((foot, [1]), axis=0)
        torso = np.concatenate((torso, [1]), axis=0)
        # Get the transformation matrix for the torso's position w.r.t the world
        m_torso = np.eye(4)
        for i in range(3):
            for j in range(3):
                m_torso[i, j] = rotation[i, j]
        m_torso[0, 3] = torso[0]
        m_torso[1, 3] = torso[1]
        m_torso[2, 3] = torso[2]
        # Get the foot's coordinates w.r.t. the TC joint
        m_torso_tc = np.matmul(m_torso, self.tc())
        foot_tc = np.linalg.inv(m_torso_tc).dot(foot)

        # Using the TC joint as reference
        theta_1 = np.arctan2(foot_tc[1], foot_tc[0])

        # Get the CTr position w.r.t. the TC joint
        ctr = np.array([self.l1 * np.cos(theta_1), self.l1 * np.sin(theta_1), 0])
        m_ctr = np.eye(4)
        m_ctr[0, 3] = self.l1 * np.cos(theta_1 + self.tau[self._id]) + self.a * np.cos(self.tau[self._id])
        m_ctr[1, 3] = self.l1 * np.sin(theta_1 + self.tau[self._id]) + self.a * np.sin(self.tau[self._id])
        m_ctr[2, 3] = 0
        ctr = m_ctr.dot(torso.T)
        # ctr = np.matmul(m_torso, m_ctr)[:, 3]
        foot_vec = np.delete(foot_tc, [3])
        #ctr_vec = np.delete(ctr, [3])

        # Auxiliary vector
        r = np.linalg.norm(foot_vec - ctr)
        beta = np.arccos(np.clip((- np.power(r, 2) + np.power(self.l2, 2) + np.power(self.l3, 2)) / (2 * self.l2 * self.l3), -1.0, 1.0))

        theta_3 = - (np.pi - beta)

        beta = np.arccos(np.clip((- np.power(self.l3, 2) + np.power(self.l2, 2) + np.power(r, 2)) / (2 * r * self.l2), -1.0, 1.0))
        gamma = np.arcsin(np.clip((ctr[2] - foot_vec[2]) / r, -1.0, 1.0))
        theta_2 = beta - gamma
        return np.array([theta_1, theta_2, theta_3])'''
