#! /usr/bin/env python3

import os
import sys
import json
import time
import rospy
import signal
import subprocess
import numpy as np
import pandas as pd
import random
from ros_client import Athena
from kinematics import Kinematics
from std_srvs.srv import Empty
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from datetime import datetime
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, GetModelStateRequest
from gazebo_msgs.srv import SetModelState, SetModelStateRequest, SetModelConfiguration, SetModelConfigurationRequest

class WalkFunction:
    def __init__(self):
        self.height = 0.3
        self.pos = np.zeros((6, 4), dtype=float)
        self.p0 = np.zeros((6, 4), dtype=float)
        self._tf = 1.0

    def p1_p(self, p0, id, t_matrix, d_h):

        r, t = np.eye(4, dtype=float), np.eye(4, dtype=float)
        for i in range(3):
            for j in range(3):
                r[i, j] = t_matrix[i, j]
        t[1, 3] = -0.045
        t[2, 3] = d_h
        transform = np.matmul(r, t)

        pf_stance = transform[:, 3]

        dt_t_p1 = pf_stance

        s = 0.045
        t1 = np.array([0, s/4, self.height/3, 0])
        t2 = np.array([0, (3/4)*s, self.height/3, 0])
        t3 = np.array([0, s, d_h, 0])

        # Control points
        p1 = t1 + p0
        p2 = t2 + p0
        pf = t3 + p0

        parameters = {'p0': p0, 'p1': p1, 'p2': p2, 'p3': pf, 'dt_t_p1': dt_t_p1, 'd_h': d_h}
        return parameters

    def p2_p(self, p0, id, t_matrix, d_h):
    #def p2_p(self, p0, id, delta_h):
        r, t = np.eye(4, dtype=float), np.eye(4, dtype=float)
        for i in range(3):
            for j in range(3):
                r[i, j] = t_matrix[i, j]
        t[1, 3] = -0.045
        t[2, 3] = d_h
        transform = np.matmul(r, t)

        pf_stance = transform[:, 3]

        dt_t_p2 = pf_stance

        s = 0.045
        t1 = np.array([0, s/4, self.height/3, 0])
        t2 = np.array([0, (3/4)*s, self.height/3, 0])
        t3 = np.array([0, s, d_h, 0])

        parameters = {'p0': p0, 't1': t1, 't2': t2, 't3': t3, 'dt_t_p2': dt_t_p2, 'd_h': d_h}
        return parameters

    def p1(self, params, t, contact, force, t_1, t_3, t_5):
        id = params.get('id')
        p0 = np.array(params.get('p0'))
        p1 = np.array(params.get('p1'))
        p2 = np.array(params.get('p2'))
        p3 = np.array(params.get('p3'))
        dt_t = np.array(params.get('dt_t_p1'))
        d_h = np.array(params.get('d_h'))

        if id == 0:
            toque = t_1
        elif id == 2:
            toque = t_3
        elif id == 4:
            toque = t_5

        if t <= self._tf:
            if t>= 1/5 * self._tf and contact is True: # and toque == 1:
                #s = -0.045 + (0.045 * t/self._tf)
                #dt_t_0 = dt_t[0] + (-dt_t[0] * t/self._tf)
                #dt_t_1 = dt_t[1] + (-dt_t[1] * t/self._tf)
                #dt_t_2 = dt_t[2] + (-dt_t[2] * t/self._tf)
                #self.pos[id] = np.array([self.pos[id][0] + dt_t_0, self.pos[id][1] + dt_t_1, self.pos[id][2] + dt_t_2, 1])
                self.pos[id][2] = self.pos[id][2]-0.004
                return Kinematics(id).ik(self.pos[id])
            else:
                self.pos[id] = np.power(1 - t, 3) * p0 + 3 * t * np.power((1 - t), 2) * p1 + 3 * np.power(t, 2) * (1 - t) * p2 + np.power(t, 3) * p3
                return Kinematics(id).ik(self.pos[id])
        else:
            s = -0.045
            #self.p0[id] = np.array([self.pos[id][0], self.pos[id][1] + (s/1.0) * (t-1.0), self.pos[id][2] + d_h, 1])
            self.p0[id] = np.array([self.pos[id][0] + (dt_t[0]/1.0) * (t-1), self.pos[id][1] + (dt_t[1]/1.0) * (t-1), self.pos[id][2] + (dt_t[2]/1.0) * (t-1), 1])
            return Kinematics(id).ik(self.p0[id])

    def p2(self, params, t, contact, force, t_2, t_4, t_6):
        id = params.get('id')
        p0 = params.get('p0')
        t1 = np.array(params.get('t1'))
        t2 = np.array(params.get('t2'))
        t3 = np.array(params.get('t3'))
        dt_t = np.array(params.get('dt_t_p2'))
        d_h = np.array(params.get('d_h'))

        if id == 1:
            toque_2 = t_2
        elif id == 3:
            toque_2 = t_4
        elif id == 5:
            toque_2 = t_6

        if t <= self._tf:
            s = -0.045
            #self.p0[id] = np.array([p0[0], p0[1] + (s/self._tf) * t, p0[2] + d_h, 1])
            self.p0[id] = np.array([p0[0] + (dt_t[0]/self._tf) * t, p0[1] + (dt_t[1]/self._tf) * t, p0[2] + (dt_t[2]/self._tf) * t, 1])
            return Kinematics(id).ik(self.p0[id])
        else:
            # Control Points
            p1 = t1 + self.p0[id]
            p2 = t2 + self.p0[id]
            p3 = t3 + self.p0[id]

            if (t-self._tf) >= 1/5 * self._tf and contact is True: # and toque_2 == 1:
                #s = -0.045 + (0.045 * (t - self._tf)/self._tf)
                #dt_t_0 = dt_t[0] + (-dt_t[0] * (t-self._tf)/self._tf)
                #dt_t_1 = dt_t[1] + (-dt_t[1] * (t-self._tf)/self._tf)
                #dt_t_2 = dt_t[2] + (-dt_t[2] * (t-self._tf)/self._tf)
                #self.pos[id] = np.array([self.pos[id][0] + dt_t_0, self.pos[id][1] + dt_t_1, self.pos[id][2] + dt_t_2, 1])
                #self.pos[id] = np.array([p0[0] + dt_t[0], p0[1] + dt_t[1], p0[2] + dt_t[2], 1])
                self.pos[id][2] = self.pos[id][2]-0.004
                return Kinematics(id).ik(self.pos[id])
            else:
                self.pos[id] = np.power(2.0 - t, 3) * self.p0[id] + 3 * (t - 1.0) * np.power((2.0 - t), 2) * p1 + 3 * np.power(t - 1.0, 2) * (2.0 - t) * p2 + np.power(t - 1.0, 3) * p3
                return Kinematics(id).ik(self.pos[id])

    '''def p_bezier(self, joints, id, height):
        # Requires the angular positions
        p0 = Kinematics(id).fk(joints)

        t1 = np.array([0, 0.045/4, self.height/3, 0])
        t2 = np.array([0, (3/4)*0.045, self.height/3, 0])
        t3 = np.array([0, 0.045, 0, 0])
        # Control points
        p1 = t1 + p0
        p2 = t2 + p0
        pf = t3 + p0

        parameters = {'p0': p0, 'p1': p1, 'p2': p2, 'p3': pf, 'height': height}
        return parameters

    def p_linear(self, joints, id, height):
        # Requires the angular positions
        p0 = Kinematics(id).fk(joints)
        parameters = {'p0': p0, 'height': height}
        return parameters

    def swing(self, params, t):
        id = params.get('id')
        p0 = np.array(params.get('p0'))
        p1 = np.array(params.get('p1'))
        p2 = np.array(params.get('p2'))
        p3 = np.array(params.get('p3'))

        self.pos[id] = np.power(1 - t, 3) * p0 + 3 * t * np.power((1 - t), 2) * p1 + 3 * np.power(t, 2) * (1 - t) * p2 + np.power(t, 3) * p3
        return Kinematics(id).ik(self.pos[id])

    def stance(self, params, t):
        id = params.get('id')
        p0 = params.get('p0')

        stride = -0.045
        pos = np.array([p0[0], p0[1] + (stride/self._tf) * t, p0[2], 1])
        return Kinematics(id).ik(pos)'''

class Env:
    def __init__(self):
        # ROS Client
        self.agent = Athena()
        self.joints = self.agent.joint_name_lst
        self.jp, self.error = {}, {} # two dicitionaries for the error and the ?
        for joint in self.joints:
            self.jp[joint] = 0.0
            self.error[joint] = 0.0

        # Gazebo shenanigans
        self.pause_proxy = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_proxy = rospy.ServiceProxy('/gazebo/unpause_physics',Empty)
        self.model_state_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.model_state_req = SetModelStateRequest()
        self.model_state_req.model_state = ModelState()
        self.model_state_req.model_state.model_name = 'hexapod'
        self.model_state_req.model_state.pose.position.x = 0.0

        random_number = random.uniform(0, 1)
        print("RANDOM NUMBER")
        print("RANDOM NUMBER")
        print(random_number)
        print("RANDOM NUMBER")
        print("RANDOM NUMBER")

        if random_number<=0.5:
            self.model_state_req.model_state.pose.position.y = 0.15 #0.38
        else:
            self.model_state_req.model_state.pose.position.y = 0.35
        self.model_state_req.model_state.pose.position.z = 0.0
        self.model_state_req.model_state.pose.orientation.x = 0.0
        self.model_state_req.model_state.pose.orientation.y = 0.0
        self.model_state_req.model_state.pose.orientation.z = 0.0
        self.model_state_req.model_state.pose.orientation.w = 0.0
        self.model_state_req.model_state.twist.linear.x = 0.0
        self.model_state_req.model_state.twist.linear.y = 0.0
        self.model_state_req.model_state.twist.linear.z = 0.0
        self.model_state_req.model_state.twist.angular.x = 0.0
        self.model_state_req.model_state.twist.angular.y = 0.0
        self.model_state_req.model_state.twist.angular.z = 0.0
        self.model_state_req.model_state.reference_frame = 'world'

        self.model_config_proxy = rospy.ServiceProxy('/gazebo/set_model_configuration',SetModelConfiguration)
        self.model_config_req = SetModelConfigurationRequest()
        self.model_config_req.model_name = 'hexapod'
        self.model_config_req.urdf_param_name = 'robot_description'
        self.model_config_req.joint_names = self.agent.joint_name_lst
        self.model_config_req.joint_positions = np.zeros(18, dtype=float)

        self.get_model_state_proxy = rospy.ServiceProxy('/gazebo/get_model_state',GetModelState)
        self.get_model_state_req = GetModelStateRequest()
        self.get_model_state_req.model_name = 'hexapod'
        self.get_model_state_req.relative_entity_name = 'world'

        self.reset_world = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        self.torso_width = 0.056/2
        self.h = 0
        self.mass = 2.1358

        self._time = np.arange(0, 2.05, 50e-3)
        #self.dt = 1.05

        self.walker = WalkFunction()

    def _pause(self):
        # pause physics
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause_proxy()
        except rospy.ServiceException:
            print('/gazebo/pause_physics service call failed')

        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.reset_world()
        except rospy.ServiceException:
            print('/gazebo/reset_simulation service call failed')

    def _reset(self):
        self._pause()
        # Set model's position from world
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            self.model_state_proxy(self.model_state_req)
        except rospy.ServiceException:
            print('/gazebo/set_model_state call failed')
        time.sleep(1)
        # Set model's joint config
        rospy.wait_for_service('/gazebo/set_model_configuration')
        try:
            self.model_config_proxy(self.model_config_req)
        except rospy.ServiceException:
            print('/gazebo/set_model_configuration call failed')
        time.sleep(1)
        # Unpause physics
        try:
           self.unpause_proxy()
        except rospy.ServiceException:
            print('/gazebo/unpause_physics service call failed')
        time.sleep(1)

        #Move Joints
        for i in self.jp.keys():
            self.jp[i] = 0.0
        print(self.jp)
        self.agent.set_angles(self.jp)
        time.sleep(1)

        self.h = self.agent.depth + self.torso_width
        self.altura = self.h
        print("ALTURA = "+str(self.altura))
        #time.sleep(1)

    def touch(self, id):
        state = self.agent.force[id].get('contact')
        return state

    def touch_force(self, id):
        t_force = self.agent.force[id].get('normal_force')
        return t_force


    def step(self, iteration): # isto antes tinha o step(self,d)
        #self.k_h = 0
        path = '/home/diogo/Desktop/Stability/'
        os.chdir(path)
        fnm = open('kh_value.json')
        params = json.load(fnm)
        d = params['kh_value']
        self.k_h = d # NAO ESQUECER DE POR d
        self.k_a = 0
        self.k_b = 0

        actuation, height, roll, pitch, tc_1_t, ctr_1_t, fti_1_t, tc_2_t, ctr_2_t, fti_2_t, F_1, F_2, F_3, F_4, F_5, F_6, H, Body_deg = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        cot, h, angular_vel, tc_1_torque, ctr_1_torque, fti_1_torque, tc_2_torque, ctr_2_torque, fti_2_torque, force_1, force_2, force_3, force_4, force_5, force_6, H_all, Body_deg_a = self.move_joints(iteration)
        actuation.append(cot)
        height.append(h)
        F_1.append(force_1)
        F_2.append(force_2)
        F_3.append(force_3)
        F_4.append(force_4)
        F_5.append(force_5)
        F_6.append(force_6)
        H.append(H_all)
        Body_deg.append(Body_deg_a)
        tc_1_t.append(tc_1_torque)
        ctr_1_t.append(ctr_1_torque)
        fti_1_t.append(fti_1_torque)
        tc_2_t.append(tc_2_torque)
        ctr_2_t.append(ctr_2_torque)
        fti_2_t.append(fti_2_torque)
        roll.append(angular_vel[0])
        pitch.append(angular_vel[1])

        return actuation, height, roll, pitch, tc_1_t, ctr_1_t, fti_1_t, tc_2_t, ctr_2_t, fti_2_t, F_1, F_2, F_3, F_4, F_5, F_6, H, Body_deg

        #actuation, height, roll = [], [], []
        #for i in range(2):
            # Move Joints
            #cot, h, angular_vel = self.move_joints(i)
            #actuation.append(cot)
            #height.append(h)
            #roll.append(angular_vel[0])
        #return self.rms(actuation), self.rms(height), self.rms(roll)

    def move_joints(self, iteration):
        perna = 0
        t_matrix, d_h, body_deg = self.posture(perna)
        #r, h, alpha = self.posture(perna)
        # Get the gait parameters
        # params, delta_h = self.tripod(r, h, alpha, iteration)
        params = self.tripod(t_matrix, d_h, iteration)
        #params = self.tripod(iteration)
        cot, height = 0.0, 0.0
        cot_total = 0
        power_values = 0
        total_power = 0
        total_power_t = 0
        t_1, t_2, t_3, t_4, t_5, t_6 = 0, 0, 0, 0, 0, 0
        Height = []
        F_1, F_2, F_3, F_4, F_5, F_6 = [], [], [], [], [], []
        previous_pos = self.get_torso()
        previous_time = rospy.get_time()
        tc_1_torque, ctr_1_torque, fti_1_torque, tc_2_torque, ctr_2_torque, fti_2_torque = [], [], [], [], [], []
        for n, t in enumerate(self._time):
            pos = np.zeros(18, dtype=float)

            for id in range(6):
                idx = [id, id + 6, id + 12]
                if params[id]['phase'] is False:
                    # Stance
                    angular_pos = self.walker.p2(params[id], t, self.touch(id), self.touch_force(id), t_2, t_4, t_6)
                else:
                    # SWing phase
                    angular_pos = self.walker.p1(params[id], t, self.touch(id), self.touch_force(id), t_1, t_3, t_5)

                for j, joint in enumerate(idx):
                    if j == 2:
                        pos[joint] = angular_pos[j] + np.pi/2
                    else:
                        pos[joint] = angular_pos[j]
                #print("perna " + str(id+1)) read
                #print(self.touch(id))
                #print(self.touch_force(id))

            for k, joint_name in enumerate(self.joints):
                self.jp[joint_name] = pos[k] #+ self.error[joint_name]

            self.agent.set_angles(self.jp)
            self.agent.rate.sleep()

            print(n)

            if 5<n<=20:
                if self.touch(0):
                    t_1 +=1
                if self.touch(2):
                    t_3 +=1
                if self.touch(4):
                    t_5 +=1

            if n>=25:
                if self.touch(1):
                    t_2 +=1
                if self.touch(3):
                    t_4 +=1
                if self.touch(5):
                    t_6 +=1

            #for i in range(6):
                #print(self.touch(i))

            F_1.append(self.touch_force(0)) # PERNA 1
            F_2.append(self.touch_force(1)) # PERNA 2
            F_3.append(self.touch_force(2)) # PERNA 3
            F_4.append(self.touch_force(3)) # PERNA 4
            F_5.append(self.touch_force(4)) # PERNA 5
            F_6.append(self.touch_force(5)) # PERNA 6

            current_pos = self.get_torso()
            current_time = rospy.get_time()
            power_values = sum(self.agent.power.values())
            tc_1_torque.append(self.agent.tc_joint_torque)
            ctr_1_torque.append(self.agent.ctr_joint_torque)
            fti_1_torque.append(self.agent.fti_joint_torque)

            tc_2_torque.append(self.agent.tc2_joint_torque)
            ctr_2_torque.append(self.agent.ctr2_joint_torque)
            fti_2_torque.append(self.agent.fti2_joint_torque)
            #vel = np.linalg.norm(current_pos - previous_pos)/(current_time - previous_time)
            #cot = sum(self.agent.power.values()) / (vel * 9.81 * self.mass)

            height = self.agent.depth + self.torso_width
            Height.append(height)
            total_power += power_values
        current_pos = self.get_torso()
        current_time = rospy.get_time()
        vel = np.linalg.norm(current_pos - previous_pos)/(current_time - previous_time)
        total_power_t = total_power/40
        cot = total_power_t / (vel * 9.81 * self.mass)
        print("VELOCIDADE = "+ str(vel))
        #cot = P_V / (vel * 9.81 * self.mass)
        print("ALTURA = "+str(height))
        return cot, height, self.agent.ang_v, tc_1_torque, ctr_1_torque, fti_1_torque, tc_2_torque, ctr_2_torque, fti_2_torque, F_1, F_2, F_3, F_4, F_5, F_6, Height, body_deg

    def posture(self, perna):
        # r_w, p_w = self.global_pos()
        # Get the ids for the limbs in the stance phase
        stance = []
        if perna == 0:
            stance = [1, 3, 5]
        else:
            stance = [0, 2 ,4]

        # Get the positions for the TC joints and end effectors
        values = self.agent.get_angles()
        foot_0 = {}
        tc_0 = {}
        path = '/home/diogo/Desktop/Stability/'
        robot_pos = self.get_torso()
        for id, num in enumerate(stance):
            joints = np.array([values.get('tc_' + str(num + 1)), values.get('ctr_' + str(num + 1))- np.pi/2, values.get('fti_' + str(num + 1)) - np.pi/2])

            foot_vec = Kinematics(num).fk(joints)
            # Remove the [1] in the [x, y, z, 1] array for both cases
            foot = np.delete(foot_vec, [3])
            foot_0[id]={'foot': foot}
            #tc = np.delete(tc_vec, [3])
            # Convert to global positions
            #foot_w = p_w + r_w.dot(foot)
            #tc_w = p_w + r_w.dot(tc)
            # Save positions
            #positions[id] = {'foot': foot_w, 'tc': tc_w} # Coordenadas globais do pé e da junta TC- até aqui tudo bem

        # Get the norm vector for the ground

        v1 = foot_0[2]['foot'] - foot_0[0]['foot']
        v2 = foot_0[1]['foot'] - foot_0[0]['foot']
        u_ground = np.cross(v1, v2)
        # Get the norm vector for the Tc Plane
        #tc_v1 = tc_0[2]['tc'] - tc_0[0]['tc']
        #tc_v2 = tc_0[1]['tc'] - tc_0[0]['tc']
        #u_torso = np.cross(tc_v1, tc_v2)

        body_incl = self.agent.get_pitch_angle()
        print(body_incl)
        body_incl_deg = np.rad2deg(body_incl)
        #print("INCLINACAO EM GRAUS")
        #print("INCLINACAO EM GRAUS")
        #print(body_incl_deg)
        #print("INCLINACAO EM GRAUS")
        #print("INCLINACAO EM GRAUS")
        #print(body_incl)
        #body_incl_deg = np.rad2deg(body_incl)
        #print(body_incl_deg)
        '''print("INCLINACAO DO BODY")
        print("INCLINACAO DO BODY")
        body_incl = np.arctan2(robot_pos[2], np.sqrt(robot_pos[0]**2 + robot_pos[1]**2))
        print(body_incl)
        body_incl_deg = np.rad2deg(body_incl)
        print(body_incl_deg)
        print("INCLINACAO DO BODY")
        print("INCLINACAO DO BODY")'''
        # unitary vector to find the norm vector of the plane regarding roll and pitch
        eta = np.array([0, 1])
        # Unitary vector for the Z-axis
        #u_z = np.array([0, 0, 1])
        # Rotation in X-Axis

        u_yz = np.delete(u_ground, [0])
        roll = 0.0 if np.linalg.norm(u_yz) == 0.0 else np.arccos((np.dot(u_yz, eta)) / (np.linalg.norm(u_yz) * np.linalg.norm(eta)))
        if roll >= np.pi/2:
            roll = np.pi - roll
        if u_yz[0] < 0:
            roll *= -1

        #Rotation in Y-Axis
        u_xz = np.delete(u_ground, [1])
        pitch = 0.0 if np.linalg.norm(u_xz) == 0.0 else np.arccos((np.dot(u_xz, eta)) / (np.linalg.norm(u_xz) * np.linalg.norm(eta)))
        if pitch >= np.pi/2:
            pitch = np.pi - pitch
        if u_xz[0] < 0:
            pitch *= -1

        # Height adjustment
        ir = self.agent.depth + self.torso_width

        Rx, Ry = np.eye(3, dtype=float), np.eye(3, dtype=float)
        Rx[1, 1] = np.cos(roll*self.k_a)
        Rx[1, 2] = -np.sin(roll*self.k_a)
        Rx[2, 1] = np.sin(roll*self.k_a)
        Rx[2, 2] = np.cos(roll*self.k_a)

        Ry[0, 0] = np.cos(pitch*self.k_b)
        Ry[0, 2] = np.sin(pitch*self.k_b)
        Ry[2, 0] = -np.sin(pitch*self.k_b)
        Ry[2, 2] = np.cos(pitch*self.k_b)

        T_Matrix = np.matmul(Rx, Ry)
        d_h = (self.k_h)*(ir - self.h)

        return T_Matrix, d_h, body_incl_deg
        #alpha = np.pi - np.arccos((np.dot(u_ground, u_z)) / (np.linalg.norm(u_ground) * np.linalg.norm(u_z)))
        #beta = np.pi - np.arccos((np.dot(u_torso, u_z)) / (np.linalg.norm(u_torso) * np.linalg.norm(u_z)))
        # Unitary vector for the Y-axis
        #u_y = np.array([0, 1, 0])
        #alpha_y = np.pi/2 - np.arccos((np.dot(u_ground, u_y)) / (np.linalg.norm(u_ground) * np.linalg.norm(u_y)))
        #beta_y = np.pi/2 - np.arccos((np.dot(u_torso, u_y)) / (np.linalg.norm(u_torso) * np.linalg.norm(u_y)))
        #print(np.rad2deg(alpha), np.rad2deg(beta), ' Diff: ', np.rad2deg(alpha) - np.rad2deg(beta))
        # Transformation matrix
        #r_x = np.eye(3)  # Matriz de rotação -do roll
        #r_x[1, 1] = np.cos(-(self.k_a) * (alpha-beta))
        #r_x[2, 1] = np.sin(-(self.k_a) * (alpha-beta))
        #r_x[1, 2] = - np.sin(-(self.k_a) * (alpha-beta))
        #r_x[2, 2] = np.cos(-(self.k_a) * (alpha-beta))

    def get_torso(self): # ToDo: Add torso height with infrared sensor
        rospy.wait_for_service('/gazebo/get_model_state')
        model_state = self.get_model_state_proxy(self.get_model_state_req)
        return np.array([model_state.pose.position.x, model_state.pose.position.y, self.agent.depth + self.torso_width])

    def tripod(self, t_matrix, d_h, iteration):
        legs = {}

        values = self.agent.get_angles()
        for i in range(6):
            joints = np.array([values.get('tc_' + str(i + 1)), values.get('ctr_' + str(i + 1)), values.get('fti_' + str(i + 1)) - np.pi/2])
            if i % 2 == 0:
                # Dictionary containing all info about the limbs actuation
                data = {'id': i, 'phase': True}
                if iteration == 0:
                    p = Kinematics(i).fk(joints)
                else:
                    p = self.walker.p0[i]

                parameters = self.walker.p1_p(p, i, t_matrix, d_h)
                legs[i] = {**data, **parameters}
            else:
                # Dictionary containing all info about the limbs actuation
                data = {'id': i, 'phase': False}
                if iteration == 0:
                    p = Kinematics(i).fk(joints)
                else:
                    p = self.walker.pos[i]
                parameters = self.walker.p2_p(p, i, t_matrix, d_h)
                #parameters = self.walker.p2_p(p, i, variable)
                legs[i] = {**data, **parameters}
        return legs

    def rms(self, data):
        n = len(data)
        square = 0
        for i in range(n):
            square += np.power(data[i], 2)
        return np.sqrt(square/float(n))

if __name__ == '__main__':
    path = '/home/diogo/Desktop/Stability/'
    os.chdir(path)
    # Open JSON file
    #f = open('parameters.json')
    # Load parameters as dictionary É PRECISO MUDAR ISTO COMPLETAMENTE
    #params = json.load(f) ######

    hexapod = Env()
    try:
        results = []
        actuation_v, height_v, roll_v, pitch_v, tc_1_t_v, ctr_1_t_v, fti_1_t_v, tc_2_t_v, ctr_2_t_v, fti_2_t_v, F_1_v, F_2_v, F_3_v, F_4_v, F_5_v, F_6_v, H_V, Body_deg_V = [], [], [], [], [], [], [], [], [], [], [], [],[], [], [], [], [], []
        height_evaluate = []
        height_important = []
        hexapod._reset()
        time.sleep(2)
        dodge = 0

        for i in range(15):
            if dodge == 0:
                results.append(hexapod.step(i))
                y_position = hexapod.agent.get_y_position()
                print(y_position)
                if  0.35<=y_position<=0.9:
                    height_important.extend(results[i][1])
                height_evaluate.extend(results[i][1])
                if height_evaluate[-1]<0.1 or height_evaluate[-1]>0.25:
                    dodge = 1
            else:
            #print(results)
                pass

        for i in results:
            actuation_v.extend(i[0])
            height_v.extend(i[1])
            roll_v.extend(i[2])
            pitch_v.extend(i[3])
            tc_1_t_v.extend(i[4])
            ctr_1_t_v.extend(i[5])
            fti_1_t_v.extend(i[6])
            tc_2_t_v.extend(i[7])
            ctr_2_t_v.extend(i[8])
            fti_2_t_v.extend(i[9])
            F_1_v.extend(i[10])
            F_2_v.extend(i[11])
            F_3_v.extend(i[12])
            F_4_v.extend(i[13])
            F_5_v.extend(i[14])
            F_6_v.extend(i[15])
            H_V.extend(i[16])
            Body_deg_V.extend(i[17])

        print("HEIGHT EVALUATE")
        print("HEIGHT EVALUATE")
        print(height_evaluate)
        print("HEIGHT IMPORTANT")
        print("HEIGHT IMPORTANT")
        print(height_important)
        actuation_RMS = hexapod.rms(actuation_v)
        height_RMS = hexapod.rms(height_v)
        roll_RMS = hexapod.rms(roll_v)
        pitch_RMS = hexapod.rms(pitch_v)

        body_incl_RMS = hexapod.rms(Body_deg_V[3:])
        height_difference = []
        reward = 0
        for i in range(len(height_important)-1):
            diff_height = height_important[i+1] - height_important[i]
            height_difference.append(abs(diff_height))
            reward += - abs(diff_height) * 1.5
        if dodge == 1:
            reward = reward - 20

        if abs(height_v[0] - height_v[-1])<= 0.6:
            reward = reward + 10


        with open('environment.txt', 'r') as file:
            env_inclination = file.read()
            env = int(env_inclination)

        Slopes = [3, 6, 9, 12, 15]
        slope_v = Slopes[env-1]

        '''print("INCLINACAO REAL")
        print("INCLINACAO REAL")
        print("INCLINACAO REAL")
        #print(slope_v)
        print(6)
        print("INCLINACAO DO BODY")
        print("INCLINACAO DO BODY")
        print("INCLINACAO DO BODY")'''
        #print(body_incl_RMS)
        #print(abs(6 - body_incl_RMS))
        if slope_v == 3:
            reward = reward
        elif slope_v == 6:
            if abs(slope_v - body_incl_RMS) <= 1:
                reward = reward + 5
            else:
                reward = reward - (abs(slope_v - body_incl_RMS)) * 2
        elif slope_v == 9:
            if abs(slope_v - body_incl_RMS) <= 1.5:
                reward = reward + 5
            else:
                reward = reward - (abs(slope_v - body_incl_RMS)) * 2
        elif slope_v == 12:
            if abs(slope_v - body_incl_RMS) <= 2.5:
                reward = reward + 5
            else:
                reward = reward - (abs(slope_v - body_incl_RMS)) * 2
        elif slope_v == 15:
            if abs(slope_v - body_incl_RMS) <= 3:
                reward = reward + 5
            else:
                reward = reward - (abs(slope_v - body_incl_RMS)) * 2

        # tirar o CoT
        output = {'Actuation_RMS': actuation_RMS, 'Height_RMS': height_RMS, 'Roll_RMS': roll_RMS, 'Pitch_RMS': pitch_RMS}

        with open ('iteration.txt', 'r') as file:
            file_contents = file.read()


        if int(file_contents) == 1:
            log_ = []
            logs = pd.DataFrame(log_, columns=['Episode', 'Reward', 'k_h', 'COT_RMS','Height_RMS','Roll_RMS', 'Pitch_RMS'])
            data = [int(file_contents), reward, hexapod.k_h, actuation_RMS, height_RMS, roll_RMS, pitch_RMS]
            row = pd.DataFrame([data], columns=['Episode', 'Reward', 'k_h', 'COT_RMS','Height_RMS','Roll_RMS', 'Pitch_RMS'])
            frames = [logs, row]
            logs = pd.concat(frames)
            fname = 'Reward_episode.csv'
            pd.DataFrame(logs).to_csv(path + fname)
        else:
            df = pd.read_csv('Reward_episode.csv')
            new_row = {'Episode': int(file_contents), 'Reward': reward, 'k_h': hexapod.k_h, 'COT_RMS': actuation_RMS, 'Height_RMS': height_RMS, 'Roll_RMS': roll_RMS,'Pitch_RMS': pitch_RMS}

            df = df.append(new_row, ignore_index=True)
            df.to_csv('Reward_episode.csv', index=False)

        filename_string = "Data_Iteration_Number"
        filename_int = int(file_contents)

        filename = f"{filename_string}_{filename_int}.txt"
        filenamej = f"{filename_string}_{filename_int}.json"

        if (int(file_contents) + 1) % 100 == 0:
            with open(filename, 'w') as file:
                file.write("EPISODE NUMBER: \n")
                file.write("EP = " + str(int(file_contents)) + "\n")
                file.write("REWARD VALUE: \n")
                file.write(" = " + str(reward) + "\n")
                file.write("COT_RMS: \n")
                file.write(" = " + str(actuation_RMS) + "\n")
                file.write("Height_RMS: \n")
                file.write(" = " + str(height_RMS) + "\n")
                file.write("Roll_RMS: \n")
                file.write(" = " + str(roll_RMS) + "\n")
                file.write("Pitch_RMS: \n")
                file.write(" = " + str(pitch_RMS) + "\n")
                file.write("Body Inclination: \n")
                file.write(" = " + str(body_incl_RMS) + "\n")
                file.write("Real Ramp Slope: \n")
                file.write(" = " + str(slope_v) + "º \n")

            data_to_save = {"ep_number": int(file_contents), "reward_value": reward, "Cot_RMS": actuation_RMS, "Height_RMS": height_RMS, "Roll_RMS": roll_RMS, "Pitch_RMS": pitch_RMS, "Height_Values": height_v}
            with open(filenamej, 'w') as json_file:
                json.dump(data_to_save, json_file)
                json_file.write("\n")
        print(reward)
            #results.append(hexapod.step()) # Parte do append tem que ser alterada
        #df = pd.DataFrame.from_dict(hexapod.force)
        #output = {'result': hexapod.rms(results)}
        # saving the dataframe
        #df.to_csv('force_leg6.csv', index=False, header=True)
        # Serializing json
        reward_safe = {'reward': reward}
        reward_file = 'reward_value.json'
        with open(reward_file, 'w') as outfile:
            json.dump(reward_safe, outfile)

        fn = 'state_values_update.json'
        print('--------- File saved in: ', fn)
        with open(fn, 'w') as outfile:
            json.dump(output, outfile)
        time.sleep(4)

        hexapod._pause()
        print('Ending simulation...')
        rospy.signal_shutdown('Simulation ended!')
        # Kill the roslaunch process
        # Get a list of all active nodes
        node_list = subprocess.check_output(["rosnode", "list"]).decode().strip().split("\n")

        # Kill each node in the list
        for node in node_list:
            print('Shutting down node: ', node)
            subprocess.call(["rosnode", "kill", node])
        '''print("412")
        subprocess.run(["pkill", "-f", "hexapod.launch"])
        print("12311231312312312312123s")'''

        '''subprocess.run(["killall", "-9", "gzserver"])
        subprocess.run(["killall", "-9", "gzclient"])
        sys.exit()'''

    except rospy.ROSInterruptException:
        pass
