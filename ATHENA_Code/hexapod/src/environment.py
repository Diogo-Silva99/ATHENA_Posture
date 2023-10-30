#! /usr/bin/env python3

import os
import time
import rospy
import signal
import sys
import subprocess
import numpy as np
import pandas as pd
from ros_client import Athena
from std_srvs.srv import Empty
from kinematics import Kinematics
from scipy.integrate import odeint
from trajectories import WalkFunction
from gazebo_msgs.msg import ModelState
from scipy.spatial.transform import Rotation as R
from gazebo_msgs.srv import GetModelState, GetModelStateRequest
from gazebo_msgs.srv import SetModelState, SetModelStateRequest, SetModelConfiguration, SetModelConfigurationRequest

class Environment:
    def __init__(self, id):
        #print("3")
        self.agent = Athena() #Chama a class Athena no ros_client.py

        self.pause_proxy = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_proxy = rospy.ServiceProxy('/gazebo/unpause_physics',Empty)
        self.model_state_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.model_state_req = SetModelStateRequest()
        self.model_state_req.model_state = ModelState()
        self.model_state_req.model_state.model_name = 'hexapod'
        self.model_state_req.model_state.pose.position.x = 0.0
        self.model_state_req.model_state.pose.position.y = 0.0
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

        # To get a estimate value for the robot height
        self.torso_width = 0.056/2
        self.mass = 2.1358

        self._time = np.arange(0, 1.05, 50e-3)
        self.dt = 1.05

        self.walker = WalkFunction(id) # Chama o ficheiro trajectories.py
        #print("4")
    def pause_(self):
        # pause physics
        print("650")
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            print("660")
            self.pause_proxy()
            print("680")
            os.system("kill $(ps aux | grep 'roscore' | awk '{print $2}')")
        except rospy.ServiceException:
            print('/gazebo/pause_physics service call failed')
        print("760")
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            print("870")
            self.reset_world()
            print("880")
            os.system("kill $(ps aux | grep 'roscore' | awk '{print $2}')")
        except rospy.ServiceException:
            print('/gazebo/reset_simulation service call failed')
        print("666")

    def reset(self):
        # pause physics
        print("20")
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            print("70")
            self.pause_proxy()
        except rospy.ServiceException:
            print('/gazebo/pause_physics service call failed')
        print("99")
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            print("500")
            self.reset_world()
        except rospy.ServiceException:
            print('/gazebo/reset_simulation service call failed')

        # Set model's position from world
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            self.model_state_proxy(self.model_state_req)
        except rospy.ServiceException:
            print('/gazebo/set_model_state call failed')
        # Set model's joint config
        rospy.wait_for_service('/gazebo/set_model_configuration')
        try:
            self.model_config_proxy(self.model_config_req)
        except rospy.ServiceException:
            print('/gazebo/set_model_configuration call failed')

        # Unpause physics
        try:
            self.unpause_proxy()
        except rospy.ServiceException:
            print('/gazebo/unpause_physics service call failed')

        # Move Joints
        self.agent._pub_joints.reset() #Vai ao ficheiro ros_client.py- Class Joints reseta
        self.agent.rate.sleep()

    def get_torso(self): # ToDo: Add torso height with infrared sensor
        #print("5")
        rospy.wait_for_service('/gazebo/get_model_state')
        model_state = self.get_model_state_proxy(self.get_model_state_req)
        return np.array([model_state.pose.position.x, model_state.pose.position.y, self.agent.ir + self.torso_width])

    def tripod(self, episode):
        #print("6")
        legs = {}
        values = self.agent.get_angles()
        if episode % 2 == 0:
            for i in range(6):
                joints = np.array([values.get('tc_' + str(i + 1)), values.get('ctr_' + str(i + 1)), values.get('fti_' + str(i + 1)) - np.pi/2])
                if i % 2 == 0:
                    # Dictionary containing all info about the limbs actuation
                    data = {'id': i, 'phase': True}
                    # Swing phase
                    parameters = self.walker.swing_parameters(joints, i)
                    legs[i] = {**data, **parameters}
                else:
                    # Dictionary containing all info about the limbs actuation
                    data = {'id': i, 'phase': False}
                    # Stance phase
                    parameters = self.walker.p_linear(joints, i)
                    legs[i] = {**data, **parameters}
        else:
            for i in range(6):
                joints = np.array([values.get('tc_' + str(i + 1)), values.get('ctr_' + str(i + 1)), values.get('fti_' + str(i + 1)) - np.pi/2])
                if i % 2 == 0:
                    # Dictionary containing all info about the limbs actuation
                    data = {'id': i, 'phase': False}
                    # Stance phase
                    parameters = self.walker.p_linear(joints, i)
                    legs[i] = {**data, **parameters}
                else:
                    # Dictionary containing all info about the limbs actuation
                    data = {'id': i, 'phase': True}
                    # Swing phase
                    parameters = self.walker.swing_parameters(joints, i)
                    legs[i] = {**data, **parameters} # Retorna qual a perna e a sua fase, parameters - retorna os pontos de controlo para qualquer o movimento da perna
        #print("7")
        return legs # Passo 4

    def step(self):
        #print("8")
        actuation, height, x, y, z, w, roll, pitch = [], [], [], [], [], [], [], []
        pos = self.get_torso()
        for i in range(2):
            # Move Joints
            #print("13")
            cot, h, orientation, angular_vel = self.move_joints(i)
            #print("15")
            actuation.append(cot)
            height.append(h)
            roll.append(angular_vel[0])
            pitch.append(angular_vel[1])
            x.append(orientation[0])
            y.append(orientation[1])
            z.append(orientation[2])
            w.append(orientation[3])
        #print("9")
        return self.rms(actuation), self.rms(height), self.rms(x), self.rms(y), self.rms(z), self.rms(w), self.rms(roll), self.rms(pitch)

    def rms(self, dt):
        #print("10")
        n = len(dt)
        square = 0
        for i in range(n):
            square += np.power(dt[i], 2)
        return square/n

    def move_joints(self, iteration):
        #print("2")
        # Get the gait parameters
        params = self.tripod(iteration) # Passo 3
        #print("14")
        cot, height = 0.0, 0.0
        previous_pos = self.get_torso()
        previous_time = rospy.get_time()
        total_power = 0
        power_values = 0
        total_power_p = 0
        for k, t in enumerate(self._time):
            pos = np.zeros(18, dtype=float) # array de 18 casas com 0's primeiras 6 casas com theta1, 6-11 com theta2,12-17 com theta 3
            #previous_pos = self.get_torso()
            #previous_time = rospy.get_time()
            for id in range(6): # Vai de 0 a 5
                idx = [id, id + 6, id + 12]
                if params[id]['phase'] is False:
                    # Stance
                    # Get the impedance values
                    angular_pos = self.walker.stance(params[id], t)
                    for j, joint in enumerate(idx): # no primeiro ciclo só vai ter a primeira coordenada
                        if j == 2: # por exemplo com id = 0, joint = [0, 6, 12]
                            pos[joint] = angular_pos[j] + np.pi/2 #Dá os theta1, theta2, theta3
                        else:
                            pos[joint] = angular_pos[j]
                else:
                    #Swing
                    angular_pos = self.walker.swing(params[id], t)
                    # Sendo idx =[0, 6, 12] por exemplo
                    for j, joint in enumerate(idx): # j conta, joint é a variável dentro de idx
                        if j == 2:
                            pos[joint] = angular_pos[j] + np.pi/2
                        else:
                            pos[joint] = angular_pos[j]
            #print("11")
            self.agent._pub_joints.move(pos)
            current_pos = self.get_torso()
            current_time = rospy.get_time()
            #if k == 0 or k == 1 or k == 20 or k == 21:
            #    power_values = 0
            #else:
            power_values = sum(self.agent.power.values())
            total_power += power_values
            #vel = np.linalg.norm(current_pos - previous_pos)/(current_time - previous_time)
            #cot = sum(self.agent.power.values()) / (vel * 9.81 * self.mass)
            height = current_pos[2]

        vel = np.linalg.norm(current_pos - previous_pos)/(current_time - previous_time)
        total_power_t = total_power/10
        cot = total_power_t / (vel * 9.81 * self.mass)
        return cot, height, self.agent.orientation, self.agent.angular_vel

if __name__ == '__main__':
    path = '/home/diogo/Desktop/Traj/Regular'
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    # 0 - 3rd order Bézier curve, 1 - 2nd order polynomial, 2 - Linear function, 3 - 4th order Bézier curve, 4 -Spline
    motion_id = 2
    #print("0")
    hexapod = Environment(motion_id) # Faz a Walk Function Primeira coisa que acontece de todas!!
    #print("1")
    try:
        log_ = []
        logs = pd.DataFrame(log_, columns=['cot', 'height', 'x', 'y', 'z', 'w', 'roll_i', 'pitch_i'])
        #logs = pd.DataFrame(log_, columns=['cot', 'roll_i'])
        hexapod.reset()
        #fig = plt.figure()
        for i in range(20): # A original é 20
            cot, height, x, y, z, w, roll, pitch = hexapod.step() # Faz acontecer o resto dos movimentos de deslocação
            data = [cot, height, x, y, z, w, roll, pitch]
            step_row = pd.DataFrame([data], columns=['cot', 'height', 'x', 'y', 'z', 'w', 'roll_i', 'pitch_i'])
            #data = [cot, roll]
            #step_row = pd.DataFrame([data], columns=['cot', 'roll_i'])
            frames = [logs, step_row]
            logs = pd.concat(frames)
            #if os.environ.get('FROM_SCRIPT'):
        fname = '/case_' +  str(motion_id) + '0.csv'
        pd.DataFrame(logs).to_csv(path + fname)

    except rospy.ROSInterruptException:
        rospy.logerr("ROS Interrupt Exception!")
        pass
    print("50")
    hexapod.pause_()
    print("100")
    rospy.signal_shutdown('Simulation ended!')
    # Kill the roslaunch process
    # Get a list of all active nodes
    node_list = subprocess.check_output(["rosnode", "list"]).decode().strip().split("\n")

    # Kill each node in the list
    for node in node_list:
        subprocess.call(["rosnode", "kill", node])
    print("412")
    subprocess.run(["pkill", "-f", "hexapod.launch"])
    print("12311231312312312312123s")
