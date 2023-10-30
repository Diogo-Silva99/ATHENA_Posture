#! /usr/bin/env python3

import sys
import rospy
import actionlib
import numpy as np
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import Imu, JointState, Range
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryActionGoal


class Joints:
    def __init__(self, ns, joint_name_lst):
        self._client = actionlib.SimpleActionClient(ns + 'joint_trajectory_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        server_up = self._client.wait_for_server()
        if not server_up:
            rospy.logerr('Timed out waiting for Joint Trajectory')
            rospy.signal_shutdown('Timed out waiting for Action Server')
            sys.exit(1)
        self.joint_name_lst = joint_name_lst

    def move(self, pos):
        msg = FollowJointTrajectoryActionGoal()
        msg.goal.trajectory.joint_names = self.joint_name_lst
        point = JointTrajectoryPoint()
        point.positions = pos
        point.time_from_start = rospy.Duration(0.045)
        msg.goal.trajectory.points.append(point)
        self._client.send_goal_and_wait(msg.goal)

    def reset(self):
        msg = FollowJointTrajectoryActionGoal()
        msg.goal.trajectory.joint_names = self.joint_name_lst
        point = JointTrajectoryPoint()
        point.positions = np.zeros(18, dtype=float)
        point.time_from_start = rospy.Duration(0.045)
        msg.goal.trajectory.points.append(point)
        self._client.send_goal_and_wait(msg.goal)

class Athena: # Primeira coisa a ser chamada no environment.py
    def __init__(self, ns='/hexapod/'):  # Dar o nome ao robo
        rospy.init_node('joint_control', anonymous=True)
        self.rate = rospy.Rate(20)

        self.joint_name_lst = None
        self.angles = None

        self._sub_joints = rospy.Subscriber(ns + 'joint_states', JointState, self._cb_joints, queue_size=1)
        rospy.loginfo('Waiting for joints to be populated...')
        while not rospy.is_shutdown():
            if self.joint_name_lst is not None:
                break
            self.rate.sleep()
            rospy.loginfo('Waiting for joints to be populated...')
        rospy.loginfo('Joints populated.')

        rospy.loginfo('Creating joint command publishers.')
        self._pub_joints = Joints(ns, self.joint_name_lst)

        self.orientation, self.angular_vel, self.linear_acc = np.zeros(4, dtype=float), np.zeros(3, dtype=float), np.zeros(3, dtype=float)
        self.imu_subscriber = rospy.Subscriber(ns + 'imu', Imu, self.imu_subscriber_callback, queue_size=1)
        self.ir = 0
        self.ir_subscriber = rospy.Subscriber('/sensor/ir', Range, self.ir_subscriber, queue_size=1)

        self.power = {}

        self.tc_1_subscriber = rospy.Subscriber('/torque_tc_1', WrenchStamped, self.tc_1_subscriber_callback, queue_size=1)
        self.tc_2_subscriber = rospy.Subscriber('/torque_tc_2', WrenchStamped, self.tc_2_subscriber_callback, queue_size=1)
        self.tc_3_subscriber = rospy.Subscriber('/torque_tc_3', WrenchStamped, self.tc_3_subscriber_callback, queue_size=1)
        self.tc_4_subscriber = rospy.Subscriber('/torque_tc_4', WrenchStamped, self.tc_4_subscriber_callback, queue_size=1)
        self.tc_5_subscriber = rospy.Subscriber('/torque_tc_5', WrenchStamped, self.tc_5_subscriber_callback, queue_size=1)
        self.tc_6_subscriber = rospy.Subscriber('/torque_tc_6', WrenchStamped, self.tc_6_subscriber_callback, queue_size=1)

        self.ctr_1_subscriber = rospy.Subscriber('/torque_ctr_1', WrenchStamped, self.ctr_1_subscriber_callback, queue_size=1)
        self.ctr_2_subscriber = rospy.Subscriber('/torque_ctr_2', WrenchStamped, self.ctr_2_subscriber_callback, queue_size=1)
        self.ctr_3_subscriber = rospy.Subscriber('/torque_ctr_3', WrenchStamped, self.ctr_3_subscriber_callback, queue_size=1)
        self.ctr_4_subscriber = rospy.Subscriber('/torque_ctr_4', WrenchStamped, self.ctr_4_subscriber_callback, queue_size=1)
        self.ctr_5_subscriber = rospy.Subscriber('/torque_ctr_5', WrenchStamped, self.ctr_5_subscriber_callback, queue_size=1)
        self.ctr_6_subscriber = rospy.Subscriber('/torque_ctr_6', WrenchStamped, self.ctr_6_subscriber_callback, queue_size=1)

        self.fti_1_subscriber = rospy.Subscriber('/torque_fti_1', WrenchStamped, self.fti_1_subscriber_callback, queue_size=1)
        self.fti_2_subscriber = rospy.Subscriber('/torque_fti_2', WrenchStamped, self.fti_2_subscriber_callback, queue_size=1)
        self.fti_3_subscriber = rospy.Subscriber('/torque_fti_3', WrenchStamped, self.fti_3_subscriber_callback, queue_size=1)
        self.fti_4_subscriber = rospy.Subscriber('/torque_fti_4', WrenchStamped, self.fti_4_subscriber_callback, queue_size=1)
        self.fti_5_subscriber = rospy.Subscriber('/torque_fti_5', WrenchStamped, self.fti_5_subscriber_callback, queue_size=1)
        self.fti_6_subscriber = rospy.Subscriber('/torque_fti_6', WrenchStamped, self.fti_6_subscriber_callback, queue_size=1)

    def _cb_joints(self, msg):
        if self.joint_name_lst is None:
            self.joint_name_lst = msg.name
        self.angles = msg.position
        self.velocities = msg.velocity

    def get_angles(self):
        if self.joint_name_lst is None:
            return None
        if self.angles is None:
            return None
        return dict(zip(self.joint_name_lst, self.angles))

    def ir_subscriber(self, sensor):
        self.ir = sensor.range

    def imu_subscriber_callback(self, imu):
        self.orientation = np.array([imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w])
        self.angular_vel = np.array([imu.angular_velocity.x, imu.angular_velocity.y])
        self.linear_acc = np.array([imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z])

    def tc_1_subscriber_callback(self, sensor):
        idx = self.joint_name_lst.index('tc_1')
        self.power['tc_1'] = abs(sensor.wrench.torque.z * self.velocities[idx])

    def tc_2_subscriber_callback(self, sensor):
        idx = self.joint_name_lst.index('tc_2')
        self.power['tc_2'] = abs(sensor.wrench.torque.z * self.velocities[idx])

    def tc_3_subscriber_callback(self, sensor):
        idx = self.joint_name_lst.index('tc_3')
        self.power['tc_3'] = abs(sensor.wrench.torque.z * self.velocities[idx])

    def tc_4_subscriber_callback(self, sensor):
        idx = self.joint_name_lst.index('tc_4')
        self.power['tc_4'] = abs(sensor.wrench.torque.z * self.velocities[idx])

    def tc_5_subscriber_callback(self, sensor):
        idx = self.joint_name_lst.index('tc_5')
        self.power['tc_5'] = abs(sensor.wrench.torque.z * self.velocities[idx])

    def tc_6_subscriber_callback(self, sensor):
        idx = self.joint_name_lst.index('tc_6')
        self.power['tc_6'] = abs(sensor.wrench.torque.z * self.velocities[idx])

    def ctr_1_subscriber_callback(self, sensor):
        idx = self.joint_name_lst.index('ctr_1')
        self.power['ctr_1'] = abs(sensor.wrench.torque.z * self.velocities[idx])

    def ctr_2_subscriber_callback(self, sensor):
        idx = self.joint_name_lst.index('ctr_2')
        self.power['ctr_2'] = abs(sensor.wrench.torque.z * self.velocities[idx])

    def ctr_3_subscriber_callback(self, sensor):
        idx = self.joint_name_lst.index('ctr_3')
        self.power['ctr_3'] = abs(sensor.wrench.torque.z * self.velocities[idx])

    def ctr_4_subscriber_callback(self, sensor):
        idx = self.joint_name_lst.index('ctr_4')
        self.power['ctr_4'] = abs(sensor.wrench.torque.z * self.velocities[idx])

    def ctr_5_subscriber_callback(self, sensor):
        idx = self.joint_name_lst.index('ctr_5')
        self.power['ctr_5'] = abs(sensor.wrench.torque.z * self.velocities[idx])

    def ctr_6_subscriber_callback(self, sensor):
        idx = self.joint_name_lst.index('ctr_6')
        self.power['ctr_6'] = abs(sensor.wrench.torque.z * self.velocities[idx])

    def fti_1_subscriber_callback(self, sensor):
        idx = self.joint_name_lst.index('fti_1')
        self.power['fti_1'] = abs(sensor.wrench.torque.z * self.velocities[idx])

    def fti_2_subscriber_callback(self, sensor):
        idx = self.joint_name_lst.index('fti_2')
        self.power['fti_2'] = abs(sensor.wrench.torque.z * self.velocities[idx])

    def fti_3_subscriber_callback(self, sensor):
        idx = self.joint_name_lst.index('fti_3')
        self.power['fti_3'] = abs(sensor.wrench.torque.z * self.velocities[idx])

    def fti_4_subscriber_callback(self, sensor):
        idx = self.joint_name_lst.index('fti_4')
        self.power['fti_4'] = abs(sensor.wrench.torque.z * self.velocities[idx])

    def fti_5_subscriber_callback(self, sensor):
        idx = self.joint_name_lst.index('fti_5')
        self.power['fti_5'] = abs(sensor.wrench.torque.z * self.velocities[idx])

    def fti_6_subscriber_callback(self, sensor):
        idx = self.joint_name_lst.index('fti_6')
        self.power['fti_6'] = abs(sensor.wrench.torque.z * self.velocities[idx])
