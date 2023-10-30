#! /usr/bin/env python3

import sys
import rospy
import actionlib
import numpy as np
from std_msgs.msg import Float64
from geometry_msgs.msg import WrenchStamped
from gazebo_msgs.msg import ContactsState, ModelStates
from tf.transformations import euler_from_quaternion
from sensor_msgs.msg import JointState, Imu, Range

class Athena:
    def __init__(self, ns='/hexapod/'):
        rospy.init_node('joint_control', anonymous=True)
        self.rate = rospy.Rate(20)


        self.joint_name_lst = None
        self.angles = None

        self._sub_joints = rospy.Subscriber(ns + 'joint_states', JointState, self._cb_joints, queue_size=1)
        rospy.loginfo('Waiting for joints to be populated...')
        rospy.Subscriber('gazebo/model_states', ModelStates, self.pose_callback, queue_size=1)
        while not rospy.is_shutdown():
            if self.joint_name_lst is not None:
                break
            self.rate.sleep()
            rospy.loginfo('Waiting for joints to be populated...')
        rospy.loginfo('Joints populated.')
        rospy.sleep(1)
        rospy.loginfo('Creating joint command publishers.')
        self._pub_joints = {}
        for j in self.joint_name_lst:
            p = rospy.Publisher(ns + j + '_controller/command', Float64, queue_size=1)
            self._pub_joints[j] = p

        rospy.sleep(1)
        self.depth = 0.0
        self.ir = rospy.Subscriber('sensor/ir', Range, self.ir_subscriber_callback, queue_size=1)
        self.orientation, self.ang_v = np.zeros(4, dtype=float), np.zeros(3, dtype=float)
        self.imu_subscriber = rospy.Subscriber(ns + 'imu', Imu, self.imu_subscriber_callback, queue_size=1)
        self.power = {}
        self.tc_joint_torque = 0
        self.ctr_joint_torque = 0
        self.fti_joint_torque = 0
        self.tc2_joint_torque = 0
        self.ctr2_joint_torque = 0
        self.fti2_joint_torque = 0

        # Contact forces
        self.force = {}
        self.foot_1 = rospy.Subscriber('/foot_1_bumper', ContactsState, self.foot_1_callback, queue_size=1)
        self.foot_2 = rospy.Subscriber('/foot_2_bumper', ContactsState, self.foot_2_callback, queue_size=1)
        self.foot_3 = rospy.Subscriber('/foot_3_bumper', ContactsState, self.foot_3_callback, queue_size=1)
        self.foot_4 = rospy.Subscriber('/foot_4_bumper', ContactsState, self.foot_4_callback, queue_size=1)
        self.foot_5 = rospy.Subscriber('/foot_5_bumper', ContactsState, self.foot_5_callback, queue_size=1)
        self.foot_6 = rospy.Subscriber('/foot_6_bumper', ContactsState, self.foot_6_callback, queue_size=1)

        # Torque Values in Joints

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

        rospy.sleep(1)
        rospy.loginfo('ROS-Client set. Ready to walk.')

        self.pitch_angle = 0
        self.roll_angle = 0

    def pose_callback(self, data):
        robot_name = "hexapod"
        robot_index = data.name.index(robot_name)
        robot_orientation = data.pose[robot_index].orientation
        roll, pitch, yaw = euler_from_quaternion([robot_orientation.x,
                                                  robot_orientation.y,
                                                  robot_orientation.z,
                                                  robot_orientation.w])
        self.robot_y_position = data.pose[robot_index].position.y
        self.pitch_angle = pitch
        self.roll_angle = roll

    def get_pitch_angle(self):
        return self.roll_angle

    def get_y_position(self):
        return self.robot_y_position

    def _cb_joints(self, msg):
        if self.joint_name_lst is None:
            self.joint_name_lst = msg.name
            print(msg)
        self.angles = msg.position
        self.velocities = msg.velocity

    def get_angles(self):
        if self.joint_name_lst is None:
            return None
        if self.angles is None:
            return None
        return dict(zip(self.joint_name_lst, self.angles))

    def set_angles(self, angles):
        rospy.loginfo('Publishing joints...')
        for j,v in angles.items():
            if j not in self.joint_name_lst:
                rospy.logerror('Invalid joint name "'+ j +'"')
                continue
            self._pub_joints[j].publish(v)

    def ir_subscriber_callback(self, sensor):
        self.depth = sensor.range

    def imu_subscriber_callback(self, imu):
        self.orientation = np.array([imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w])
        self.ang_v = np.array([imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z])

    def foot_1_callback(self, bumper):
        if len(bumper.states) >= 1:
            self.force[0] = {'id': 1, 'contact': True, 'normal_force': bumper.states[0].total_wrench.force.z}
        else:
            self.force[0] = {'id': 1, 'contact': False, 'normal_force': 0.0}

    def foot_2_callback(self, bumper):
        if len(bumper.states) >= 1:
            self.force[1] = {'id': 2, 'contact': True, 'normal_force': bumper.states[0].total_wrench.force.z}
        else:
            self.force[1] = {'id': 2, 'contact': False, 'normal_force': 0.0}

    def foot_3_callback(self, bumper):
        if len(bumper.states) >= 1:
            self.force[2] = {'id': 3, 'contact': True, 'normal_force': bumper.states[0].total_wrench.force.z}

        else:
            self.force[2] = {'id': 3, 'contact': False, 'normal_force': 0.0}

    def foot_4_callback(self, bumper):
        if len(bumper.states) >= 1:
            self.force[3] = {'id': 4, 'contact': True, 'normal_force': bumper.states[0].total_wrench.force.z}
        else:
            self.force[3] = {'id': 4, 'contact': False, 'normal_force': 0.0}

    def foot_5_callback(self, bumper):
        if len(bumper.states) >= 1:
            self.force[4] = {'id': 5, 'contact': True, 'normal_force': bumper.states[0].total_wrench.force.z}
        else:
            self.force[4] = {'id': 5, 'contact': False, 'normal_force': 0.0}

    def foot_6_callback(self, bumper):
        if len(bumper.states) >= 1:
            self.force[5] = {'id': 6, 'contact': True, 'normal_force': bumper.states[0].total_wrench.force.z}
        else:
            self.force[5] = {'id': 6, 'contact': False, 'normal_force': 0.0}

    ###############################
    ###############################


    def tc_1_subscriber_callback(self, sensor):
        idx = self.joint_name_lst.index('tc_1')
        self.power['tc_1'] = abs(sensor.wrench.torque.z * self.velocities[idx])
        self.tc_joint_torque = abs(sensor.wrench.torque.z)
        #self.power['tc_1'] = (sensor.wrench.torque.y)

    def tc_2_subscriber_callback(self, sensor):
        idx = self.joint_name_lst.index('tc_2')
        self.power['tc_2'] = abs(sensor.wrench.torque.z * self.velocities[idx])
        #self.power['tc_2'] = (sensor.wrench.torque.y)

    def tc_3_subscriber_callback(self, sensor):
        idx = self.joint_name_lst.index('tc_3')
        self.power['tc_3'] = abs(sensor.wrench.torque.z * self.velocities[idx])

    def tc_4_subscriber_callback(self, sensor):
        idx = self.joint_name_lst.index('tc_4')
        self.power['tc_4'] = abs(sensor.wrench.torque.z * self.velocities[idx])
        self.tc2_joint_torque = abs(sensor.wrench.torque.z)

    def tc_5_subscriber_callback(self, sensor):
        idx = self.joint_name_lst.index('tc_5')
        self.power['tc_5'] = abs(sensor.wrench.torque.z * self.velocities[idx])

    def tc_6_subscriber_callback(self, sensor):
        idx = self.joint_name_lst.index('tc_6')
        self.power['tc_6'] = abs(sensor.wrench.torque.z * self.velocities[idx])

    def ctr_1_subscriber_callback(self, sensor):
        idx = self.joint_name_lst.index('ctr_1')
        self.power['ctr_1'] = abs(sensor.wrench.torque.y * self.velocities[idx])
        self.ctr_joint_torque = abs(sensor.wrench.torque.y)
        #self.power['ctr_1'] = abs(sensor.wrench.torque.y * self.velocities[idx])

    def ctr_2_subscriber_callback(self, sensor):
        idx = self.joint_name_lst.index('ctr_2')
        self.power['ctr_2'] = abs(sensor.wrench.torque.y * self.velocities[idx])
        #self.power['ctr_2'] = abs(sensor.wrench.torque.y * self.velocities[idx])

    def ctr_3_subscriber_callback(self, sensor):
        idx = self.joint_name_lst.index('ctr_3')
        self.power['ctr_3'] = abs(sensor.wrench.torque.y * self.velocities[idx])

    def ctr_4_subscriber_callback(self, sensor):
        idx = self.joint_name_lst.index('ctr_4')
        self.power['ctr_4'] = abs(sensor.wrench.torque.y * self.velocities[idx])
        self.ctr2_joint_torque = abs(sensor.wrench.torque.y)

    def ctr_5_subscriber_callback(self, sensor):
        idx = self.joint_name_lst.index('ctr_5')
        self.power['ctr_5'] = abs(sensor.wrench.torque.y * self.velocities[idx])

    def ctr_6_subscriber_callback(self, sensor):
        idx = self.joint_name_lst.index('ctr_6')
        self.power['ctr_6'] = abs(sensor.wrench.torque.y * self.velocities[idx])

    def fti_1_subscriber_callback(self, sensor):
        idx = self.joint_name_lst.index('fti_1')
        self.power['fti_1'] = abs(sensor.wrench.torque.y * self.velocities[idx])
        self.fti_joint_torque = abs(sensor.wrench.torque.y)

    def fti_2_subscriber_callback(self, sensor):
        idx = self.joint_name_lst.index('fti_2')
        self.power['fti_2'] = abs(sensor.wrench.torque.y * self.velocities[idx])

    def fti_3_subscriber_callback(self, sensor):
        idx = self.joint_name_lst.index('fti_3')
        self.power['fti_3'] = abs(sensor.wrench.torque.y * self.velocities[idx])

    def fti_4_subscriber_callback(self, sensor):
        idx = self.joint_name_lst.index('fti_4')
        self.power['fti_4'] = abs(sensor.wrench.torque.y * self.velocities[idx])
        self.fti2_joint_torque = abs(sensor.wrench.torque.y)

    def fti_5_subscriber_callback(self, sensor):
        idx = self.joint_name_lst.index('fti_5')
        self.power['fti_5'] = abs(sensor.wrench.torque.y * self.velocities[idx])

    def fti_6_subscriber_callback(self, sensor):
        idx = self.joint_name_lst.index('fti_6')
        self.power['fti_6'] = abs(sensor.wrench.torque.y * self.velocities[idx])
