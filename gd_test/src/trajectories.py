#! /usr/bin/env python3

import rospy
import numpy as np
from kinematics import Kinematics
from scipy.interpolate import CubicSpline


class WalkFunction:
    def __init__(self):
        print(60)
        self.height = 0.3
        self.pos = np.zeros((6, 4), dtype=float)
        self._tf = 1.05

    def p_bezier(self, joints, id, transformation, height):
        # rotation = {'transformation': r, 'torso_position': p_w, 'torso_orientation': r_w}
        p_w = np.array(transformation.get('torso_position'))
        r = np.array(transformation.get('transformation'))
        r_w = np.array(transformation.get('torso_orientation'))

        # Requires the angular positions
        p0_vec = Kinematics(id).fk(joints) #descobre a posicao do pé de acordo com os valores atuais das juntas
        p0 = np.delete(p0_vec, [3])
        # In global coordinates
        p0_w = p_w + r_w.dot(p0) #posicão do pé de acordo com
        p0_w_vec = np.concatenate((p0_w, [1]), axis=0)

        stride = 0.045
        # Real translation of the foot
        tr = np.eye(4)
        tr[1, 3] = stride
        for i in range(3):
            for j in range(3):
                tr[i, j] = r[i, j]

        # Foot's final position
        pf_w = tr.dot(p0_w_vec)
        delta_t = pf_w - p0_w_vec
        #print('p0: ', p0, ', p0_w: ', p0_w, ', p_w: ', p_w, ', r_w: ', r_w, ', r: ', r)

        t1, t2, t3 = np.eye(4, dtype=float), np.eye(4, dtype=float), np.eye(4, dtype=float)

        t1[0, 3] = delta_t[0]/4
        t1[1, 3] = delta_t[1]/4
        t1[2, 3] = (self.height + delta_t[2])/3

        t2[0, 3] = (3/4) * delta_t[0]
        t2[1, 3] = (3/4) * delta_t[1]
        t2[2, 3] = (self.height + delta_t[2])/3

        t3[0, 3] = delta_t[0]
        t3[1, 3] = delta_t[1]
        t3[2, 3] = delta_t[2] + height

        # Control points
        p1 = t1.dot(p0_w_vec)
        p2 = t2.dot(p0_w_vec)
        p3 = t3.dot(p0_w_vec)

        parameters = {'p0': p0_w_vec, 'p1': p1, 'p2': p2, 'p3': p3, 'torso': p_w, 'torso_orientation': r_w}
        return parameters

    def p_linear(self, joints, id, transformation, height):
        # rotation = {'transformation': r, 'torso_position': p_w, 'torso_orientation': r_w}
        p_w = np.array(transformation.get('torso_position'))
        r = np.array(transformation.get('transformation'))
        r_w = np.array(transformation.get('torso_orientation'))

        # Requires the angular positions
        p0_vec = Kinematics(id).fk(joints)
        p0 = np.delete(p0_vec, [3])
        # In global coordinates
        p0_w = p_w + r_w.dot(p0)
        p0_w_vec = np.concatenate((p0_w, [1]), axis=0)

        stride = -0.045
        # Real translation of the foot
        tr = np.eye(4)
        tr[1, 3] = stride
        for i in range(3):
            for j in range(3):
                tr[i, j] = r[i, j]

        # Foot's final position
        pf_w = tr.dot(p0_w_vec)
        delta_t = pf_w - p0_w_vec

        parameters = {'p0': p0_w_vec, 'height': height, 'torso': p_w, 'transformation': delta_t, 'torso_orientation': r_w}
        return parameters

    def swing(self, params, t):
        id = params.get('id')
        p0 = np.array(params.get('p0'))
        p1 = np.array(params.get('p1'))
        p2 = np.array(params.get('p2'))
        p3 = np.array(params.get('p3'))
        o_w = np.array(params.get('torso'))
        r_w = np.array(params.get('torso_orientation'))

        self.pos[id] = np.power(1 - t, 3) * p0 + 3 * t * np.power((1 - t), 2) * p1 + 3 * np.power(t, 2) * (1 - t) * p2 + np.power(t, 3) * p3

        return Kinematics(id).ik(self.pos[id])
        #return Kinematics(id).ik(self.pos[id], o_w, r_w)

    def stance(self, params, t):
        id = params.get('id')
        p0 = params.get('p0')
        height = params.get('height')
        tr = params.get('transformation')
        o_w = np.array(params.get('torso'))
        r_w = np.array(params.get('torso_orientation'))

        #pos = np.array([p0[0] + (tr[0]/self._tf) * t, p0[1] + (tr[1]/self._tf) * t, p0[2] + height + (tr[2]/self._tf)])

        pos = np.array([p0[0] + (tr[0]/self._tf) * t, p0[1] + (tr[1]/self._tf) * t, p0[2] + height + (tr[2]/self._tf), 1])
        return Kinematics(id).ik(pos)
        #return Kinematics(id).ik(pos, o_w, r_w)
