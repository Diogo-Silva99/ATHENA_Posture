#! /usr/bin/env python3
from scipy.interpolate import CubicSpline
import numpy as np
from kinematics import Kinematics

class WalkFunction: # Determinar parametros da locomocao
    def __init__(self, id):
        self.height = 0.2
        self.pos = np.zeros((6, 4), dtype=float) # Posicao de cada perna
        self._tf = 1.05
        # 0 - 3rd order Bézier curve, 1 - 2nd order polynomial, 2 - Linear function, 3 - 4th order Bézier curve
        self.id = id

    def swing_parameters(self, joints, leg_id):
        if self.id == 0:
            return self.bezier3(joints, leg_id)
        elif self.id == 1:
            return self.polynomial2(joints, leg_id)
        elif self.id == 3:
            return self.bezier4(joints, leg_id)
        elif self.id == 4:
            return self.spline(joints, leg_id)
        else: #elif self.id == 2:
            return self.p_linear(joints, leg_id)
        # ToDo: CHANGE HERE

    def polynomial2(self, joints, id):
        p0 = Kinematics(id).fk(joints)
        stride = 0.045
        height = 0.06
        t1,t2 = np.eye(4, dtype=float), np.eye(4, dtype=float)

        t1[1, 3] = stride / 2
        t2[1, 3] = stride

        t1[2, 3] = height

        p1 = t1.dot(p0.T)
        p2 = t2.dot(p0.T)

        parameters ={'p0': p0, 'p1': p1, 'p2': p2}
        return parameters
    def spline(self, joints, id):
        p0 = Kinematics(id).fk(joints)
        stride = 0.045
        height = 0.05
        t1, t2, t3, t4 = np.eye(4, dtype=float), np.eye(4, dtype=float), np.eye(4, dtype=float), np.eye(4,dtype=float)

        t1[1, 3] = stride / 4
        t2[1, 3] = stride / 2
        t3[1, 3] = stride * 3 / 4
        t4[1, 3] = stride

        t1[2, 3] = height / 1.35
        t2[2, 3] = height
        t3[2, 3] = height / 1.35
        t4[2, 3] = 0

        p1 = t1.dot(p0.T)
        p2 = t2.dot(p0.T)
        p3 = t3.dot(p0.T)
        p4 = t4.dot(p0.T)
        # Control points
        x = [p0[1], p1[1], p2[1], p3[1], p4[1]]
        y = [p0[2], p1[2], p2[2], p3[2], p4[2]]
        f = CubicSpline(x, y, bc_type='natural')

        parameters = {'p0': p0, 'p1': p1, 'p2': p2, 'p3': p3, 'p4': p4, 'x':x, 'y': y, 'f':f}
        return parameters

    def bezier3(self, joints, id): # alterar aqui com as outras equações
        # Requires the angular positions
        p0 = Kinematics(id).fk(joints) #id é a Perna, vai ao file Kinematics e faz a cinematica direta, retorna array [x,x,x,1]
        # as joints são os arg da def fk
        t1, t2, t3 = np.eye(4, dtype=float), np.eye(4, dtype=float), np.eye(4, dtype=float)
        stride = 0.045

        t1[1, 3] = stride/4
        t2[1, 3] = (3/4) * stride
        t3[1, 3] = stride

        t1[2, 3] = self.height/3
        t2[2, 3] = self.height/3
        # Control points
        p1 = t1.dot(p0.T) # p1, p2 e p3 ficam arrays to tipo [x,x,x,x]
        p2 = t2.dot(p0.T)
        p3 = t3.dot(p0.T)

        parameters = {'p0': p0, 'p1': p1, 'p2': p2, 'p3': p3}
        return parameters

    def bezier4(self, joints, id):
        #Pegar nas posições angulares
        p0 = Kinematics(id).fk(joints)

        t1, t2, t3, t4 = np.eye(4, dtype=float), np.eye(4, dtype=float), np.eye(4, dtype=float), np.eye(4, dtype=float)
        stride = 0.045
        height = 0.09
        t1[1, 3] = stride/4
        t2[1, 3] = stride/2
        t3[1, 3] = (3/4) * stride
        t4[1, 3] = stride

        t1[2, 3] = height / 3
        t2[2, 3] = height
        t3[2, 3] = height / 3
        # Control Points
        p1 = t1.dot(p0.T)
        p2 = t2.dot(p0.T)
        p3 = t3.dot(p0.T)
        p4 = t4.dot(p0.T)

        parameters = {'p0': p0, 'p1': p1, 'p2': p2, 'p3': p3, 'p4': p4}
        return parameters

    def linear_motion(self, params, t):
        id = params.get('id')
        p0 = params.get('p0')

        dt = 1.0/2
        dt_y = 1.0/2
        stride = 0.045/2
        height = 0.05
        pz, py = p0[2], p0[1]

        if t <= dt:
            # Ascending motion
            self.pos[id] = np.array([p0[0], p0[1] + (stride/dt_y) * t, ((height - p0[2])/dt) * t + p0[2], 1])
            pz = self.pos[id][2]
            py = self.pos[id][1]
            return Kinematics(id).ik(self.pos[id])
        else:
            # Descending motion
            m_y = (py - (2*stride)) / (dt - (2 * dt))
            m = (pz - p0[2]) / (dt - (2 * dt))
            self.pos[id] = np.array([p0[0], py + (stride/dt_y) * t, m * t + p0[2] - (m * 2 * dt), 1]) #(pz / (dt - 2 * dt)) * t - (2 * dt) * (pz / (dt - 2 * dt))
            return Kinematics(id).ik(self.pos[id])

    def p_linear(self, joints, id): # Quando a perna está parada
        # Requires the angular positions
        p0 = Kinematics(id).fk(joints)
        parameters = {'p0': p0}
        return parameters

    def spline_motion(self, params, t):
        id = params.get('id')
        p0 = np.array(params.get('p0'))
        p1 = np.array(params.get('p1'))
        p2 = np.array(params.get('p2'))
        p3 = np.array(params.get('p3'))
        p4 = np.array(params.get('p4'))
        x = np.array(params.get('x'))
        y = np.array(params.get('y'))
        f = params.get('f')
        stride = 0.045
        dt = 1
        coord_z = f(p0[1]+(t*(stride/dt)))

        self.pos[id] = np.array([p0[0],p0[1]+(stride/dt)*t, coord_z , 1])

        return Kinematics(id).ik(self.pos[id])

    def polynomial2_motion(self, params, t): # xy é feito pelas coordenadas
        id = params.get('id')
        p0 = np.array(params.get('p0')) #[x, y, z, 1] y= ax² + bx + c
        p1 = np.array(params.get('p1')) #[x, y, z, 1]
        p2 = np.array(params.get('p2')) #[x, y, z, 1]
        stride = 0.045
        dt = 1.05
        matrix = np.array([[1, 0, 0],
                           [1, 1/2, 1/4],
                          [1, 1, 1]])
        inv = np.linalg.inv(matrix)
        coord_z = np.array([[p0[2]],
                            [p1[2]],
                            [p2[2]]])
        coord_y = np.array([[p0[1]],
                            [p1[1]],
                            [p2[1]]])
        const_z = np.dot(inv, coord_z)
        const_y = np.dot(inv, coord_y)
        # y = a + bx + cx² const[0][0] + const[1][0] * ((t * p0[2])/dt) + const[2][0] * np.power((t * p0[2])/dt, 2)
        self.pos[id] = np.array([p0[0], const_y[0][0] + const_y[1][0] * t + const_y[2][0] * np.power(t, 2),
         const_z[0][0] + const_z[1][0] * t  + const_z[2][0] * np.power(t, 2) , 1])
        return Kinematics(id).ik(self.pos[id])

    def bezier3_motion(self, params, t):
        id = params.get('id')
        p0 = np.array(params.get('p0'))
        p1 = np.array(params.get('p1'))
        p2 = np.array(params.get('p2'))
        p3 = np.array(params.get('p3'))

        self.pos[id] = np.power(1 - t, 3) * p0 + 3 * t * np.power((1 - t), 2) * p1 + 3 * np.power(t, 2) * (1 - t) * p2 + np.power(t, 3) * p3
        return Kinematics(id).ik(self.pos[id])

    def bezier4_motion(self, params, t):
        id = params.get('id')
        p0 = np.array(params.get('p0'))
        p1 = np.array(params.get('p1'))
        p2 = np.array(params.get('p2'))
        p3 = np.array(params.get('p3'))
        p4 = np.array(params.get('p4'))
        # Equação 4ºordem Bezier
        self.pos[id] = np.power((1- t), 4) * p0 + 4 * t * np.power((1 - t), 3) * p1 + 6 * np.power(t, 2) * np.power((1 - t), 2) * p2 + 4 * np.power(t, 3) * (1- t) * p3 + np.power(t, 4) * p4
        return Kinematics(id).ik(self.pos[id])
        # pos[id] = [0=x, 1=y, 2=z, 3=1]

    def swing(self, params, t):
        if self.id == 0:
            return self.bezier3_motion(params, t)
        elif self.id == 1:
            return self.polynomial2_motion(params, t)
        elif self.id == 3:
            return self.bezier4_motion(params, t)
        elif self.id == 4:
            return self.spline_motion(params, t)
        else: # self.id == 2
            return self.linear_motion(params, t)
        # ToDo: CHANGE HERE

    def stance(self, params, t):
        id = params.get('id')
        p0 = params.get('p0')
        stride = -0.045
        pos = np.array([p0[0], p0[1] + (stride/self._tf) * t, p0[2], 1])
        return Kinematics(id).ik(pos) #ik recebe um array [x,x,x,x]
        # Retorna os valores de theta1, theta2 e theta 3
