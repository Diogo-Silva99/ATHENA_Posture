#! /usr/bin/env python3

import numpy as np

class Kinematics:
    def __init__(self, id):
        self.l1 = 0.063875
        self.l2 = 0.080
        self.l3 = 0.120
        # 0.114308 + 0.04
        self._id = id

        self.torso_width = 0.2
        self.torso_depth = 0.12

        if id == 0 or id == 3:
            self.a = self.torso_depth / 2
        else:
            self.a = np.sqrt(np.power(self.torso_width / 2, 2) + np.power(self.torso_depth / 2, 2)) # Teorema de Pitagoras porque??


        ro = np.arctan((self.torso_width / 2) / (self.torso_depth / 2)) # ãngulo entre o centro de massa e uma perna no canto do corpo
        self.tau = [0, ro, np.pi - ro, np.pi, np.pi + ro, -ro] # 0 e PI são os ângulos das pernas do meio!!!

    def fk(self, arg):   # FORWARD KINEMATICSS DA PERNAAAA - os args sao a pos das juntas
        return np.array([np.cos(self.tau[self._id] + arg[0]) * (self.l3 * np.cos(arg[1] + arg[2]) + self.l2 * np.cos(arg[1]) + self.l1) + self.a * np.cos(self.tau[self._id]), np.sin(self.tau[self._id] + arg[0]) * (self.l3 * np.cos(arg[1] + arg[2]) + self.l2 * np.cos(arg[1]) + self.l1) + self.a * np.sin(self.tau[self._id]), self.l3 * np.sin(arg[1] + arg[2]) + self.l2 * np.sin(arg[1]), 1])

    '''def fk(self, arg):
        return np.array([np.cos(arg[0]) * (self.l3 * np.cos(arg[1] + arg[2]) + self.l2 * np.cos(arg[1]) + self.l1), np.sin(arg[0]) * (self.l3 * np.cos(arg[1] + arg[2]) + self.l2 * np.cos(arg[1]) + self.l1), self.l3 * np.sin(arg[1] + arg[2]) + self.l2 * np.sin(arg[1]), 1])'''

    def m_04(self, arg): # Isto Não é necessário (Para testes talvez)
        # Transformation matrix between the COM and the foot
        m = np.eye(4, dtype=float)
        m[0, 0] = np.cos(self.tau[self._id] + arg[0]) * np.cos(arg[1] + arg[2])
        m[0, 1] = - np.cos(self.tau[self._id] + arg[0]) * np.sin(arg[1] + arg[2])
        m[0, 2] = np.sin(self.tau[self._id] + arg[0])
        m[0, 3] = np.cos(self.tau[self._id] + arg[0]) * (self.l3 * np.cos(arg[1] + arg[2]) + self.l2 * np.cos(arg[1]) + self.l1) + self.a * np.cos(self.tau[self._id])
        ''' Coordenada m[0, 3] é deslocação em x '''
        ''' Coordenada m[1, 3] é deslocação em y '''
        ''' Coordenada m[2, 3] é deslocação em z '''
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
        return self.transformation_matrix([0.0, self.tau[self._id], self.a, 0.0])  # a e tau mudam de acordo com a perna! #Retorna uma matrix de Transformacao entre o centro de massa e a junta TC

    def ik(self, foot): # foot é um array [x,x,x,x] e as coordenadas do pé a ser analizado
        m_torso_tc = self.tc()
        foot_tc = np.linalg.inv(m_torso_tc).dot(foot.T) # Resulta num array [x,x,x,x] Usando a Matriz de transformação descobre-se as coordenadas da junta TC

        # Using the TC joint as reference
        theta_1 = np.arctan2(foot_tc[1], foot_tc[0]) # Com as coordenadas de x e y obtem o angulo theta1 aka angulo da junta TC

        # Get the CTr position  USANDO o THETA 1 Obtido anteriormente e usando a cinematica direta com o TC como referência conseguimos obter as coordenadas [x,y,z=0?] da junta CTr
        ctr = np.array([self.l1 * np.cos(theta_1 + self.tau[self._id]) + self.a * np.cos(self.tau[self._id]), self.l1 * np.sin(theta_1 + self.tau[self._id]) + self.a * np.sin(self.tau[self._id]), 0])
        # ctr = np.array([self.l1 * np.cos(theta_1), self.l1 * np.sin(theta_1), 0])
        foot_vec = np.delete(foot, [3]) # Ou seja temos as coordenas [x,y,z] do pé

        # Auxiliary vector #Vetor auxiliar que vai dar a "distancia" entre a junta CTR e o ponto de contacto com o chão AKA o pé
        r = np.linalg.norm(foot_vec - ctr) #Descobrir o comprimento do vetor +-   np.clip limita os valores(neste caso entre -1 e 1) se o valor se encontrar abaixo de -1 torna-se -1 e se tiver acima de +1 torna-se +1.
        beta = np.arccos(np.clip((- np.power(r, 2) + np.power(self.l2, 2) + np.power(self.l3, 2)) / (2 * self.l2 * self.l3), -1.0, 1.0)) # LEI DOS COSSENOS!!!
        # Sei que beta + theta_3 é igual a 180=pi, e assim consigo chegar ao theta_3
        theta_3 = - (np.pi - beta)

        beta = np.arccos(np.clip((- np.power(self.l3, 2) + np.power(self.l2, 2) + np.power(r, 2)) / (2 * r * self.l2), -1.0, 1.0))
        gamma = np.arcsin(np.clip((ctr[2] - foot_vec[2]) / r, -1.0, 1.0)) #Ctr com z1=0 e foot_vec com z2=?, sen(gamma)=Lado oposto(z2-z1) / Hipotenusa BAsicamente o np.clip impede erros
        theta_2 = beta - gamma
        return np.array([theta_1, theta_2, theta_3]) #E assim finalmente temos os 3 ângulos de cada junta!

    ''' m_torso_tc = self.tc()
        #print(m_torso_tc)
        #print(foot.T)

        foot_tc = np.linalg.inv(m_torso_tc).dot(foot.T)

        # Using the TC joint as reference
        theta_1 = np.arctan2(foot_tc[1], foot_tc[0])

        # Get the CTr position
        ctr = np.array([self.l1 * np.cos(theta_1 + self.tau[self._id]) + self.a * np.cos(self.tau[self._id]), self.l1 * np.sin(theta_1 + self.tau[self._id]) + self.a * np.sin(self.tau[self._id]), 0])

        foot_vec = np.delete(foot, [3])

        # Auxiliary vector
        r = np.linalg.norm(foot_vec - ctr)
        beta = np.arccos(np.clip((- np.power(r, 2) + np.power(self.l2, 2) + np.power(self.l3, 2)) / (2 * self.l2 * self.l3), -1.0, 1.0))

        theta_3 = - (np.pi - beta)

        beta = np.arccos(np.clip((- np.power(self.l3, 2) + np.power(self.l2, 2) + np.power(r, 2)) / (2 * r * self.l2), -1.0, 1.0))
        gamma = np.arcsin(np.clip((ctr[2] - foot_vec[2]) / r, -1.0, 1.0))
        theta_2 = beta - gamma
        return np.array([theta_1, theta_2, theta_3])'''
#teste = Kinematics(3)
#print(teste._id)
#print(teste.tau)
'''def ik(self, foot):
        # m_torso_tc = self.tc()
        # foot_tc = np.linalg.inv(m_torso_tc).dot(foot.T)

        # Using the TC joint as reference
        theta_1 = np.arctan2(foot[1], foot[0])

        # Get the CTr position
        ctr = np.array([self.l1 * np.cos(theta_1), self.l1 * np.sin(theta_1), 0])

        foot_vec = np.delete(foot, [3])

        # Auxiliary vector
        r = np.linalg.norm(foot_vec - ctr)
        beta = np.arccos(np.clip((- np.power(r, 2) + np.power(self.l2, 2) + np.power(self.l3, 2)) / (2 * self.l2 * self.l3), -1.0, 1.0))

        theta_3 = - (np.pi - beta)

        beta = np.arccos(np.clip((- np.power(self.l3, 2) + np.power(self.l2, 2) + np.power(r, 2)) / (2 * r * self.l2), -1.0, 1.0))
        gamma = np.arcsin(np.clip((ctr[2] - foot_vec[2]) / r, -1.0, 1.0))
        theta_2 = beta - gamma
        return np.array([theta_1, theta_2, theta_3])'''
