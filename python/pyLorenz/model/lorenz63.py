#! /usr/bin/env python

#__________________________________________________
# pyLorenz/model/
# lorenz63.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/13
#__________________________________________________
#
# classes to handle the a Lorenz 1963 model
# i.e. process compute the dervative at a given point according to a Lorenz 1963 model
#

#__________________________________________________

class Lorenz63Model(object):

    #_________________________

    def __init__(self, t_sigma = 10.0, t_beta = 8.0 / 3.0, t_rho = 28.0):
        self.setLorenz63ModelParameters(t_sigma, t_beta, t_rho)

    #_________________________

    def setLorenz63ModelParameters(self, t_sigma, t_beta, t_rho):
        # state space dimension
        self.m_spaceDimension = 3
        # model parameters
        self.m_sigma          = t_sigma
        self.m_beta           = t_beta
        self.m_rho            = t_rho

    #_________________________

    def __call__(self, t_x, t_t, t_dx):
        # compute dx at point x and time t according to Lorenz 1963 model
        t_dx[..., 0] = self.m_sigma * ( t_x[..., 1] - t_x[..., 0] )
        t_dx[..., 1] = ( self.m_rho - t_x[..., 2] ) * t_x[..., 0] - t_x[..., 1]
        t_dx[..., 2] = t_x[..., 0] * t_x[..., 1] - self.m_beta * t_x[..., 2]

#__________________________________________________

