#! /usr/bin/env python

#__________________________________________________
# pyLorenz/model/
# lorenz95.py
#__________________________________________________
# author        : colonel
# last modified : 2016/11/3
#__________________________________________________
#
# classes to handle Lorenz 1995 model
# i.e. process compute the dervative at a given point according to Lorenz 1995 model
#

import numpy as np

#__________________________________________________

class Lorenz95Model(object):

    #_________________________

    def __init__(self, t_spaceDimension, t_f):
        self.setLorenz95ModelParameters(t_spaceDimension, t_f)

    #_________________________

    def setLorenz95ModelParameters(self, t_spaceDimension, t_f):
        # state space dimension
        self.m_spaceDimension = t_spaceDimension
        # model parameters
        self.m_f              = t_f

    #_________________________

    def __call__(self, t_x, t_t, t_dx):
        # compute dx at point x and time t according to Lorenz 1995 model

        # np.roll version
        t_dx[..., :] = ( np.roll(t_x, -1, axis=-1) - np.roll(t_x, 2, axis=-1) ) * np.roll(t_x, 1, axis=-1) - t_x + self.m_f

        # slice version
        # this version may be faster for high dimensions and/or number of samples
        # t_dx[..., 2:-1] = ( t_x[..., 3:] - t_x[..., 0:-3] ) * t_x[..., 1:-2] - t_x[..., 2:-1] + self.m_f
        # t_dx[..., -1]   = ( t_x[..., 0]  - t_x[..., -3] )   * t_x[..., -2]   - t_x[..., -1]   + self.m_f
        # t_dx[..., 0]    = ( t_x[..., 1]  - t_x[..., -2] )   * t_x[..., -1]   - t_x[..., 0]    + self.m_f
        # t_dx[..., 1]    = ( t_x[..., 2]  - t_x[..., -1] )   * t_x[..., 0]    - t_x[..., 1]    + self.m_f

#__________________________________________________

