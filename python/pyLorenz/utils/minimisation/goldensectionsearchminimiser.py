#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/minimisation/
# goldensectionsearchminimiser.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/17
#__________________________________________________
#
# class to perform 1D minimisation according to the golden section search algorithm
#

import numpy as np

#__________________________________________________

class GoldenSectionMinimiser(object):

    #_________________________

    def __init__(self, t_maxIt, t_tolerance):
        self.setGoldenSectionMinimiserParameters(t_maxIt, t_tolerance)

    #_________________________

    def setGoldenSectionMinimiserParameters(self, t_maxIt, t_tolerance):
        # maximum number of iterations
        self.m_maxIt     = t_maxIt
        # tolerance criterium
        self.m_tolerance = t_tolerance
        # golden ratio
        self.m_resphi = 2.0 - 0.5 * ( 1.0 + np.sqrt( 5.0 ) )

    #_________________________

    def recMinimiseInterval(self, t_f, t_x1, t_x2, t_x3, t_fx2, t_nIt):
        # minimisation step according to the golden section search

        if t_nIt > self.m_maxIt:
            raise Exception
            print('too much iterations')
            return (0.5*(t_x3+t_x1), t_nIt)

        # largest interval on the right
        if t_x3 - t_x2 > t_x2 - t_x1:

            # probe point
            x4 = t_x2 + self.m_resphi * ( t_x3 - t_x2 )

            # convergence criterion
            if abs(t_x3 - t_x1) < self.m_tolerance * ( abs(t_x2) + abs(x4) ):
                return (0.5*(t_x3+t_x1), t_nIt)

            # evaluation at the new point
            fx4 = t_f(x4)

            if fx4 < t_fx2:
                return self.recMinimiseInterval(t_f, t_x2, x4, t_x3, fx4, t_nIt+1)
            else:
                return self.recMinimiseInterval(t_f, t_x1, t_x2, x4, t_fx2, t_nIt+1)

        # largest interval on the left
        else:

            # probe point
            x4 = t_x2 - self.m_resphi * ( t_x2 - t_x1 )

            # convergence criterion
            if abs(t_x3 - t_x1) < self.m_tolerance * ( abs(t_x2) + abs(x4) ):
                return (0.5*(t_x3+t_x1), t_nIt)

            # evaluation at the new point
            fx4 = t_f(x4)

            if fx4 < t_fx2:
                return self.recMinimiseInterval(t_f, t_x1, x4, t_x2, fx4, t_nIt+1)
            else:
                return self.recMinimiseInterval(t_f, x4, t_x2, t_x3, t_fx2, t_nIt+1)

    #_________________________

    def minimiseInterval(self, t_f, t_x1, t_x3, t_x):
        # minimise the scalar function f over interval [x1, x3], with the first guess x
        x2  = t_x1 + ( t_x3 - t_x1 ) / ( 3.0 - self.m_resphi )
        fx2 = t_f(x2)
        return self.recMinimiseInterval(t_f, t_x1, x2, t_x3, fx2, 1)

#__________________________________________________

