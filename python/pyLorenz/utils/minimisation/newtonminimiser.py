#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/minimisation/
# newtonminimiser.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/28
#__________________________________________________
#
# class to perform minimisation according to Newton's method
#

import numpy as np

#__________________________________________________

class NewtonMinimiser(object):

    #_________________________

    def __init__(self, t_dx = 1.0e-5, t_maxIt = 100, t_tolerance = 1.0e-8):
        self.setNewtonMinimiserParameters(t_dx, t_maxIt, t_tolerance)

    #_________________________

    def setNewtonMinimiserParameters(self, t_dx = 0.001, t_maxIt = 100, t_tolerance = 1.0e-8):
        # differential step
        self.m_dx        = t_dx
        # maximum number of iterations
        self.m_maxIt     = t_maxIt
        # tolerance criterium
        self.m_tolerance = t_tolerance

    #_________________________

    def computeGradient(self, t_f, t_x):
        # compute gradient of t_f at point t_x

        df = np.zeros(t_x.size)

        xp = np.copy(t_x)
        xm = np.copy(t_x)

        xp[-1] += self.m_dx
        xm[-1] -= self.m_dx

        for i in np.arange(t_x.size):

            xp[i-1] -= self.m_dx
            xm[i-1] += self.m_dx
            xp[i]   += self.m_dx
            xm[i]   -= self.m_dx

            df[i] = t_f(xp) - t_f(xm)

        return df / ( 2.0 * self.m_dx )

    #_________________________

    def computeHessian(self, t_f, t_x):
        # compute hessian of t_f at point t_x

        ddf = np.zeros((t_x.size, t_x.size))

        xpp = np.copy(t_x)
        xpm = np.copy(t_x)
        xmp = np.copy(t_x)
        xmm = np.copy(t_x)

        xpp[-1] += self.m_dx
        xpm[-1] += self.m_dx
        xmp[-1] -= self.m_dx
        xmm[-1] -= self.m_dx

        for i in np.arange(t_x.size):

            xpp[i-1] -= self.m_dx
            xpm[i-1] -= self.m_dx
            xmp[i-1] += self.m_dx
            xmm[i-1] += self.m_dx

            xpp[i]   += self.m_dx
            xpm[i]   += self.m_dx
            xmp[i]   -= self.m_dx
            xmm[i]   -= self.m_dx

            xpp[-1] += self.m_dx
            xpm[-1] -= self.m_dx
            xmp[-1] += self.m_dx
            xmm[-1] -= self.m_dx

            for j in np.arange(i+1):

                xpp[j-1] -= self.m_dx
                xpm[j-1] += self.m_dx
                xmp[j-1] -= self.m_dx
                xmm[j-1] += self.m_dx

                xpp[j]   += self.m_dx
                xpm[j]   -= self.m_dx
                xmp[j]   += self.m_dx
                xmm[j]   -= self.m_dx

                ddf[i, j] = t_f(xpp) - t_f(xpm) - t_f(xmp) + t_f(xmm)
                ddf[j, i] = ddf[i, j]

            xpp[i]   -= self.m_dx
            xpm[i]   += self.m_dx
            xmp[i]   -= self.m_dx
            xmm[i]   += self.m_dx

        return ddf / ( 4.0 * self.m_dx * self.m_dx )

    #_________________________

    def recMinimise(self, t_f, t_x, t_nIt):
        # minimisation step according to Newton's method

        ddf = self.computeHessian(t_f, t_x)

        if t_nIt > self.m_maxIt:
            print('too much iterations')
            return (t_x, ddf, t_nIt)

        df = self.computeGradient(t_f, t_x)

        if ( df * df ).sum() < self.m_tolerance:
            return (t_x, ddf, t_nIt)

        dx = np.linalg.solve(ddf, -df)
        return self.recMinimise(t_f, t_x+dx, t_nIt+1)

    #_________________________

    def minimise(self, t_f, t_x):
        return self.recMinimise(t_f, t_x, 1)

    #_________________________

    def recFindLevel(self, t_f, t_level, t_x, t_nIt):
        # solve t_f(t_x) = t_level
        # assuming that t_x is a scalar

        if t_nIt > self.m_maxIt:
            print('too much iterations')
            return (t_x, t_nIt)

        df = ( t_f(t_x+self.m_dx) - t_f(t_x-self.m_dx) ) / ( 2.0 * self.m_dx )
        if df * df < self.m_tolerance:
            return (t_x, t_nIt)

        dx = - ( t_f(t_x) - t_level ) / df
        if dx * dx < self.m_tolerance:
            return (t_x, t_nIt)

        return self.recFindLevel(t_f, t_level, t_x+dx, t_nIt+1)

    #_________________________

    def findLevel(self, t_f, t_level, t_x):
        return self.recFindLevel(t_f, t_level, t_x, 1)

#__________________________________________________

