#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/integration/
# abstractintegrator.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/9
#__________________________________________________
#
# classes to handle integration processes
#

import numpy        as np
import numpy.random as rnd

from ...model.lorenz63 import DeterministicLorenz63Model

#__________________________________________________

class AbstractIntegrator(object):

    #_________________________

    def __init__(self, t_dt = 0.01, t_model = DeterministicLorenz63Model()):
        self.setIntegratorParameters(t_dt, t_model)

    #_________________________

    def setIntegratorParameters(self, t_dt = 0.01, t_model = DeterministicLorenz63Model()):
        # integration parameters
        self.m_dt    = t_dt
        self.m_model = t_model

        # space dimension
        self.m_spaceDimension = self.m_model.m_spaceDimension

    #_________________________

    def trajectoryPDF(self, t_ntStart, t_ntEnd, t_xStart, t_x):
        # - pdf (in log scale) associated to the probability of the trajectory x[ntStart+1:ntEnd+1] given x[ntStart] = xStart.
        # For DeterministicProcess subclasses, it will not work since self.m_errorGenerator does not exist, which is ok.
        # For MultiStochasticProces subclasses either (for the same reason) : for theses subclasses the pdf is
        # more complex and depends on the integration scheme (in particular it depends on the steps of the integration).
        # Hence it must be implemented differently for each integration scheme.

        # shortcuts
        r = t_ntEnd - t_ntStart
        d = self.m_spaceDimension

        # model error when integrating xStart
        me = t_x[:d] - self.deterministicProcess(t_xStart, t_ntStart)
        # pdf : - 2 * log ( p ( x[ntStart+1] | x[ntStart] ) )
        p  = - self.m_errorGenerator.pdf(me, t_ntStart)
        for nt in np.arange(r-1):
            me  = t_x[d*(nt+1):d*(nt+2)] - self.deterministicProcess(t_x[d*nt:d*(nt+1)], t_ntStart+nt+1)
            p  -= self.m_errorGenerator.pdf(me, t_ntStart+nt+1)
        return p 

    #_________________________

    def initialiseTrajectory(self, t_ntStart, t_ntEnd, t_xStart):
        # initialise a trajectory that will be used to minimise the trajectoryPDF().
        # For MultiStochasticProces subclasses, you may want to reimplement this method.

        # shortcuts
        r = t_ntEnd - t_ntStart
        d = self.m_spaceDimension

        # initialise trajectory with a model run
        trajectory     = np.zeros(d*r)
        trajectory[:d] = self.deterministicProcess(t_xStart, t_ntStart)
        for nt in np.arange(r-1):
            trajectory[d*(nt+1):d*(nt+2)] = self.deterministicProcess(trajectory[d*nt:d*(nt+1)], t_ntStart+nt+1)
        return trajectory

    #_________________________

    def randomSampleForTrajectory(self, t_ntStart, t_ntEnd):
        # draw a random sample according to N(0, I)
        # whose dimension matches the dimension of initialiseTrajectory().
        # For MultiStochasticProces subclasses, you may want to reimplement this method. 

        # shortcuts
        r = t_ntEnd - t_ntStart
        d = self.m_spaceDimension
        return rnd.standard_normal(d*r)

#__________________________________________________

