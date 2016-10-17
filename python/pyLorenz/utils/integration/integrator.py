#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/integration/
# integrator.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/9
#__________________________________________________
#
# classes to handle integration processes
#
# DeterministicIntegrator has no model error.
# In BasicStochasticIntegrator instance, model error is added at the end of the integration.
# In StochasticIntegrator, model error is added after each integration substep.
#

import numpy        as np
#import numpy.random as rnd

#__________________________________________________

class DeterministicIntegrator(object):

    #_________________________

    def __init__(self, t_integrationStep):
        self.setDeterministicIntegratorParameters(t_integrationStep)

    #_________________________

    def setDeterministicIntegratorParameters(self, t_integrationStep):
        # space dimension
        self.m_spaceDimension  = t_integrationStep.m_spaceDimension
        # integration step
        self.m_integrationStep = t_integrationStep
        # number of integration substeps per integration step
        self.m_nIntSStep       = t_integrationStep.m_nSStep
        # time step interval
        self.m_dt              = t_integrationStep.m_dt

    #_________________________

    def deterministicIntegrate(self, t_x, t_tStart, t_tEnd, t_dx):
        # integrate x from tStart to step tEnd
        # first dimension (time) of x must have size at least equal to 1 + ( t_ntEnd - t_ntStart ) * self.m_nIntSStep
        # first dimension (time) of dx (working array) must have size at least equal to 1 + self.m_nIntSStep
        Nt = np.rint( ( t_tEnd - t_tStart ) / self.m_dt ).astype(int)
        for nt in range(Nt):
            self.m_integrationStep.deterministicIntegrate(t_x[nt*self.m_nIntSStep:(nt+1)*self.m_nIntSStep+1], t_tStart+nt*self.m_dt, t_dx)

    #_________________________

    def integrate(self, t_x, t_tStart, t_tEnd, t_dx):
        # call deterministicIntegrate() method
        self.deterministicIntegrate(t_x, t_tStart, t_tEnd, t_dx)

    #_________________________

    def iArrayMaxSize(self, t_times):
        # maximum size of the integration arrays to integrate between times
        longestCycle = np.diff(t_times).max()
        longestCycle = np.rint( longestCycle / self.m_dt ).astype(int)
        return ( longestCycle * self.m_nIntSStep + 1 , self.m_nIntSStep )

    #_________________________

    def indexTEnd(self, t_tStart, t_tEnd):
        # return index of x at time tEnd for an array containing x that have been integrated from tStart to tEnd
        return np.rint( ( t_tEnd - t_tStart ) / self.m_dt ).astype(int) * self.m_nIntSStep

#__________________________________________________

class StochasticIntegrator(DeterministicIntegrator):

    #_________________________

    def __init__(self, t_integrationStep):
        DeterministicIntegrator.__init__(self, t_integrationStep)

    #_________________________

    def integrate(self, t_x, t_tStart, t_tEnd, t_dx):
        # the same as deterministicIntegrate() but call stochasticIntegrate() instead of deterministicIntegrate()
        Nt = np.rint( ( t_tEnd - t_tStart ) / self.m_dt ).astype(int)
        for nt in range(Nt):
            self.m_integrationStep.stochasticIntegrate(t_x[nt*self.m_nIntSStep:(nt+1)*self.m_nIntSStep+1], t_tStart+nt*self.m_dt, t_dx)

#__________________________________________________

class BasicStochasticIntegrator(DeterministicIntegrator):

    #_________________________

    def __init__(self, t_integrationStep):
        DeterministicIntegrator.__init__(self, t_integrationStep)

    #_________________________

    def integrate(self, t_x, t_tStart, t_tEnd, t_dx):
        # add random error only at the end of the process with some trick
        self.deterministicIntegrate(t_x, t_tStart, t_tEnd, t_dx)
        dt      = t_tEnd - t_tStart
        i       = self.indexTEnd(t_tStart, t_tEnd)
        # trick
        t_x[i] += self.m_integrationStep.m_errorGenerator.drawSamples(t_tStart, t_x[i].shape, np.sqrt(dt))

    #_________________________

    def errorCovarianceMatrix_diag(self, t_tStart, t_tEnd):
        # return the covariance matrix used for the tick
        dt = t_tEnd - t_tStart
        return self.m_integrationStep.m_errorGenerator.covarianceMatrix_diag(t_tStart) * dt

#__________________________________________________

"""
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
            me  = t_x[d*(nt+1):d*(nt+2)] - self.m_deterministicProcess(t_x[d*nt:d*(nt+1)], t_ntStart+nt+1)
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
"""

