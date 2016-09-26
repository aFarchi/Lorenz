#! /usr/bin/env python

#__________________________________________________
# pyLorenz/simulation/
# multifiltersimulation.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/21
#__________________________________________________
#
# classes to handle a basic simulation of any model
# with multiple filtering processes
#

import numpy as np

from basicsimulation                       import BasicSimulation
from filtersimulation                      import FilterSimulation
from ..utils.integration.rk4integrator     import DeterministicRK4Integrator
from ..utils.random.independantgaussianrng import IndependantGaussianRNG
from ..utils.output.basicoutputprinter     import BasicOutputPrinter
from ..filters.pf.sir                      import SIRPF
from ..filters.kalman.stochasticenkf       import StochasticEnKF
from ..observations.iobservations          import StochasticIObservations

default = object()

#__________________________________________________

class MultiFilterSimulation(FilterSimulation):

    #_________________________

    def __init__(self, t_Nt = 1000, t_integrator = DeterministicRK4Integrator(),
            t_initialiser = IndependantGaussianRNG(), t_outputPrinter = BasicOutputPrinter(), t_Ns = 10, t_ntObs = default, t_obsOp = StochasticIObservations()):
        FilterSimulation.__init__(self, t_Nt, t_integrator, t_initialiser, t_outputPrinter, t_Ns, t_ntObs, [], t_obsOp)

    #_________________________

    def addFilter(self, t_filter):
        # add filter to list
        self.m_filter.append(t_filter)

    #_________________________

    def initialise(self):
        BasicSimulation.initialise(self)
        ###_______________________
        ### --->>> HACK <<<--- ###
        ###_______________________
        self.m_xt           = np.copy(self.m_initialiser.m_mean)
        self.m_xt_record[0] = self.m_xt
        # initialise the filters
        for tfilter in self.m_filter:
            tfilter.initialise(self.m_initialiser.drawSamples(self.m_Ns))
        # arrays for tracking
        self.m_xa_record = np.zeros((len(self.m_filter), self.m_Nt, self.m_integrator.m_model.m_stateDimension))
        self.m_xo_record = np.zeros((self.m_Nt, self.m_integrator.m_model.m_stateDimension))
        for i in np.arange(len(self.m_filter)):
            self.m_xa_record[i, 0] = self.m_filter[i].estimate()

    #_________________________

    def analyseCycle(self, t_ntStart, t_ntEnd):
        if t_ntEnd < t_ntStart:
            return
        # perform an algorithm step
        # t_ntStart is the current time step
        # t_ntEnd is the next time step where a measurement is available
        if t_ntEnd > t_ntStart:
            self.m_outputPrinter.printStep(t_ntStart, self)

        # apply time step to the truth and record it
        for nt in np.arange(t_ntEnd-t_ntStart) + t_ntStart:
            self.m_xt              = self.m_integrator.process(self.m_xt, nt)
            self.m_xt_record[nt+1] = self.m_xt

        # observe the truth at time step t_ntEnd and record it
        observation               = self.m_observationOperator.process(self.m_xt, t_ntEnd*self.m_integrator.m_dt)
        self.m_xo_record[t_ntEnd] = observation

        # forecast until time step t_ntEnd
        # pententially making use of the observation
        # (e.g. for sampling according to a proposal)
        if t_ntEnd > t_ntStart:
            for i in np.arange(len(self.m_filter)):
                self.m_xa_record[i, t_ntStart+1:t_ntEnd+1] = self.m_filter[i].forecast(t_ntStart, t_ntEnd, observation)

        # Analyse observation
        for i in np.arange(len(self.m_filter)):
            self.m_xa_record[i, t_ntEnd] = self.m_filter[i].analyse(t_ntEnd, observation)

    #_________________________

    def recordToFile(self, t_outputDir='./'):
        BasicSimulation.recordToFile(self, t_outputDir)
        self.m_xa_record.tofile(t_outputDir+'xa_record.bin')
        self.m_xo_record.tofile(t_outputDir+'xo_record.bin')
        ###_____________________________
        ### --->>> TO IMPROVE <<<--- ###
        ###_____________________________
        (1.0*self.m_ntObs).tofile(t_outputDir+'nt_obs.bin')
        #self.m_filter.resampledSteps().tofile(t_outputDir+'nt_resampling.bin')

    #_________________________

    def computeFilterPerformance(self, t_ntDrop=0):
        self.meanFP = np.zeros((len(self.m_filter), self.m_Nt))
        self.FP     = np.zeros((len(self.m_filter), self.m_Nt))
        for i in np.arange(len(self.m_filter)):
            mse            = np.sqrt ( np.power ( self.m_xt_record - self.m_xa_record[i] , 2.0 ) . sum ( axis = 1 ) )
            self.FP[i]     = np.copy(mse)
            mse[:t_ntDrop] = 0.0 
            self.meanFP[i] = mse.cumsum() / np.maximum( np.arange(self.m_Nt) - ( t_ntDrop - 1.0 ) , 1.0 )

#__________________________________________________

