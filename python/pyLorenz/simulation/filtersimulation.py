#! /usr/bin/env python

#__________________________________________________
# pyLorenz/simulation/
# filtersimulation.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/21
#__________________________________________________
#
# classes to handle a basic simulation of any model
# with a filtering process
#

import numpy as np

from basicsimulation                       import BasicSimulation
from ..utils.integration.rk4integrator     import DeterministicRK4Integrator
from ..utils.random.independantgaussianrng import IndependantGaussianRNG
from ..utils.output.basicoutputprinter     import BasicOutputPrinter
from ..filters.pf.sir                      import SIRPF
from ..filters.kalman.stochasticenkf       import StochasticEnKF
from ..observations.iobservations          import StochasticIObservations

default = object()

#__________________________________________________

class FilterSimulation(BasicSimulation):

    #_________________________

    def __init__(self, t_Nt = 1000, t_integrator = DeterministicRK4Integrator(), 
            t_initialiser = IndependantGaussianRNG(), t_outputPrinter = BasicOutputPrinter(), t_Ns = 10, t_ntObs = default,
            t_filter = StochasticEnKF(), t_obsOp = StochasticIObservations()):
        BasicSimulation.__init__(self, t_Nt, t_integrator, t_initialiser, t_outputPrinter)
        self.setFilterSimulationParameters(t_Ns, t_ntObs, t_filter, t_obsOp)

    #_________________________

    def setFilterSimulationParameters(self, t_Ns = 10, t_ntObs = default, t_filter = StochasticEnKF(), t_obsOp = StochasticIObservations()):
        # set number of particles / samples
        self.m_Ns       = t_Ns
        # set observation times
        if t_ntObs is default:
            self.m_ntObs = np.arange(self.m_Nt)
        else:
            self.m_ntObs = t_ntObs
        # set filter
        self.m_filter = t_filter
        # set observation operator
        self.m_observationOperator = t_obsOp

    #_________________________

    def initialise(self):
        BasicSimulation.initialise(self)
        ###_______________________
        ### --->>> HACK <<<--- ###
        ###_______________________
        self.m_xt = np.copy(self.m_initialiser.m_mean)
        # initialise the filter
        self.m_filter.initialise(self.m_initialiser.drawSamples(self.m_Ns))
        # arrays for tracking
        self.m_xa_record = np.zeros((self.m_Nt, self.m_integrator.m_model.m_stateDimension))
        self.m_xo_record = np.zeros((self.m_Nt, self.m_integrator.m_model.m_stateDimension))

    #_________________________

    def analyseCycle(self, t_ntStart, t_ntEnd):
        if t_ntEnd < t_ntStart:
            return
        # perform an algorithm step
        # t_ntStart is the current time step
        # t_ntEnd is the next time step where a measurement is available
        self.m_outputPrinter.printStep(t_ntStart, self)

        # apply time step to the truth and record it
        for nt in np.arange(t_ntEnd-t_ntStart) + t_ntStart:
            self.m_xt            = self.m_integrator.process(self.m_xt, nt)
            self.m_xt_record[nt] = self.m_xt

        # observe the truth at time step t_ntEnd and record it
        observation               = self.m_observationOperator.process(self.m_xt, t_ntEnd*self.m_integrator.m_dt)
        self.m_xo_record[t_ntEnd] = observation

        # forecast until time step t_ntEnd
        # pententially making use of the observation
        # (e.g. for sampling according to a proposal)
        if t_ntEnd > t_ntStart:
            self.m_xa_record[t_ntStart:t_ntEnd] = self.m_filter.forecast(t_ntStart, t_ntEnd, observation)
        #for nt in np.arange(t_ntEnd-t_ntStart) + t_ntStart:
            #self.m_filter.forecast(t_nt = nt, t_nextObservationTime = t_ntEnd, t_nextObservation = observation)
            #self.m_xa_record[nt] = self.m_filter.estimate()

        # Analyse observation
        self.m_xa_record[t_ntEnd] = self.m_filter.analyse(t_ntEnd, observation)

    #_________________________

    def run(self):
        # run function
        self.m_outputPrinter.printStart(self)

        self.initialise()
        self.analyseCycle(0, self.m_ntObs[0])
        for i in np.arange(self.m_ntObs.size-1):
            self.analyseCycle(self.m_ntObs[i], self.m_ntObs[i+1])
        self.analyseCycle(self.m_ntObs[-1], self.m_Nt-1)

        self.m_outputPrinter.printEnd(self)

    #_________________________

    def recordToFile(self, t_outputDir='./'):
        BasicSimulation.recordToFile(self, t_outputDir)
        self.m_xa_record.tofile(t_outputDir+'xa_record.bin')
        self.m_xo_record.tofile(t_outputDir+'xo_record.bin')
        ###_____________________________
        ### --->>> TO IMPROVE <<<--- ###
        ###_____________________________
        (1.0*self.m_ntObs).tofile(t_outputDir+'nt_obs.bin')
        if isinstance(self.m_filter, SIRPF):
            self.m_filter.resampledSteps().tofile(t_outputDir+'nt_resampling.bin')

    #_________________________

    def computeFilterPerformance(self, t_ntDrop=0):
        mse            = np.sqrt ( np.power ( self.m_xt_record - self.m_xa_record , 2.0 ) . sum ( axis = 1 ) )
        mse[:t_ntDrop] = 0.0 
        self.meanFP    = mse.cumsum() / np.maximum( np.arange(self.m_Nt) - ( t_ntDrop - 1.0 ) , 1.0 )

    #_________________________
    
    def filterPerformanceToFile(self, t_outputDir='./'):
        self.meanFP.tofile(t_outputDir+'meanFP.bin')

#__________________________________________________

