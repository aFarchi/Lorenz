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

from basicsimulation import BasicSimulation

#__________________________________________________

class FilterSimulation(BasicSimulation):

    #_________________________

    def setParameters(self, t_Nt, t_Ns, t_ntModObs, t_ntFstObs):
        # set various parameters
        BasicSimulation.setParameters(self, t_Nt)
        self.m_Ns       = t_Ns
        self.m_ntModObs = t_ntModObs
        self.m_ntFstObs = t_ntFstObs

    #_________________________

    def setFilter(self, t_filter):
        # set filter
        self.m_filter = t_filter

    #_________________________

    def setObservationOperator(self, t_obsOp):
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
        self.m_xa_record = np.zeros((self.m_Nt, self.m_model.m_stateDimension))
        self.m_xo_record = np.zeros(((self.m_Nt-self.m_ntFstObs)/self.m_ntModObs, self.m_model.m_stateDimension))

    #_________________________

    def timeStep(self, t_nt):
        self.m_outputPrinter.printStep(t_nt, self)

        if np.mod(t_nt-self.m_ntFstObs, self.m_ntModObs) == 0:
            # observe the truth
            obs = self.m_observationOperator.process(self.m_xt)
            # record observation
            self.m_xo_record[(t_nt-self.m_ntFstObs)/self.m_ntModObs] = obs
            # analyse observation
            self.m_filter.analyse(t_nt, obs)

        # record truth and analyse
        self.m_xt_record[t_nt] = self.m_xt
        self.m_xa_record[t_nt] = self.m_filter.estimate()

        # apply time step
        self.m_xt = self.m_integrator.process(self.m_model, self.m_xt)
        self.m_filter.forecast()

    #_________________________

    def recordToFile(self, t_outputDir='./'):
        BasicSimulation.recordToFile(self, t_outputDir)
        self.m_xa_record.tofile(t_outputDir+'xa_record.bin')
        self.m_xo_record.tofile(t_outputDir+'xo_record.bin')
        ###_____________________________
        ### --->>> TO IMPROVE <<<--- ###
        ###_____________________________
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

