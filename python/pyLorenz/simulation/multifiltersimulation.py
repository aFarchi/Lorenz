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

from basicsimulation  import BasicSimulation
from filtersimulation import FilterSimulation

#__________________________________________________

class MultiFilterSimulation(FilterSimulation):

    #_________________________

    def setFilter(self, *args):
        pass

    #_________________________

    def setFilters(self, t_filters):
        # set filter list
        self.m_filters = t_filters

    #_________________________

    def addFilter(self, t_filter):
        # add filter to list
        try:
            self.m_filters.append(t_filter)
        except:
            self.m_filters = [t_filter]

    #_________________________

    def initialise(self):
        BasicSimulation.initialise(self)
        ###_______________________
        ### --->>> HACK <<<--- ###
        ###_______________________
        self.m_xt = np.copy(self.m_initialiser.m_mean)
        # initialise the filters
        for tfilter in self.m_filters:
            tfilter.initialise(self.m_initialiser.drawSamples(self.m_Ns))
        # arrays for tracking
        self.m_xa_record = np.zeros((len(self.m_filter), self.m_Nt, self.m_model.m_stateDimension))
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
            for tfilter in self.m_filters:
                tfilter.analyse(t_nt, obs)

        # record truth and analyse
        self.m_xt_record[t_nt] = self.m_xt
        for nf in len(self.m_filters):
            self.m_xa_record[nf, t_nt] = self.m_filters[nf].estimate()

        # apply time step
        self.m_xt = self.m_integrator.process(self.m_model, self.m_xt)
        for tfilter in self.m_filters:
            tfilter.forecast()

    #_________________________

    def recordToFile(self, t_outputDir='./'):
        BasicSimulation.recordToFile(self, t_outputDir)
        self.m_xa_record.tofile(t_outputDir+'xa_record.bin')
        self.m_xo_record.tofile(t_outputDir+'xo_record.bin')
        ###_____________________________
        ### --->>> TO IMPROVE <<<--- ###
        ###_____________________________
        #self.m_filter.resampledSteps().tofile(t_outputDir+'nt_resampling.bin')

    #_________________________

    def computeFilterPerformance(self, t_ntDrop=0):
        mse         = np.zeros(self.m_Nt)
        self.meanFP = np.zeros((len(self.m_filters), self.m_Nt))
        for nf in len(self.m_filters):
            mse                = np.sqrt ( np.power ( self.m_xt_record - self.m_xa_record[nf] , 2.0 ) . sum ( axis = 1 ) )
            mse[:t_ntDrop]     = 0.0 
            self.meanFP[nf]    = mse.cumsum() / np.maximum( np.arange(self.m_Nt) - ( t_ntDrop - 1.0 ) , 1.0 )

#__________________________________________________

