#! /usr/bin/env python

#__________________________________________________
# pyLorenz/simulation/
# filtersimulation.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/20
#__________________________________________________
#
# classes to handle a basic simulation of any model
# with a filtering process
#

import numpy as np

#__________________________________________________

class FilterSimulation:

    #_________________________

    def setParameters(self, t_Nt, t_Ns, t_ntModObs, t_ntFstObs):
        # set various parameters
        self.m_Nt       = t_Nt
        self.m_Ns       = t_Ns
        self.m_ntModObs = t_ntModObs
        self.m_ntFstObs = t_ntFstObs

    #_________________________

    def setFilter(self, t_filter):
        # set filter
        self.m_filter = t_filter

    #_________________________

    def setModel(self, t_model):
        # set model
        self.m_model = t_model

    #_________________________

    def setIntegrator(self, t_integrator):
        # set integrator
        self.m_integrator = t_integrator
    #_________________________

    def setInitialiser(self, t_initialiser):
        # set initialiser
        self.m_initialiser = t_initialiser

    #_________________________

    def setObservationOperator(self, t_obsOp):
        # set observation operator
        self.m_observationOperator = t_obsOp

    #_________________________

    def setOutputPrinter(self, t_outputPrinter):
        # set output printer
        self.m_outputPrinter = t_outputPrinter

    #_________________________

    def initialise(self):
        self.m_outputPrinter.printInitialisation(self)
        # initialise the truth
        #self.m_xt = self.m_initialiser.drawSample()
        # HACK
        self.m_xt = np.copy(self.m_initialiser.m_mean)
        # initialise the filter
        self.m_filter.initialise(self.m_initialiser.drawSamples(self.m_Ns))
        # arrays for tracking
        self.m_xt_record = np.zeros((self.m_Nt, self.m_model.m_stateDimension))
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

    def run(self):
        # run function
        self.m_outputPrinter.printStart(self)

        self.initialise()
        for nt in np.arange(self.m_Nt):
            self.timeStep(nt)

        self.m_outputPrinter.printEnd(self)

    #_________________________

    def recordToFile(self, t_outputDir='./'):
        self.m_xt_record.tofile(t_outputDir+'xt_record.bin')
        self.m_xa_record.tofile(t_outputDir+'xa_record.bin')
        self.m_xo_record.tofile(t_outputDir+'xo_record.bin')
        (1.0*np.array(self.m_filter.m_resampled)).tofile(t_outputDir+'nt_resampling.bin')

#__________________________________________________

