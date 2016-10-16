#! /usr/bin/env python

#__________________________________________________
# pyLorenz/simulation/
# basicsimulation.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/15
#__________________________________________________
#
# classes to handle the simulation of a model
# with a filtering process
#

import numpy as np

#__________________________________________________

class BasicSimulation(object):

    #_________________________

    def __init__(self, t_integrator, t_initialiser, t_outputPrinter, t_observationTimes, t_observationOperator):
        self.setBasicSimulationParameters(t_integrator, t_initialiser, t_outputPrinter, t_observationTimes, t_observationOperator)

    #_________________________

    def setBasicSimulationParameters(self, t_integrator, t_initialiser, t_outputPrinter, t_observationTimes, t_observationOperator):
        # space dimension
        self.m_spaceDimension      = t_integrator.m_spaceDimension
        # integrator
        self.m_integrator          = t_integrator
        # initialiser              
        self.m_initialiser         = t_initialiser
        # output printer
        self.m_outputPrinter       = t_outputPrinter
        # observation operator
        self.m_observationOperator = t_observationOperator
        # observation times
        self.m_observationTimes    = t_observationTimes

    #_________________________

    def initialise(self):
        self.m_outputPrinter.printInitialisation(self)

        # size of the integration arrays
        (sx, sdx)    = self.m_integrator.iArrayMaxSize(self.m_observationTimes)

        # initialise the truth
        self.m_xt    = np.zeros((sx, self.m_spaceDimension))
        self.m_dxt   = np.zeros((sdx, self.m_spaceDimension))
        self.m_xt[0] = self.m_initialiser.initialiseTruth()

        #-------------------------------------------------------------
        # Array for tracking  (if there is enough memory to afford it)
        #-------------------------------------------------------------
        self.m_xt_record = np.zeros((self.m_observationTimes.size, self.m_spaceDimension))
        self.m_xo_record = np.zeros((self.m_observationTimes.size, self.m_observationOperator.m_spaceDimension))

    #_________________________

    def cycle(self, t_tStart, t_tEnd, t_index):
        # perform simulation from tStart to tEnd
        self.m_outputPrinter.printCycle(t_index, self.m_observationTimes.size)

        # aux index variable
        iEnd = self.m_integrator.indexTEnd(t_tStart, t_tEnd)

        # integrate truth and record it
        self.m_integrator.integrate(self.m_xt, t_tStart, t_tEnd, self.m_dxt)
        self.m_xt_record[t_index] = self.m_xt[iEnd]

        # observe the truth at time tEnd and record it
        self.m_xo_record[t_index] = self.m_observationOperator.observe(self.m_xt[iEnd], t_tEnd)

        # permute self.m_xt to prepare next cycle
        self.m_xt[0] = self.m_xt[iEnd]

    #_________________________

    def run(self):
        # run function
        self.m_outputPrinter.printStart(self)

        # initialisation
        self.initialise()

        # simulation from 0.0 to self.m_observationTimes[0]
        tStart = 0.0
        tEnd   = self.m_observationTimes[0]
        index  = 0
        self.cycle(tStart, tEnd, index)
        for i in range(self.m_observationTimes.size-1):
            tStart = self.m_observationTimes[i]
            tEnd   = self.m_observationTimes[i+1]
            index  = i+1
            # simulation from tStart to tEnd
            self.cycle(tStart, tEnd, index)

        self.m_outputPrinter.printEnd()

    #_________________________

    def recordToFile(self, t_outputDir):
        self.m_xt_record.tofile(t_outputDir+'xt_record.bin')
        self.m_xo_record.tofile(t_outputDir+'xo_record.bin')
        self.m_observationTimes.tofile(t_outputDir+'observationTimes.bin')

#__________________________________________________

