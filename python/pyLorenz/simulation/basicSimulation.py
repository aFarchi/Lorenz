#! /usr/bin/env python

#__________________________________________________
# pyLorenz/simulation/
# basicSimulation.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/20
#__________________________________________________
#
# class to handle a basic simulation of any model
# i.e. no filtering is performed
#

import numpy as np

#__________________________________________________

class BasicSimulation:

    #_________________________

    def setModel(self, t_model):
        # set model
        self.m_model = t_model

    #_________________________

    def setInitialiser(self, t_initialiser):
        # set initialiser
        self.m_initialiser = t_initialiser

    #_________________________

    def setSimulationParameters(self, t_Nt):
        # set number of time steps
        self.m_Nt = t_Nt

    #_________________________

    def setOutputPrinter(self, t_outputPrinter):
        # set output printer
        self.m_outputPrinter = t_outputPrinter

    #_________________________

    def initialise(self):
        self.m_outputPrinter.printInitialisation(self)
        # initialise the truth
        self.m_xt = self.m_initialiser.drawSample()
        # Array for tracking
        self.m_xt_record = np.zeros((self.m_Nt, 3))

    #_________________________

    def stochasticProcessForward(self, t_nt):
        # first record truth
        self.m_xt_record[t_nt, :] = self.m_xt[:]
        # then apply time step
        self.m_xt = self.m_model.stochasticProcessForward(self.m_xt)

    #_________________________

    def deterministicProcessForward(self, t_nt):
        # first record truth
        self.m_xt_record[t_nt, :] = self.m_xt[:]
        # then apply time step
        self.m_xt = self.m_model.deterministicProcessForward(self.m_xt)

    #_________________________

    def run(self, t_stochastic):
        # run function
        self.m_outputPrinter.printStart(self)
        self.initialise()

        if t_stochastic:
            for nt in np.arange(self.m_Nt):
                self.m_outputPrinter.printStep(nt, self)
                self.stochasticProcessForward(nt)
        else:
            for nt in np.arange(self.m_Nt):
                self.m_outputPrinter.printStep(nt, self)
                self.deterministicProcessForward(nt)

        self.m_outputPrinter.printEnd(self)

    #_________________________

    def recordToFile(self, t_outputDir='./'):
        self.m_xt_record.tofile(t_outputDir+'xt_record.bin')

#__________________________________________________

