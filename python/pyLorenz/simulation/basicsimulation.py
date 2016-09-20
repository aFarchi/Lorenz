#! /usr/bin/env python

#__________________________________________________
# pyLorenz/simulation/
# basicsimulation.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/20
#__________________________________________________
#
# classes to handle a basic simulation of any model
# i.e. no filtering is performed
#

import numpy as np

#__________________________________________________

class BasicSimulation:

    #_________________________

    def setParameters(self, t_Nt):
        # set number of time steps
        self.m_Nt = t_Nt

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

    def timeStep(self, t_nt):
        self.m_outputPrinter.printStep(t_nt, self)
        # first record truth
        self.m_xt_record[t_nt, :] = self.m_xt[:]
        # then apply time step
        self.m_xt = self.m_integrator.process(self.m_model, self.m_xt)

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

#__________________________________________________

