#! /usr/bin/env python

#__________________________________________________
# pyLorenz/simulation/
# basicsimulation.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/21
#__________________________________________________
#
# classes to handle a basic simulation of any model
# i.e. no filtering is performed
#

import numpy as np

from ..utils.integration.rk4integrator     import DeterministicRK4Integrator
from ..utils.random.independantgaussianrng import IndependantGaussianRNG
from ..utils.output.basicoutputprinter     import BasicOutputPrinter

#__________________________________________________

class BasicSimulation(object):

    #_________________________

    def __init__(self, t_Nt = 1000, t_integrator = DeterministicRK4Integrator(),
            t_initialiser = IndependantGaussianRNG(), t_outputPrinter = BasicOutputPrinter()):
        self.setBasicSimulationParameters(t_Nt, t_integrator, t_initialiser, t_outputPrinter)

    #_________________________

    def setBasicSimulationParameters(self, t_Nt = 1000, t_integrator = DeterministicRK4Integrator(), 
            t_initialiser = IndependantGaussianRNG(), t_outputPrinter = BasicOutputPrinter()):
        # set number of time steps
        self.m_Nt = t_Nt
        # set integrator
        self.m_integrator = t_integrator
        # set initialiser
        self.m_initialiser = t_initialiser
        # set output printer
        self.m_outputPrinter = t_outputPrinter

    #_________________________

    def initialise(self):
        self.m_outputPrinter.printInitialisation(self)
        # initialise the truth
        self.m_xt = self.m_initialiser.drawSample()
        # Array for tracking
        self.m_xt_record = np.zeros((self.m_Nt, self.m_integrator.m_model.m_stateDimension))

    #_________________________

    def timeStep(self, t_nt):
        self.m_outputPrinter.printStep(t_nt, self)
        # first record truth
        self.m_xt_record[t_nt] = self.m_xt
        # then apply time step
        self.m_xt = self.m_integrator.process(self.m_xt, t_nt)

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

