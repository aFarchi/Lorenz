#! /usr/bin/env python

#__________________________________________________
# pyLorenz/simulation/
# basicsimulation.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/6
#__________________________________________________
#
# classes to handle a basic simulation of any model
# i.e. no filtering is performed
#

import numpy as np

from ..utils.integration.rk4integrator                     import DeterministicRK4Integrator
from ..utils.initialisation.gaussianindependantinitialiser import GaussianIndependantInitialiser
from ..utils.output.basicoutputprinter                     import BasicOutputPrinter

#__________________________________________________

class BasicSimulation(object):

    #_________________________

    def __init__(self, t_Nt = 1000, t_integrator = DeterministicRK4Integrator(),
            t_initialiser = GaussianIndependantInitialiser(), t_outputPrinter = BasicOutputPrinter()):
        self.setBasicSimulationParameters(t_Nt, t_integrator, t_initialiser, t_outputPrinter)

    #_________________________

    def setBasicSimulationParameters(self, t_Nt = 1000, t_integrator = DeterministicRK4Integrator(), 
            t_initialiser = GaussianIndependantInitialiser(), t_outputPrinter = BasicOutputPrinter()):
        # set number of time steps
        self.m_Nt             = t_Nt
        # set integrator
        self.m_integrator     = t_integrator
        # set initialiser
        self.m_initialiser    = t_initialiser
        # set output printer
        self.m_outputPrinter  = t_outputPrinter
        # space dimension
        self.m_spaceDimension = self.m_integrator.m_spaceDimension

    #_________________________

    def initialise(self):
        self.m_outputPrinter.printInitialisation(self)
        # initialise the truth
        self.m_xt = self.m_initialiser.initialiseTruth()
        # Array for tracking
        self.m_xt_record       = np.zeros((self.m_Nt, self.m_spaceDimension))
        self.m_xt_record[0, :] = self.m_xt[:]

    #_________________________

    def timeStep(self, t_nt):
        self.m_outputPrinter.printStep(t_nt, self)
        # first record truth
        self.m_xt_record[t_nt :] = self.m_xt[:]
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

    def recordToFile(self, t_outputDir = './'):
        self.m_xt_record.tofile(t_outputDir+'xt_record.bin')

#__________________________________________________

