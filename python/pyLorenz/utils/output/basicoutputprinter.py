#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/output/
# basicoutputprinter.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/21
#__________________________________________________
#
# class to handle a basic output printer for any simulation
# i.e. only time step number gets printed
#

import numpy as np

#__________________________________________________

class BasicOutputPrinter:

    #_________________________

    def __init__(self, t_ntMod = 100, t_ntFst = 0):
        self.setBasicOutputPrinterParameters(t_ntMod, t_ntFst)

    #_________________________

    def setBasicOutputPrinterParameters(self, t_ntMod = 100, t_ntFst = 0):
        self.m_ntMod = t_ntMod
        self.m_ntFst = t_ntFst

    #_________________________

    def printStart(self, t_simulation):
        # print output when starting simulation
        print('Starting simulation')

    #_________________________

    def printInitialisation(self, t_simulation):
        # print output when initialising simulation
        print('Initialisation')

    #_________________________

    def printStep(self, t_nt, t_simulation):
        # print output for step nt of simulation
        if np.mod(t_nt, self.m_ntMod) == self.m_ntFst:
            print('Running step # '+str(t_nt)+' / '+str(t_simulation.m_Nt))

    #_________________________

    def printEnd(self, t_simulation):
        # print output when ending simulation
        print('Simulation finished')

#__________________________________________________

