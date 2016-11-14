#! /usr/bin/env python

#__________________________________________________
# launcher/utils/
# multisimulationlauncher.py
#__________________________________________________
# author        : colonel
# last modified : 2016/11/14
#__________________________________________________
#
# launcher for multi-simulation
#

from subprocess import run, TimeoutExpired, CalledProcessError

#__________________________________________________

class MultiSimulationLauncher(object):

    #_________________________

    def __init__(self, t_executorClass, t_max_workers, t_chunksize, t_output):
        # executor class : concurrent.futures.ProcessPoolExecutor or concurrent.futures.ThreadPoolExecutor
        self.m_executorClass = t_executorClass
        # maximum numbers of workers for the executor
        self.m_max_workers   = t_max_workers
        # chunksize for the executor
        self.m_chunksize     = t_chunksize
        # output
        self.m_output        = t_output

    #_________________________

    def runSimulations(self, t_target, t_common, t_configList):
        # run the program for each config in configList
        # returns the time taken for each config
        tOrigin             = self.m_output.startMultiSimulation(len(t_configList), self.m_max_workers)
        t_common.m_tOrigin  = tOrigin

        configurations       = [(t_common, n, config) for (n, config) in enumerate(t_configList)]

        with self.m_executorClass(max_workers = self.m_max_workers) as executor:
            result = executor.map(t_target, configurations, chunksize = self.m_chunksize)

        self.m_output.endMultiSimulation(tOrigin)
        return result

#__________________________________________________

