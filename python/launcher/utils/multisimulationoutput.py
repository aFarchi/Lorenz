#! /usr/bin/env python

#__________________________________________________
# launcher/utils/
# multisimulationoutput.py
#__________________________________________________
# author        : colonel
# last modified : 2016/11/14
#__________________________________________________
#
# output for multi-simulation
#

import time as tm

#__________________________________________________

class MultiSimulationOutput(object):

    #_________________________

    def __init__(self):
        pass

    #_________________________

    def makeTopDir(self, t_outputTopDir):
        # making output top-directory
        msg = 'Making output top-directory: ' + t_outputTopDir
        print(msg)

    #_________________________

    def makeSubDir(self, t_outputSubDir):
        # making output sub-directory
        msg = 'Making output sub-directory: ' + t_outputSubDir
        print(msg)

    #_________________________

    def archiveInDir(self, t_directory):
        # archiving in directory
        msg = 'Archiving in directory: ' + t_directory
        print(msg)

    #_________________________

    def startMultiSimulation(self, t_nSim, t_max_workers):
        # start multi simulation
        msg  = '__________________________________________________\n'
        msg += 'Starting multi simulation\n'
        msg += 'number of simulations     = ' + str(t_nSim) + '\n'
        msg += 'maximum number of workers = ' + str(t_max_workers) + '\n'
        print(msg)
        return tm.time()

    #_________________________

    def endMultiSimulation(self, t_tOrigin):
        # start multi simulation
        tt   = tm.time() - t_tOrigin
        msg  = '\n'
        msg += 'multi simulation finished\n'
        msg += 'total time taken = ' + str(tt) + '\n'
        msg += '__________________________________________________\n'
        print(msg)

    #_________________________

    def startSimulation(self, t_tOrigin, t_n, t_N):
        # start one simulation
        tStart  = tm.time()
        msg     = 'Running simulation # ' + str(t_n+1) + ' / ' + str(t_N)
        msg    += '  *** et = ' + str(tStart-t_tOrigin)
        print(msg)
        return tStart

    #_________________________

    def endSimulation(self, t_tStart, t_n, t_N):
        # start one simulation
        tt   = tm.time() - t_tStart
        msg  = '    > Finished simulation # ' + str(t_n+1) + ' / ' + str(t_N)
        msg += '  *** time taken = ' + str(tt)
        print(msg)
        return tt

    #_________________________

    def timoutSimulation(self, t_tStart, t_n, t_N):
        # timeout warning message
        tt   = tm.time() - t_tStart
        msg  = '  > > Timeout for simulation # ' + str(t_n+1) + ' / ' + str(t_N)
        msg += '  *** time taken = ' + str(tt)
        print(msg)
        return tt

    #_________________________

    def unexpectedCrashSimulation(self, t_tStart, t_n, t_N):
        # unexpected crash warning message
        tt   = tm.time() - t_tStart
        msg  = '  > > Unexpected crash for simulation # ' + str(t_n+1) + ' / ' + str(t_N)
        msg += '  *** time taken = ' + str(tt)
        print(msg)
        return tt

#__________________________________________________

