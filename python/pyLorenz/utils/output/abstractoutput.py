#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/output/
# abstractoutput.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/23
#__________________________________________________
#
# class to handle output for any simulation
#

import numpy as np
import time  as tm

from ..bash.bash import createDir

#__________________________________________________

class AbstractOutput(object):

    #_________________________

    def __init__(self, t_outputDir, t_modWrite, t_modPrint):
        self.setAbstractOutputParameters(t_outputDir, t_modWrite, t_modPrint)

    #_________________________

    def setAbstractOutputParameters(self, t_outputDir, t_modWrite, t_modPrint)
        # output dir
        self.m_outputDir = t_outputDir
        # mod write
        self.m_modWrite  = t_modWrite
        # mod print
        self.m_modPrint  = t_modPrint

    #_________________________

    def openFiles(self, t_filters):
        # open output files
        raise NotImplementedError

    #_________________________

    def closeFiles(self):
        # close output files
        raise NotImplementedError

    #_________________________

    def allocTemporaryArrays(self, t_xDim, t_yDim t_filters):
        # alloc tmp arrays
        raise NotImplementedError

    #_________________________

    def start(self, t_xDim, t_yDim, t_filters):
        # start simulation
        self.m_timeStart = tm.time()
        print('Starting simulation')

        # create output dir
        createDir(self.m_outputDir)
        # open output files
        self.openFiles(t_filters)
        # alloc tmp arrays
        self.allocTemporaryArrays(t_xDim, t_yDim, t_filters)

        # writing counter
        self.m_writingCounter = 0

    #_________________________

    def initialise(self):
        # initialise simulation
        print('Initialisation')

    #_________________________

    def cycle(self, t_nCycle, t_NCycles):
        # print output for cycle nCycle of NCycles
        if np.mod(t_nCycle, self.m_modPrint) == 0:
            elapsedTime = str(tm.time()-self.m_timeStart)
            try:
                estimatedTimeRemaining = str((tm.time()-self.m_timeStartLoop)*(t_NCycles-t_nCycle)/t_nCycle)
            except:
                self.m_timeStartLoop   = tm.time()
                estimatedTimeRemaining = '***'
            print('Running cycle # '+str(t_nCycle)+' / '+str(t_NCycles)+' *** et = '+elapsedTime+' *** etr = '+estimatedTimeRemaining)

    #_________________________

    def finalise(self, t_observationTimes):
        # finalise simulation
        self.writeAll()
        self.writeObservationTimes(t_observationTimes)
        self.closeFiles()

        elapsedTime = str(tm.time()-self.m_timeStart)
        print('Simulation finished')
        print('Total elapsed time = '+elapsedTime)

    #_________________________

    def recordOutput(self, t_filterOrTruthOrObservations, t_output, t_value):
        # record field output for filter or observations
        raise NotImplementedError

    #_________________________

    def writeAll(self):
        # write all to files
        raise NotImplementedError

    #_________________________

    def writeObservationTimes(self, t_observationTimes):
        # write observation times to file
        t_observationTimes.tofile(fileName(self.m_outputDir, 'observations', 'time'))

    #_________________________

    def write(self):
        # check if necessary to write and call writeAll()
        self.m_writingCounter += 1
        if self.m_writingCounter == self.m_nModWrite:
            print('Writing to files')
            self.writeAll()
            self.m_writingCounter = 0

#__________________________________________________

