#! /usr/bin/env python

#__________________________________________________
# pyLorenz/truth/
# truthfromfile.py
#__________________________________________________
# author        : colonel
# last modified : 2016/11/5
#__________________________________________________
#
# class to handle the truth recovered from a file
#

import numpy as np

from abstracttruth import AbstractTruth

#__________________________________________________

class TruthFromFile(AbstractTruth):

    #_________________________

    def __init__(self, t_truthFile, t_observationsFile, t_bufferSize, 
            t_xSpaceDimension, t_ySpaceDimension, t_observationTimes, t_output, t_truthOutputFields):
        AbstractTruth.__init__(self, t_xSpaceDimension, t_ySpaceDimension, t_observationTimes, t_output, t_truthOutputFields)
        self.setTruthFromFileParameters(t_truthFile, t_observationsFile, t_bufferSize)

    #_________________________

    def setTruthFromFileParameters(self, t_truthFile, t_observationsFile, t_bufferSize):
        # truth file
        self.m_truthFile        = t_truthFile.open('rb')
        # observation file
        self.m_observationsFile = t_observationsFile.open('rb')
        # buffer size
        self.m_bufferSize       = t_bufferSize

    #_________________________

    def observation(self):
        # access function for observation array
        return self.m_y[self.m_bufferIndex]

    #_________________________

    def truth(self):
        # access function for truth array
        return self.m_x[self.m_bufferIndex]

    #_________________________

    def recoverFromFiles(self):
        # recover truth and observations from files
        self.m_x           = np.fromfile(self.m_truthFile, count = self.m_bufferSize*self.m_xSpaceDimension)
        self.m_y           = np.fromfile(self.m_observationsFile, count = self.m_bufferSize*self.m_ySpaceDimension)
        self.m_bufferIndex = 0

        curBufSize         = self.m_x.size / self.m_xSpaceDimension
        self.m_x           = self.m_x.reshape((curBufSize, self.m_xSpaceDimension))
        curBufSize         = self.m_y.size / self.m_ySpaceDimension
        self.m_y           = self.m_y.reshape((curBufSize, self.m_ySpaceDimension))

    #_________________________

    def initialise(self):
        # alloc m_x and m_y
        self.recoverFromFiles()
        AbstractTruth.initialise(self)

    #_________________________

    def forecast(self, t_tStart, t_tEnd):
        # integrate truth from tStart to tEnd
        if not np.allclose(t_tStart, t_tEnd):
            self.m_bufferIndex += 1
            if self.m_bufferIndex == self.m_bufferSize:
                self.recoverFromFiles()
        # record time
        self.m_time = t_tEnd

    #_________________________

    def permute(self):
        # permute array to prepare next cycle
        pass

    #_________________________

    def finalise(self):
        # close files
        self.m_truthFile.close()
        self.m_observationsFile.close()

#__________________________________________________

