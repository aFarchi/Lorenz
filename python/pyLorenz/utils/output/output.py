#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/output/
# customoutput.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/29
#__________________________________________________
#
# class to handle output for any simulation
# RMSE and estimation are writte
#

import numpy as np
import time  as tm

from ..auxiliary.bash import createDir

#__________________________________________________

class Output(object):

    #_________________________

    def __init__(self, t_outputDir, t_modWrite, t_modPrint):
        self.setOutputParameters(t_outputDir, t_modWrite, t_modPrint)

    #_________________________

    def setOutputParameters(self, t_outputDir, t_modWrite, t_modPrint):
        # output dir
        self.m_outputDir = t_outputDir
        # mod write
        self.m_modWrite  = t_modWrite
        # mod print
        self.m_modPrint  = t_modPrint

    #_________________________

    def start(self):
        # start simulation
        self.m_timeStart = tm.time()
        print('Starting simulation')
        # create output dir
        print('Creating output directory')
        createDir(self.m_outputDir)
        # writing counter
        self.m_writingCounter = 0
        # output files
        self.m_files          = {}
        # temporary arrays
        self.m_tmpArrays      = {}

    #_________________________

    def initialise(self):
        # initialise simulation
        print('Initialisation')

    #_________________________

    def initialiseTruthOutput(self, t_truthOutputFields, t_observationsOutputFields, t_tmpRecordShape):
        # open output files and alloc temporary record arrays
        for (truthOrObservations, outputFields) in zip(['truth', 'observations'], [t_truthOutputFields, t_observationsOutputFields]):
            self.m_files[truthOrObservations]     = {}
            self.m_tmpArrays[truthOrObservations] = {}
            for field in outputFields:
                self.m_files[truthOrObservations][field]     = open(self.fileName(self.m_outputDir, truthOrObservations, field), 'wb')
                self.m_tmpArrays[truthOrObservations][field] = np.zeros(t_tmpRecordShape(self.m_modWrite, truthOrObservations, field))

    #_________________________

    def initialiseFilterOutput(self, t_filterLabel, t_outputFields, t_tmpRecordShape):
        # open output files and alloc temporary record arrays
        self.m_files[t_filterLabel]     = {}
        self.m_tmpArrays[t_filterLabel] = {}
        for field in t_outputFields:
            self.m_files[t_filterLabel][field]     = open(self.fileName(self.m_outputDir, t_filterLabel, field), 'wb')
            self.m_tmpArrays[t_filterLabel][field] = np.zeros(t_tmpRecordShape(self.m_modWrite, field))

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

    def record(self, t_filterOrTruthOrObservations, t_output, t_value):
        # record output before writing
        try:
            self.m_tmpArrays[t_filterOrTruthOrObservations][t_output][self.m_writingCounter] = t_value
        except:
            pass

    #_________________________

    def write(self):
        # check if necessary to write and call writeAll()
        self.m_writingCounter += 1
        if self.m_writingCounter == self.m_modWrite:
            print('Writing to files')
            self.writeAll()
            self.m_writingCounter = 0

    #_________________________

    def writeAll(self):
        # write all
        for filterOrTruthOrObservations in self.m_files:
            for field in self.m_files[filterOrTruthOrObservations]:
                self.m_tmpArrays[filterOrTruthOrObservations][field][:self.m_writingCounter].tofile(
                        self.m_files[filterOrTruthOrObservations][field] )

    #_________________________

    def finalise(self, t_observationTimes):
        # finalise simulation
        self.writeAll()
        t_observationTimes.observationTimes().tofile(self.fileName(self.m_outputDir, 'observations', 'time'))
        self.closeFiles()

        elapsedTime = str(tm.time()-self.m_timeStart)
        print('Simulation finished')
        print('Total elapsed time = '+elapsedTime)

    #_________________________

    def closeFiles(self):
        # close all files
        for filterOrTruthOrObservations in self.m_files:
            for field in self.m_files[filterOrTruthOrObservations]:
                self.m_files[filterOrTruthOrObservations][field].close()

    #_________________________

    def fileName(t_outputDir, t_label, t_output):
        # file name for field output
        # label can be filter's name or 'truth' or 'observation'
        return ( t_outputDir + t_label + '_' + t_output + '.bin' )
    fileName = staticmethod(fileName)

#__________________________________________________

