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
from ..auxiliary.bash import moveFile

#__________________________________________________

class Output(object):

    #_________________________

    def __init__(self, t_outputDir, t_modWrite, t_modPrint, t_outputLabel):
        self.setOutputParameters(t_outputDir, t_modWrite, t_modPrint, t_outputLabel)

    #_________________________

    def setOutputParameters(self, t_outputDir, t_modWrite, t_modPrint, t_outputLabel):
        # output dir
        self.m_outputDir   = t_outputDir
        # mod write
        self.m_modWrite    = t_modWrite
        # mod print
        self.m_modPrint    = t_modPrint
        # output label
        self.m_outputLabel = t_outputLabel

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

    def initialiseTruthOutput(self, t_truthOutputFields, t_tmpRecordShape):
        # open output files and alloc temporary record arrays
        self.m_files['truth']     = {}
        self.m_tmpArrays['truth'] = {}
        for field in t_truthOutputFields:
            self.m_files['truth'][field]     = open(self.fileName('truth', field), 'wb')
            self.m_tmpArrays['truth'][field] = np.zeros(t_tmpRecordShape(self.m_modWrite, field))

    #_________________________

    def initialiseFilterOutput(self, t_filterLabel, t_outputFields, t_tmpRecordShape):
        # open output files and alloc temporary record arrays
        self.m_files[t_filterLabel]     = {}
        self.m_tmpArrays[t_filterLabel] = {}
        for field in t_outputFields:
            self.m_files[t_filterLabel][field]     = open(self.fileName(t_filterLabel, field), 'wb')
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

    def record(self, t_filterOrTruth, t_output, t_value):
        # record output before writing
        try:
            self.m_tmpArrays[t_filterOrTruth][t_output][self.m_writingCounter] = t_value
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
        for filterOrTruth in self.m_files:
            for field in self.m_files[filterOrTruth]:
                self.m_tmpArrays[filterOrTruth][field][:self.m_writingCounter].tofile(self.m_files[filterOrTruth][field])

    #_________________________

    def finalise(self, t_crashedFilters):
        # finalise simulation
        self.writeAll()
        self.closeFiles()

        for filter in t_crashedFilters:
            label  = filter.m_label
            fields = filter.m_outputFields
            for field in fields:
                oldFileName = self.fileName(label, field)
                newFileName = oldFileName.replace('.bin', '.bin.crash')
                moveFile(oldFileName, newFileName)

        elapsedTime = str(tm.time()-self.m_timeStart)
        print('Simulation finished')
        print('Total elapsed time = '+elapsedTime)

    #_________________________

    def closeFiles(self):
        # close all files
        for filterOrTruth in self.m_files:
            for field in self.m_files[filterOrTruth]:
                self.m_files[filterOrTruth][field].close()

    #_________________________

    def fileName(self, t_truthOrFilter, t_field):
        # file name for output
        return ( self.m_outputDir + self.m_outputLabel + '_' + t_truthOrFilter + '_' + t_field + '.bin' )

#__________________________________________________

