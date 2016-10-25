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

from ..bash.bash import *

#__________________________________________________

class AbstractOutput(object):

    #_________________________

    def __init__(self, t_printCycleTrigger, t_outputDir, t_nModWrite):
        self.setAbstractOutputParameters(t_printCycleTrigger, t_outputDir, t_nModWrite)

    #_________________________

    def setAbstractOutputParameters(self, t_printCycleTrigger, t_outputDir, t_nModWrite):
        # trigger
        self.m_printCycleTrigger = t_printCycleTrigger
        # output dir
        self.m_outputDir         = t_outputDir
        # n mod write
        self.m_nModWrite         = t_nModWrite

    #_________________________

    def createOutputDir(self):
        # create output dir
        createDir(self.m_outputDir)

    #_________________________

    def copyConfigToOutputDir(self, t_config, t_label):
        # copy config file to output dir
        d = self.m_outputDir + t_label + '.cfg'
        f = open(d, 'w')
        t_config.write(f)
        f.close()

    #_________________________

    def start(self, t_xDim, t_yDim, t_filters):
        # start simulation
        self.m_timeStart = tm.time()
        print('Starting simulation')

        # create output dir
        self.createOutputDir()

        # counter
        self.m_counter = 0

    #_________________________

    def initialise(self):
        # initialise simulation
        print('Initialisation')

    #_________________________

    def cycle(self, t_nCycle, t_NCycles):
        # print output for cycle nCycle of NCycles
        if self.m_printCycleTrigger.trigger(0.0, t_nCycle):
            elapsedTime = str(tm.time()-self.m_timeStart)
            try:
                estimatedTimeRemaining = str((tm.time()-self.m_timeStartLoop)*(t_NCycles-t_nCycle)/t_nCycle)
            except:
                self.m_timeStartLoop   = tm.time()
                estimatedTimeRemaining = '***'
            print('Running step # '+str(t_nCycle)+' / '+str(t_NCycles)+' *** et = '+elapsedTime+' *** etr = '+estimatedTimeRemaining)

    #_________________________

    def finalise(self, t_filters, t_observationTimes):
        # finalise simulation
        self.writeAll(t_filters, self.m_counter)
        self.writeTime(t_observationTimes)

        elapsedTime = str(tm.time()-self.m_timeStart)
        print('Simulation finished')
        print('Total elapsed time = '+elapsedTime)

    #_________________________

    def writeTime(self, t_time):
        # write time
        t_time.tofile(self.m_outputDir+'time.bin')

    #_________________________

    def writeTruth(self, t_xt, t_xo):
        # write truth and observation
        raise NotImplementedError

    #_________________________

    def writeFilterForecast(self, t_filterLabel, t_x):
        # write filter forecast
        raise NotImplementedError

    #_________________________

    def writeFilterForecastRMSE(self, t_filterLabel, t_x):
        # write filter forecast RMSE
        raise NotImplementedError

    #_________________________

    def writeFilterForecastEnsemble(self, t_filterLabel, t_x):
        # write filter forecast ensemble
        raise NotImplementedError

    #_________________________

    def writeFilterAnalyse(self, t_filterLabel, t_x):
        # write filter analyse
        raise NotImplementedError

    #_________________________

    def writeFilterAnalyseRMSE(self, t_filterLabel, t_x):
        # write filter analyse RMSE
        raise NotImplementedError

    #_________________________

    def writeFilterAnalyseEnsemble(self, t_filterLabel, t_x):
        # write filter analyse ensemble
        raise NotImplementedError

    #_________________________

    def writeFilterForecastNeff(self, t_filterLabel, t_x):
        # write filter forecast Neff
        raise NotImplementedError

    #_________________________

    def writeFilterAnalseNeff(self, t_filterLabel, t_x):
        # write filter analyse Neff
        raise NotImplementedError

    #_________________________

    def writeFilterAnalseResampled(self, t_filterLabel, t_x):
        # write filter analyse resampled
        raise NotImplementedError

    #_________________________

    def writeAll(self, t_filters, t_count):
        # write tmp to files
        raise NotImplementedError

    #_________________________

    def write(self, t_filters):
        self.m_counter += 1
        if self.m_counter == self.m_nModWrite:
            print('Writing to files')
            self.writeAll(t_filters, self.m_counter)
            self.m_counter = 0

#__________________________________________________

