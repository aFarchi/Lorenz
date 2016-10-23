#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/output/
# onlyrmseoutput.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/23
#__________________________________________________
#
# class to handle output for any simulation
# only RMSE are written
#

import numpy as np

from abstractoutput import AbstractOutput

#__________________________________________________

class OnlyRMSEOutput(AbstractOutput):

    #_________________________

    def __init__(self, t_printCycleTrigger, t_outputDir, t_nModWrite):
        AbstractOutput.__init__(self, t_printCycleTrigger, t_outputDir, t_nModWrite)

    #_________________________

    def start(self, t_xDim, t_yDim, t_filters):
        AbstractOutput.start(self, t_xDim, t_yDim, t_filters)

        # open files
        self.m_fileFilters = {}
        for f in t_filters:
            self.m_fileFilters[f.m_label]                           = {}
            self.m_fileFilters[f.m_label]['filterForecastRMSE']     = open(self.m_outputDir+f.m_label+'_forecastRMSE.bin', 'wb')
            self.m_fileFilters[f.m_label]['filterAnalyseRMSE']      = open(self.m_outputDir+f.m_label+'_analyseRMSE.bin', 'wb')
            self.m_fileFilters[f.m_label]['filterForecastNeff']     = open(self.m_outputDir+f.m_label+'_forecastNeff.bin', 'wb')
            self.m_fileFilters[f.m_label]['filterAnalyseNeff']      = open(self.m_outputDir+f.m_label+'_analyseNeff.bin', 'wb')
            self.m_fileFilters[f.m_label]['filterAnalyseResampled'] = open(self.m_outputDir+f.m_label+'_analyseResampled.bin', 'wb')

        # temporary arrays
        self.m_tmpArrayFilters = {}
        for f in t_filters:
            self.m_tmpArrayFilters[f.m_label]                           = {}
            self.m_tmpArrayFilters[f.m_label]['filterForecastRMSE']     = np.zeros(self.m_nModWrite)
            self.m_tmpArrayFilters[f.m_label]['filterAnalyseRMSE']      = np.zeros(self.m_nModWrite)
            self.m_tmpArrayFilters[f.m_label]['filterForecastNeff']     = np.zeros(self.m_nModWrite)
            self.m_tmpArrayFilters[f.m_label]['filterAnalyseNeff']      = np.zeros(self.m_nModWrite)
            self.m_tmpArrayFilters[f.m_label]['filterAnalyseResampled'] = np.zeros(self.m_nModWrite)

    #_________________________

    def finalise(self, t_filters, t_observationTimes):
        AbstractOutput.finalise(self, t_filters, t_observationTimes)

        # close files
        for f in t_filters:
            self.m_fileFilters[f.m_label]['filterForecastRMSE'].close()
            self.m_fileFilters[f.m_label]['filterAnalyseRMSE'].close()
            self.m_fileFilters[f.m_label]['filterForecastNeff'].close()
            self.m_fileFilters[f.m_label]['filterAnalyseNeff'].close()
            self.m_fileFilters[f.m_label]['filterAnalyseResampled'].close()

    #_________________________

    def writeTruth(self, t_xt, t_xo):
        pass

    #_________________________

    def writeFilterForecast(self, t_filterLabel, t_x):
        pass

    #_________________________

    def writeFilterForecastRMSE(self, t_filterLabel, t_x):
        self.m_tmpArrayFilters[t_filterLabel]['filterForecastRMSE'][self.m_counter] = t_x

    #_________________________

    def writeFilterForecastEnsemble(self, t_filterLabel, t_x):
        pass

    #_________________________

    def writeFilterAnalyse(self, t_filterLabel, t_x):
        pass

    #_________________________

    def writeFilterAnalyseRMSE(self, t_filterLabel, t_x):
        self.m_tmpArrayFilters[t_filterLabel]['filterAnalyseRMSE'][self.m_counter] = t_x

    #_________________________

    def writeFilterAnalyseEnsemble(self, t_filterLabel, t_x):
        pass

    #_________________________

    def writeFilterForecastNeff(self, t_filterLabel, t_x):
        self.m_tmpArrayFilters[t_filterLabel]['filterForecastNeff'][self.m_counter] = t_x

    #_________________________

    def writeFilterAnalyseNeff(self, t_filterLabel, t_x):
        self.m_tmpArrayFilters[t_filterLabel]['filterAnalyseNeff'][self.m_counter] = t_x

    #_________________________

    def writeFilterAnalyseResampled(self, t_filterLabel, t_x):
        self.m_tmpArrayFilters[t_filterLabel]['filterAnalyseResampled'][self.m_counter] = t_x

    #_________________________

    def writeAll(self, t_filters, t_count):
        for f in t_filters:
            self.m_tmpArrayFilters[f.m_label]['filterForecastRMSE'][:t_count].tofile(self.m_fileFilters[f.m_label]['filterForecastRMSE'])
            self.m_tmpArrayFilters[f.m_label]['filterAnalyseRMSE'][:t_count].tofile(self.m_fileFilters[f.m_label]['filterAnalyseRMSE'])
            self.m_tmpArrayFilters[f.m_label]['filterForecastNeff'][:t_count].tofile(self.m_fileFilters[f.m_label]['filterForecastNeff'])
            self.m_tmpArrayFilters[f.m_label]['filterAnalyseNeff'][:t_count].tofile(self.m_fileFilters[f.m_label]['filterAnalyseNeff'])
            self.m_tmpArrayFilters[f.m_label]['filterAnalyseResampled'][:t_count].tofile(self.m_fileFilters[f.m_label]['filterAnalyseResampled'])

#__________________________________________________

