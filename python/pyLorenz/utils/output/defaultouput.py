#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/output/
# defaultouput.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/23
#__________________________________________________
#
# class to handle output for any simulation
# RMSE and estimation are writte
#

from onlyrmseoutput import OnlyRMSEOutput

#__________________________________________________

class DefaultOutput(OnlyRMSEOutput):

    #_________________________

    def __init__(self, t_printCycleTrigger, t_outputDir):
        OnlyRMSEOutput.__init__(self, t_printCycleTrigger, t_outputDir)

    #_________________________

    def start(self, t_xDim, t_yDim, t_filters):
        OnlyRMSEOutput.start(self, t_xDim, t_yDim, t_filters)

        # open files
        self.m_fileTruth           = open(self.m_outputDir+'truth.bin', 'wb')
        self.m_fileObservation     = open(self.m_outputDir+'observation.bin', 'wb')

        self.m_tmpArrayTruth       = np.zeros((self.m_nModWrite, t_xDim))
        self.m_tmpArrayObservation = np.zeros((self.m_nModWrite, t_yDim))

        for f in t_filters:
            self.m_fileFilters[f.m_label]['filterForecast']     = open(self.m_outputDir+f.m_label+'_forecast.bin', 'wb')
            self.m_fileFilters[f.m_label]['filterAnalyse']      = open(self.m_outputDir+f.m_label+'_analyse.bin', 'wb')

            self.m_tmpArrayFilters[f.m_label]['filterForecast'] = np.zeros((self.m_nModWrite, t_xDim))
            self.m_tmpArrayFilters[f.m_label]['filterAnalyse']  = np.zeros((self.m_nModWrite, t_xDim))

    #_________________________

    def finalise(self, t_filters, t_observationTimes):
        OnlyRMSEOutput.finalise(self, t_filters, t_observationTimes)

        # close files
        self.m_fileTruth.close()
        self.m_fileObservation.close()
        for f in t_filters:
            self.m_fileFilters[f.m_label]['filterForecast'].close()
            self.m_fileFilters[f.m_label]['filterAnalyse'].close()

    #_________________________

    def writeTruth(self, t_xt, t_xo):
        self.m_tmpArrayTruth[self.m_counter]       = t_xt
        self.m_tmpArrayObservation[self.m_counter] = t_xo

    #_________________________

    def writeFilterForecast(self, t_filterLabel, t_x):
        self.m_tmpArrayFilters[t_filterLabel]['filterForecast'][self.m_counter] = t_x

    #_________________________

    def writeFilterAnalyse(self, t_filterLabel, t_x):
        self.m_tmpArrayFilters[t_filterLabel]['filterAnalyse'][self.m_counter] = t_x

    #_________________________

    def writeAll(self, t_filters, t_count):
        OnlyRMSEOutput.writeAll(self, t_filters, t_count)

        self.m_tmpArrayTruth[:t_count].tofile(self.m_fileTruth)
        self.m_tmpArrayObservation[:t_count].tofile(self.m_fileObservation)

        for f in t_filters:
            self.m_tmpArrayFilters[f.m_label]['filterForecast'][:t_count].tofile(self.m_fileFilters[f.m_label]['filterForecast'])
            self.m_tmpArrayFilters[f.m_label]['filterAnalyse'][:t_count].tofile(self.m_fileFilters[f.m_label]['filterAnalyse'])

#__________________________________________________

