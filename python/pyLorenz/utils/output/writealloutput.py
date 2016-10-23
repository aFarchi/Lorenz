#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/output/
# writealloutput.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/23
#__________________________________________________
#
# class to handle output for any simulation
# RMSE, estimations and ensembles are written
#

from defaultouput import DefaultOutput

#__________________________________________________

class WriteAllOutput(DefaultOutput):

    #_________________________

    def __init__(self, t_printCycleTrigger, t_outputDir):
        DefaultOutput.__init__(self, t_printCycleTrigger, t_outputDir)

    #_________________________

    def start(self, t_xDim, t_yDim, t_filters):
        DefaultOutput.start(self, t_xDim, t_yDim, t_filters)

        # open files
        for f in t_filters:
            self.m_fileFilters[f.m_label]['filterForecastEnsemble']     = open(self.m_outputDir+f.m_label+'_forecastEnsemble.bin', 'wb')
            self.m_fileFilters[f.m_label]['filterAnalyseEnsemble']      = open(self.m_outputDir+f.m_label+'_analyseEnsemble.bin', 'wb')

            self.m_tmpArrayFilters[f.m_label]['filterForecastEnsemble'] = np.zeros((self.m_nModWrite, f.m_Ns, t_xDim))
            self.m_tmpArrayFilters[f.m_label]['filterAnalyseEnsemble']  = np.zeros((self.m_nModWrite, f.m_Ns, t_xDim))

    #_________________________

    def finalise(self, t_filters, t_observationTimes):
        DefaultOutput.finalise(self, t_filters, t_observationTimes)

        # close files
        for f in t_filters:
            self.m_fileFilters[f.m_label]['filterForecastEnsemble'].close()
            self.m_fileFilters[f.m_label]['filterAnalyseEnsemble'].close()

    #_________________________

    def writeFilterForecastEnsemble(self, t_filterLabel, t_x):
        self.m_tmpArrayFilters[t_filterLabel]['filterForecastEnsemble'][self.m_counter] = t_x

    #_________________________

    def writeFilterAnalyseEnsemble(self, t_filterLabel, t_x):
        self.m_tmpArrayFilters[t_filterLabel]['filterAnalyseEnsemble'][self.m_counter] = t_x

    #_________________________

    def writeAll(self, t_filters, t_count):
        DefaultOutput.writeAll(self, t_filters, t_count)

        for f in t_filters:
            self.m_tmpArrayFilters[t_filterLabel]['filterForecastEnsemble'][:t_count].tofile(self.m_fileFilters[f.m_label]['filterForecastEnsemble'])
            self.m_tmpArrayFilters[t_filterLabel]['filterAnalyseEnsemble'][:t_count].tofile(self.m_fileFilters[f.m_label]['filterAnalyseEnsemble'])

#__________________________________________________

