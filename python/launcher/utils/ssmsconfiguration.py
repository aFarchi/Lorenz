#! /usr/bin/env python

#__________________________________________________
# launcher/utils/
# ssmsconfiguration.py
#__________________________________________________
# author        : colonel
# last modified : 2016/11/1
#__________________________________________________
#
# configuration for a single-stage-multi-simulation
#

import numpy as np

from ConfigParser import SafeConfigParser
from itertools    import product
from cPickle      import Pickler

from bash         import createDir
from bash         import tarDir
from bash         import removeDir
from bash         import configFileNamesFromCommand
from bash         import runConfigList
from dictutils    import makeKeyListDict
from dictutils    import listFromDict

#__________________________________________________

def filterClassHierarchy():
    # hierarchy of implemented filter classes
    fch                = {}
    fch['EnF']         = {}
    fch['EnF']['EnKF'] = ['senkf', 'entkf', 'entkfn-dual', 'entkfn-dual-capped']
    fch['EnF']['PF']   = ['sirpf', 'asirpf', 'oisirpf']
    return fch

#__________________________________________________

def transformedFilterClassHierarchy():
    # tranform class hierarchy to make it useful
    # each key is now a filter class
    # and is associated with its inheritance hierarchy
    return makeKeyListDict(filterClassHierarchy())

#__________________________________________________

def variableTruthParameters():
    # the list of parameters for which different simulations are performed
    truthParameters = []
    truthParameters.append(('observation-times', 'dt'))
    truthParameters.append(('observation-operator', 'variance'))
    truthParameters.append(('integration', 'variance'))
    return truthParameters

#__________________________________________________

def outputDir(t_outputTopDir, t_outputLabel, t_truthParameters):
    # output directory for given top directory and truth parameters
    outputDirectory = t_outputTopDir + t_outputLabel + '/'
    for parameter in t_truthParameters:
        outputDirectory += str(parameter).replace('.', 'p') + '/'
    return outputDirectory

#__________________________________________________

def variableFilterParameters():
    # for each filters, the list of parameters for which different simulations are performed
    fp = {}
    for f in ['senkf', 'entkf', 'entkfn-dual', 'entkfn-dual-capped']:
        fp[f] = ['Ns', 'inflation', 'integration_jitter']
    for f in ['sirpf', 'asirpf', 'oisirpf']:
        fp[f] = ['Ns', 'resampling_thr', 'integration_jitter']
    return fp

#__________________________________________________

def label(t_outputLabel, t_filter, t_filterParameters):
    # label for given filter and parameters
    filterLabel = t_outputLabel + '_' + t_filter
    for parameter in t_filterParameters:
        filterLabel += '_' + str(parameter).replace('.', 'p')
    return filterLabel

#__________________________________________________

def configFileName(t_outputTopDir, t_outputLabel, t_filter, t_truthParameters, t_filterParameters):
    # file name for simulation configuration
    return outputDir(t_outputTopDir, t_outputLabel, t_truthParameters) + label(t_outputLabel, t_filter, t_filterParameters) + '.cfg'

#__________________________________________________

class SingleStageMultiSimulationConfiguration(object):

    #_________________________

    def __init__(self):
        # read command
        configFileNames = configFileNamesFromCommand()
        # config parser
        self.m_config   = SafeConfigParser()
        self.m_config.read(configFileNames)
        # filter class hierarchy
        self.m_tfch     = transformedFilterClassHierarchy()
        # variable truth parameters
        self.m_vtp      = variableTruthParameters()
        # variable filter parameters
        self.m_vfp      = variableFilterParameters()

        self.removeDisabledFilters()
        self.checkDeterministicIntegration()

    #_________________________

    def getString(self, t_section, t_option):
        return self.m_config.get(t_section, t_option)

    #_________________________

    def getInt(self, t_section, t_option):
        return int(eval(self.m_config.get(t_section, t_option)))

    #_________________________

    def getFloat(self, t_section, t_option):
        return float(eval(self.m_config.get(t_section, t_option)))

    #_________________________

    def getStringList(self, t_section, t_option):
        s = self.m_config.get(t_section, t_option)
        if s == '':
            return []
        if s[0] == '[':
            s = s[1:]
        if s[-1] == ']':
            s = s[:-1]
        l = s.split(',')
        return [e.strip() for e in l]

    #_________________________

    def getNPArray(self, t_section, t_option):
        a = eval(self.m_config.get(t_section, t_option))
        if isinstance(a, np.ndarray):
            return a
        elif isinstance(a, list):
            return np.array(a)
        else:
            return np.array([a])

    #_________________________

    def configList(self):
        # return the list of config file names contained in self.m_configs
        return listFromDict(self.m_configs)

    #_________________________

    def removeDisabledFilters(self):
        # remove disabled filters from configuration
        # and keep a list containing the enabled filters
        self.m_enabledFilters = []
        for f in self.m_tfch:
            if self.m_config.has_section(f):
                try:
                    if not self.m_config.getboolean(f, 'enable'):
                        self.m_config.remove_section(f)
                    else:
                        self.m_enabledFilters.append(f)
                except:
                    self.m_config.remove_section(f)

    #_________________________

    def checkDeterministicIntegration(self):
        # make sure zero integration variance is used with deterministic integrator
        integration_var = self.getNPArray('integration', 'variance')
        if np.allclose(integration_var, 0.0):
            self.m_config.set('integration', 'class', 'Deterministic')

        if self.getString('integration', 'class') == 'Deterministic':
            self.m_config.set('integration', 'variance', '[0.0]')

        for f in self.m_enabledFilters:

            integration_jit = self.getNPArray(f, 'integration_jitter')
            if np.allclose(integration_jit, 0.0):
                self.m_config.set(f, 'integration_class', 'Deterministic')

            if self.getString(f, 'integration_class') == 'Deterministic':
                self.m_config.set(f, 'integration_jitter', '[0.0]')

    #_________________________

    def makefilterConfiguration(self, t_filter, t_truthParameters, t_filterParameters):
        # make configuration for given filter, truth parameters and filter parameters
        config = SafeConfigParser()

        # fill basic parameters
        for section in ['dimensions', 'random', 'initialisation', 'model', 'integration', 'observation-operator', 'observation-times', 'output', 'truth']:
            config.add_section(section)
            for option in self.m_config.options(section):
                config.set(section, option, self.m_config.get(section, option))

        # output directory and label
        config.set('output', 'directory', outputDir(self.getString('output', 'directory'), self.getString('output', 'label'), t_truthParameters))
        config.set('output', 'label', label(self.getString('output', 'label'), t_filter, t_filterParameters))

        # fill truth parameters
        for ((section, option), value) in zip(self.m_vtp, t_truthParameters):
            config.set(section, option, str(value))

        # fill filter parameters
        config.add_section('assimilation')
        config.set('assimilation', 'filter', t_filter)
        config.set('assimilation', 'label', 'filter')
        for option in self.m_config.options(t_filter):
            config.set('assimilation', option, self.m_config.get(t_filter, option))
        config.remove_option('assimilation', 'enable')

        for (option, value) in zip(self.m_vfp[t_filter], t_filterParameters):
            config.set('assimilation', option, str(value))

        return config

    #_________________________

    def makeFilterConfigurations(self, t_filter, t_truthParameters):
        # make configurations for given filter and truth parameters, 
        # for all possible values for filter parameters
        parameterNames = self.m_vfp[t_filter]
        parameterList  = []
        for option in parameterNames:
            parameterList.append(self.getNPArray(t_filter, option))

        self.m_configs[t_truthParameters][t_filter] = {}

        for filterParameters in product(*parameterList):
            config     = self.makefilterConfiguration(t_filter, t_truthParameters, filterParameters)
            configFN   = configFileName(self.getString('output', 'directory'), self.getString('output', 'label'), t_filter, t_truthParameters, filterParameters)
            configFile = open(configFN, 'w')
            config.write(configFile)
            configFile.close()

            self.m_configs[t_truthParameters][t_filter][filterParameters] = configFN

    #_________________________

    def makeConfigurations(self):
        # make configurations
        print('Making output top-directory')
        createDir(self.getString('output', 'directory'))

        self.m_configs      = {}

        truthParameterList  = []
        for (section, option) in self.m_vtp:
            truthParameterList.append(self.getNPArray(section, option))

        for truthParameters in product(*truthParameterList):

            print('Making output directory')
            createDir(outputDir(self.getString('output', 'directory'), self.getString('output', 'label'), truthParameters))
            self.m_configs[truthParameters] = {}

            for f in self.m_enabledFilters:
                self.makeFilterConfigurations(f, truthParameters)

        configFile = open(self.getString('output', 'directory')+self.getString('output', 'label')+'_configs.bin', 'wb')
        p          = Pickler(configFile, protocol = -1)
        p.dump(self.m_configs)
        configFile.close()

    #_________________________

    def runConfigurations(self):
        # run configurations
        program = self.getString('program', 'launcher')
        nProcs  = self.getInt('program', 'nProcessors')

        runConfigList(program, self.configList(), nProcs)

    #_________________________

    def operatorForExtraction(self):
        # operator for extraction
        operatorName = self.getString('results', 'operator')

        if operatorName == 'mean':
            def operator(t_array):
                return t_array.mean(axis = 0)

        return operator

    #_________________________

    def extractResuts(self):
        # extract results

        outputField    = '_filter_' + self.getString('results', 'field') + '.bin'
        Nt             = self.getInt('observation-times', 'Nt')
        Nt_burnout     = self.getInt('results', 'Nt_burnout')
        operator       = self.operatorForExtraction()

        self.m_results = {}
        for truthParameters in self.m_configs:
            self.m_results[truthParameters] = {}
            for filter in self.m_configs[truthParameters]:
                self.m_results[truthParameters][filter] = {}
                for filterParameters in self.m_configs[truthParameters][filter]:
                    configFileName = self.m_configs[truthParameters][filter][filterParameters]
                    fieldFileName  = configFileName.replace('.cfg', outputField)
                    field          = np.fromfile(fieldFileName)
                    self.m_results[truthParameters][filter][filterParameters] = operator(field.reshape((Nt, field.size/Nt))[Nt_burnout:])

        resultFile = open(self.getString('output', 'directory')+self.getString('output', 'label')+'_results.bin', 'wb')
        p          = Pickler(resultFile, protocol = -1)
        p.dump(self.m_results)
        resultFile.close()

    #_________________________

    def archiveHeavyOutput(self):
        # archive heavy output from simulations
        simDir = outputDir(self.getString('output', 'directory'), self.getString('output', 'label'), [])
        print('Archiving heavy output')
        tarDir(simDir)
        print('Removing archived directory')
        removeDir(simDir)

#__________________________________________________

