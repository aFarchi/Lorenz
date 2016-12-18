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

import numpy   as np
import tarfile as trf

from collections                   import OrderedDict
from path                          import Path
from itertools                     import product, chain
from pickle                        import Pickler
from subprocess                    import run, TimeoutExpired, CalledProcessError
from concurrent.futures            import ThreadPoolExecutor, ProcessPoolExecutor

from utils.bash                    import workingDirectory, configFileNamesFromCommand
from utils.dictutils               import makeKeyDictRecDict, recDictSet
from utils.stringutils             import stringListToString
from utils.configparser            import ConfigParser
from utils.multisimulationoutput   import MultiSimulationOutput
from utils.multisimulationlauncher import MultiSimulationLauncher

#__________________________________________________

def variableTruthParameters():
    # the list of parameters for which different simulations are performed
    truthParameters = []
    truthParameters.append(('integration', 'variance'))
    truthParameters.append(('observation-operator', 'variance'))
    return truthParameters

#__________________________________________________

def variableFilterParameters():
    # for each filters, the list of parameters for which different simulations are performed
    fp = OrderedDict()
    for f in ['StoEnKF', 'ETKF', 'ETKF-N-dual', 'ETKF-N-primal', 'LAStoEnKF', 'CLStoEnKF', 'LETKF', 'LETKF-N-dual', 'LETKF-N-primal']:
        fp[f] = []
        fp[f].append(('ensemble', 'Ns'))
        fp[f].append(('ensemble', 'inflation'))
        fp[f].append(('integration', 'variance'))
    for f in ['LAStoEnKF', 'CLStoEnKF', 'LETKF', 'LETKF-N-dual', 'LETKF-N-primal']:
        fp[f].append(('localisation', 'radius'))
    for f in ['SIR', 'ASIR', 'OISIR']:
        fp[f] = []
        fp[f].append(('ensemble', 'Ns'))
        fp[f].append(('resampling', 'trigger', 'threshold_value'))
        fp[f].append(('integration', 'variance'))
        fp[f].append(('resampling', 'regularisation', 'variance'))
    for f in ['PoterjoysLPF']:
        fp[f] = []
        fp[f].append(('ensemble', 'Ns'))
        fp[f].append(('integration', 'variance'))
        fp[f].append(('localisation', 'radius'))
        fp[f].append(('localisation', 'relaxation'))
    for f in ['PennysLPF']:
        fp[f] = []
        fp[f].append(('ensemble', 'Ns'))
        fp[f].append(('integration', 'variance'))
        fp[f].append(('localisation', 'radius'))
        fp[f].append(('localisation', 'smoothing_strength'))
        fp[f].append(('resampling', 'adaptative_inflation'))
    for f in ['CustomLPF']:
        fp[f] = []
        fp[f].append(('ensemble', 'Ns'))
        fp[f].append(('integration', 'variance'))
        fp[f].append(('localisation', 'radius'))
        fp[f].append(('localisation', 'smoothing_strength'))
    return fp

#__________________________________________________

def outputDir(t_outputTopDir, t_outputLabel, t_dt = None, t_truthParameters = []):
    # output directory for given top directory and truth parameters
    outputDirectory = Path(t_outputTopDir) / t_outputLabel
    if t_dt is None:
        return outputDirectory
    for parameter in chain([t_dt], t_truthParameters):
        outputDirectory = outputDirectory / str(parameter).replace('.', 'p')
    return outputDirectory

#__________________________________________________

def label(t_outputLabel, t_filter, t_filterParameters):
    # label for given filter and parameters
    filterLabel = t_outputLabel + '_' + t_filter
    for parameter in t_filterParameters:
        filterLabel += '_' + str(parameter).replace('.', 'p')
    return filterLabel

#__________________________________________________

def configFileName(t_outputTopDir, t_outputLabel, t_filter, t_dt, t_truthParameters, t_filterParameters):
    # file name for simulation configuration
    return outputDir(t_outputTopDir, t_outputLabel, t_dt, t_truthParameters) / label(t_outputLabel, t_filter, t_filterParameters)+'.cfg'

#__________________________________________________

def fieldOperator(t_name):
    # operator
    if t_name == 'mean':
        def operator(t_array):
            return t_array.mean(axis = 0)
    elif t_name == 'std':
        def operator(t_array):
            return t_array.std(axis = 0)
    return operator

#__________________________________________________

def runSimulation(t_args):
    # run one simulation with the n-th config
    (t_common, t_n, t_config) = t_args

    # command
    tStart  = t_common.m_output.startSimulation(t_common.m_tOrigin, t_n, t_common.m_Nconfigs)
    log     = t_config.replace('.cfg', '.log')
    command = [t_common.m_interpretor, t_common.m_program, t_config]

    with open(log, 'w') as logFile:
        try:
            # run the simulation
            status = run(command, stdout = logFile, stderr = logFile, timeout = t_common.m_timelimit, check = True)
            time   = t_common.m_output.endSimulation(tStart, t_n, t_common.m_Nconfigs)
        except TimeoutExpired:
            # simulation took too much time
            time   = t_common.m_output.timoutSimulation(tStart, t_n, t_common.m_Nconfigs)
            logFile.write('\n')
            logFile.write('______________________________________\n')
            logFile.write('*** Simulation canceled by timeout ***\n')
            logFile.write('*** Time taken = '+str(time)+'\n')
            logFile.write('______________________________________\n')
        except CalledProcessError:
            # an unexpected error occured
            time   = t_common.m_output.unexpectedCrashSimulation(tStart, t_n, t_common.m_Nconfigs)
            logFile.write('\n')
            logFile.write('___________________________________\n')
            logFile.write('*** Simulation probably crashed ***\n')
            logFile.write('*** Time taken = '+str(time)+'\n')
            logFile.write('___________________________________\n')

    if t_common.m_aggregateResults:
        # aggregates results
        try:
            # find file name
            (parent, name) = Path(t_config).abspath().splitpath()
            fieldFileName  = parent.files(name.replace('.cfg', '_*'+t_common.m_outputField))
            if not fieldFileName:
                raise CrashedError
            fieldFileName  = fieldFileName[0]
            # get data
            field          = np.fromfile(fieldFileName)
            size           = field.size
            if np.mod(size, t_common.m_Nt) or not size:
                raise CrashedError
            dimension      = size // t_common.m_Nt
            field          = field.reshape((t_common.m_Nt, dimension))[t_common.m_Nt_relax:]
            # array for results
            # note: results[-1] will hold time and results[-2] will hold a boolean which is true iff the simulation has crashed
            if dimension > 1:
                results    = np.zeros((len(t_common.m_operators)+2, dimension))
            else:
                results    = np.zeros(len(t_common.m_operators)+2)
            # apply operators
            for (i, operator) in enumerate(t_common.m_operators):
                results[i] = fieldOperator(operator)(field)

        except CrashedError:
            # fill result with the crashed value
            results        = t_common.m_crash_value * np.ones(len(t_common.m_operators)+2)
            results[-2]    = 1.0

        # save time
        results[-1]        = time

    else:
        results            = time

    # return aggregated results
    return results

#__________________________________________________

class CrashedError(Exception):
    # Exception raised when a simulation crashed
    pass

#__________________________________________________

class Common(object):
    # empty class
    # used to contain value that will be passed to child 'processes' or 'threads' as function argument
    pass

#__________________________________________________

class SingleStageMultiSimulationConfiguration(object):

    #_________________________

    def __init__(self):
        # config parser
        self.m_config   = ConfigParser(t_commentChar = '#', t_referenceChar = '%')
        self.m_config.readfiles(configFileNamesFromCommand())
        # list of filters
        self.buildListOfFilters()
        # output
        self.buildOutput()
        # launcher
        self.buildMultiSimulationLauncher()

    #_________________________

    def buildListOfFilters(self):
        # build the list of filters
        sections       = self.m_config.options()
        self.m_filters = [section for section in sections if not section in ['launcher', 'common', 'observation-times', 'output', 'truth', 'results']]

    #_________________________

    def buildMultiSimulationLauncher(self):
        # build multi process launcher
        executor    = self.m_config.get('launcher', 'executor')
        if executor == 'ProcessPool':
            executor = ProcessPoolExecutor
        elif executor == 'ThreadPool':
            executor = ThreadPoolExecutor
        max_workers = self.m_config.getInt('launcher', 'max_workers')
        chunksize   = self.m_config.getInt('launcher', 'chunksize')

        self.m_msLauncher = MultiSimulationLauncher(executor, max_workers, chunksize, self.m_output)

    #_________________________

    def buildOutput(self):
        # output
        self.m_output = MultiSimulationOutput()

    #_________________________

    def runConfigurations(self, t_configDict, t_aggregateResults = True):
        # run configurations
        tconfigDict = makeKeyDictRecDict(t_configDict)
        configList  = tconfigDict.keys()

        common                    = Common()
        common.m_output           = self.m_output
        common.m_interpretor      = self.m_config.get('launcher', 'interpretor')
        common.m_program          = self.m_config.get('launcher', 'program')
        common.m_timelimit        = self.m_config.getFloat('launcher', 'timelimit')
        common.m_Nconfigs         = len(configList)
        common.m_aggregateResults = t_aggregateResults

        if t_aggregateResults:
            common.m_Nt           = self.m_config.getInt('common', 'observation-times', 'Nt')
            common.m_Nt_relax     = self.m_config.getInt('results', 'Nt_relax')
            common.m_operators    = self.m_config.getStringList('results', 'operators')
            common.m_crash_value  = self.m_config.getFloat('results', 'crash_value')
            common.m_outputField  = self.m_config.get('results', 'field') + '.bin'

        tmpCF      = outputDir(self.m_config.get('output', 'directory'), self.m_config.get('output', 'label')) / 'running_configs.tmp.txt'
        tmpCF.write_lines(configList)
        rawResults = self.m_msLauncher.runSimulations(runSimulation, common, configList)
        tmpCF.remove_p()

        if t_aggregateResults:
            results = OrderedDict()
            for keys, result in zip(tconfigDict.values(), rawResults):
                recDictSet(results, keys, result)
            return results

        return rawResults

    #_________________________

    def makeOutputDirs(self):
        # making all output directories
        outputTopDir = outputDir(self.m_config.get('output', 'directory'), self.m_config.get('output', 'label'))
        self.m_output.makeTopDir(outputTopDir)
        outputTopDir.makedirs_p()

        # loop over all possible parameters
        dtArray            = self.getDtArray()
        truthParameterList = self.getTruthParameterList()

        for dt in dtArray:
            for truthParameters in product(*truthParameterList):
                outputSubDir = outputDir(self.m_config.get('output', 'directory'), self.m_config.get('output', 'label'), dt, truthParameters)
                self.m_output.makeSubDir(outputSubDir)
                outputSubDir.makedirs_p()

    #_________________________

    def appendTrajectoryAndObservationsToTruthOutputFields(self):
        # make sure 'trajectory' and 'observations' are in truth output fields
        outputFields = self.m_config.getStringList('common', 'truth', 'output', 'fields')
        outputFields = stringListToString(list(set(outputFields+['trajectory', 'observations'])))
        self.m_config.set('common', 'truth', 'output', 'fields', outputFields)

    #_________________________

    def switchTruthToSimulation(self):
        # make sure truth will be simulated
        self.m_config.set('common', 'truth', 'class', 'Simulated')
        self.appendTrajectoryAndObservationsToTruthOutputFields()

    #_________________________

    def removeTrajectoryAndObservationsFromTruthOutputFields(self):
        # remove 'trajectory' and 'observations' from truth output fields
        outputFields = self.m_config.getStringList('common', 'truth', 'output', 'fields')
        if 'trajectory' in outputFields:
            outputFields.remove('trajectory')
        if 'observations' in outputFields:
            outputFields.remove('observations')
        self.m_config.set('common', 'truth', 'output', 'fields', stringListToString(outputFields))

    #_________________________

    def switchTruthToFiles(self):
        # make sur truth will be backed up from files that just have been simulated
        self.m_config.set('common', 'truth', 'class', 'FromFile')
        truthLabel = label(self.m_config.get('output', 'label'), 'truth', [])
        self.m_config.set('common', 'truth', 'files', 'truth', '$output.directory$/'+truthLabel+'_trajectory.bin')
        self.m_config.set('common', 'truth', 'files', 'observations', '$output.directory$/'+truthLabel+'_observations.bin')
        self.removeTrajectoryAndObservationsFromTruthOutputFields()

    #_________________________

    def getDtArray(self):
        # return a numpy array containing all the values for dt
        return self.m_config.getNumpyArray('observation-times', 'dt')

    #_________________________

    def getTruthParameterList(self):
        # return a list of numpy arrays containing all the values for the truth parameters
        return [self.m_config.getNumpyArray('truth', *options) for options in variableTruthParameters()]

    #_________________________

    def getFilterParameterList(self, t_filter):
        # return a list of numpy arrays containing all the values for the truth parameters
        return [self.m_config.getNumpyArray(t_filter, *options) for options in variableFilterParameters()[self.m_config.get('common', t_filter, 'filter')]]

    #_________________________

    def dumpToFile(self, t_rawFileName, t_object):
        # dump objects to file
        with open((outputDir(self.m_config.get('output', 'directory'), self.m_config.get('output', 'label')) / t_rawFileName+'.bin').abspath(), 'wb') as f:
            p = Pickler(f, protocol = -1)
            p.dump(t_object)

    #_________________________

    def cloneCommonConfig(self, t_dt, t_truthParameters, t_filters, t_filtersParameters):
        # clone common configuration and fill parameters
        config        = self.m_config.clone()
        config.m_tree = config.m_tree.m_children['common']

        # remove unnecessary sections
        sections = config.options()
        for section in list(sections):
            if not section in chain(['dimensions', 'model', 'observation-times', 'output', 'truth'], t_filters):
                config.removeOption(section)

        # set dt
        config.set('observation-times', 'dt', str(t_dt))

        # set truth parameters
        for (options, parameter) in zip(variableTruthParameters(), t_truthParameters):
            config.set('truth', *(options+(str(parameter),)))

        # set filter parameters
        for filter in t_filters:
            for (options, parameter) in zip(variableFilterParameters()[self.m_config.get('common', filter, 'filter')], t_filtersParameters[filter]):
                config.set(filter, *(options+(str(parameter),)))

        # set output directory
        config.set('output', 'directory', outputDir(self.m_config.get('output', 'directory'), self.m_config.get('output', 'label'), t_dt, t_truthParameters))

        return config
    #_________________________

    def makeTruthConfigurationForParameters(self, t_dt, t_truthParameters):
        # make configuration for truth simulation
        config = self.cloneCommonConfig(t_dt, t_truthParameters, [], {})
        config.removeOption('truth', 'files')
        config.set('output', 'label', self.m_config.get('output', 'label'))
        return config

    #_________________________

    def makeTruthConfigurations(self):
        # make all configurations for truth simulation
        self.m_truthConfigs = OrderedDict()

        # make sure truth will be simulated
        self.switchTruthToSimulation()

        # loop over all possible parameters
        dtArray            = self.getDtArray()
        truthParameterList = self.getTruthParameterList()

        for dt in dtArray:
            self.m_truthConfigs[dt] = OrderedDict()
            for truthParameters in product(*truthParameterList):

                config   = self.makeTruthConfigurationForParameters(dt, truthParameters)
                configFN = configFileName(self.m_config.get('output', 'directory'), self.m_config.get('output', 'label'), 'truth', dt, truthParameters, [])
                config.tofile(configFN)
                
                self.m_truthConfigs[dt][truthParameters] = configFN

        # for filter simulation, make sure truth is backed up from files
        self.switchTruthToFiles()

        # save truth configs to file
        self.dumpToFile('truth_configs', self.m_truthConfigs)

    #_________________________

    def runTruthSimulations(self):
        # if necessary, run simulations for the truth
        truth_cls = self.m_config.get('truth', 'class')

        if truth_cls == 'SimulatedOnce':
            self.makeTruthConfigurations()
            times = self.runConfigurations(self.m_truthConfigs, False)
            times = np.array([t for t in times])
            self.dumpToFile('truth_times', times)
        else:
            self.m_config.set('common', 'truth', 'class', truth_cls)

    #_________________________

    def makeFilterConfigurationForParameters(self, t_dt, t_truthParameters, t_filter, t_filterParameters):
        # make configuration for filter simulation
        filters           = [t_filter]
        filtersParameters = {t_filter: t_filterParameters}
        config            = self.cloneCommonConfig(t_dt, t_truthParameters, filters, filtersParameters)
        config.set('output', 'label', label(self.m_config.get('output', 'label'), t_filter, t_filterParameters))
        return config

    #_________________________

    def makeFilterConfigurations(self):
        # make all configurations for filter simulation
        self.m_filterConfigs = OrderedDict()

        # loop over all possible parameters
        dtArray            = self.getDtArray()
        truthParameterList = self.getTruthParameterList()

        for dt in dtArray:
            self.m_filterConfigs[dt] = OrderedDict()
            for truthParameters in product(*truthParameterList):
                self.m_filterConfigs[dt][truthParameters] = OrderedDict()

                for filter in self.m_filters:
                    self.m_filterConfigs[dt][truthParameters][filter] = OrderedDict()
                    filterParameterList = self.getFilterParameterList(filter)

                    for filterParameters in product(*filterParameterList):

                        config   = self.makeFilterConfigurationForParameters(dt, truthParameters, filter, filterParameters)
                        configFN = configFileName(self.m_config.get('output', 'directory'), self.m_config.get('output', 'label'), filter, dt, truthParameters, filterParameters)
                        config.tofile(configFN)

                        self.m_filterConfigs[dt][truthParameters][filter][filterParameters] = configFN

        # save truth configs to file
        self.dumpToFile('filter_configs', self.m_filterConfigs)

    #_________________________

    def runFilterSimulations(self):
        # run simulations for the filters

        self.makeFilterConfigurations()
        results = self.runConfigurations(self.m_filterConfigs, True)
        self.dumpToFile('filter_results', results)

    #_________________________

    def archiveHeavyOutput(self):
        # archive heavy output from simulations

        outputTopDir = outputDir(self.m_config.get('output', 'directory'), self.m_config.get('output', 'label'))
        self.m_output.archiveInDir(outputTopDir)

        with workingDirectory(outputTopDir), trf.open('allresults.tar.gz', mode='w:gz') as out:
            # loop over all possible parameters
            dtArray  = self.getDtArray()
            for dt in dtArray:
                outputSubDir = outputDir(self.m_config.get('output', 'directory'), self.m_config.get('output', 'label'), dt, [])
                out.add(outputSubDir.relpath())
                outputSubDir.rmtree_p()

#__________________________________________________

