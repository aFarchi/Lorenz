#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/output/
# filenames.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/23
#__________________________________________________
#
# fonctions used to determine file names from config
#

from ..auxiliary.dictutils import dictElement
from ..auxiliary.dictutils import makeKeyListDict

#__________________________________________________

class SimulationOutputs(object):

    #_________________________

    def __init__(self):
        # transformed filter class hierarchy
        self.m_fch = transformedFilterClassHierarchy()
        # filter possible outputs
        self.m_fpo = filtersPossibleOutputs()

    #_________________________

    def filterPossibleOutputs(self, t_filter):
        # return the list of possible outputs for filter
        try:
            return dictElement(self.m_fpo, self.m_fch[t_filter])
        except:
            return []

    #_________________________

    def truthOrObservationsPossibleOutputs(self, t_truthOrObservations):
        if t_truthOrObservations == 'truth':
            return ['trajectory']
        elif t_truthOrObservations == 'observations':
            return ['trajectory']

    #_________________________

    def EnKFLabel(self, t_flavor, t_Ns, t_inflation, t_jitter):
        return ( t_flavor +
                '_' + str(t_Ns) +
                '_' + str(t_inflation).replace('.', 'p') +
                '_' + str(t_jitter).replace('.', 'p') )

    #_________________________

    def PFLabel(self, t_flavor, t_Ns, t_resampling_thr, t_jitter):
        return ( t_flavor +
                '_' + str(t_Ns) +
                '_' + str(t_resampling_thr).replace('.', 'p') +
                '_' + str(t_jitter).replace('.', 'p') )

#__________________________________________________

    """
    def truthOrObservationsTmpRecordShape(self, t_truthOrObservations, t_field, t_nRecord, t_xDim, t_yDim):
        if t_field == 'trajectory':
            if t_truthOrObservations == 'truth':
                return (t_modWrite, t_xDim)
            elif t_truthOrObservations == 'observations':
                return (t_modWrite, t_yDim)
    """



#__________________________________________________

def filterClassHierarchy():
    # hierarchy of implemented filter classes
    fch                = {}
    fch['EnF']         = {}
    fch['EnF']['EnKF'] = ['senkf', 'entkf', 'entkfn-dual', 'entkfn-dual-capped']
    fch['EnF']['PF']   = ['sir', 'asir', 'oisir']
    return fch

#__________________________________________________

def transformedFilterClassHierarchy():
    # tranform class hierarchy to make it useful
    # each key is now a filter class
    # and is associated with its inheritance hierarchy
    return makeKeyListDict(filterClassHierarchy())

#__________________________________________________

def availableOutputs():
    # availlable outputs for the hierarchy of implemented classes
    fch                = filterClassHierarchy()
    ao                 = {}
    ao['EnF']          = {}
    ao['EnF']['EnKF']  = {}
    ao['EnF']['PF']    = {}
    for enkf in fch['EnF']['EnKF']:
        ao['EnF']['EnKF'][enkf] = ['forecastRMSE', 'forecastTrajectory', 'analyseRMSE', 'analyseTrajectory']
    for pf in fch['EnF']['PF']:
        ao['EnF']['PF'][pf]     = ['forecastRMSE', 'forecastTrajectory', 'forecastNeff', 'analyseRMSE', 'analyseTrajectory', 'analyseNeff', 'analyseResampled']
    ao['truth']        = ['trajectory']
    ao['observations'] = ['trajectory']
    return ao

#__________________________________________________

"""
def fileName(t_outputDir, t_label, t_output):
    # file name for field output
    # label can be filter's name or 'truth' or 'observation'
    return ( t_outputDir + t_label + '_' + t_output + '.bin' )

#__________________________________________________

def outputSubDir(t_config):
    # output sub directory
    outputDir = t_config.get('output', 'directory')
    obs_dt    = str(eval(t_config.get('observation', 'dt'))).replace('.', 'p')
    obs_var   = str(eval(t_config.get('observation', 'variance'))).replace('.', 'p')
    int_var   = str(eval(t_config.get('integration', 'variance'))).replace('.', 'p')
    return ( outputDir + obs_dt + '/' + obs_var + '/' + int_var + '/' )

#__________________________________________________

def EnKFLabel(t_config):
    # generic EnKF label
    flavor    = t_config.get('assimilation', 'filter')
    Ns        = str(t_config.getint('assimilation', 'Ns'))
    inflation = str(eval(t_config.get('assimilation', 'inflation'))).replace('.', 'p')
    jitter    = str(eval(t_config.get('assimilation', 'integration_jitter'))).replace('.', 'p')
    return ( flavor + '_' + Ns + '_' + inflation + '_' + jitter )

#__________________________________________________

def PFLabel(t_config):
    # generic particle filter label
    flavor  = t_config.get('assimilation', 'filter')
    Ns      = str(t_config.getint('assimilation', 'Ns'))
    res_thr = str(eval(t_config.get('assimilation', 'resampling_thr'))).replace('.', 'p')
    jitter  = str(eval(t_config.get('assimilation', 'integration_jitter'))).replace('.', 'p')
    return ( flavor + '_' + Ns + '_' + res_thr + '_' + jitter )
    
#__________________________________________________

def filterLabel(t_config):
    # generic filter label
    flavor = t_config.get('assimilation', 'filter')
    if 'en' in flavor and 'kf' in flavor:
        return EnKFLabel(t_config, 'assimilation')
    elif 'pf' in flavor:
        return PFLabel(t_config, 'assimilation')

#__________________________________________________
"""
