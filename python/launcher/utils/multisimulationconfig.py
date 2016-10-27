#! /usr/bin/env python

#__________________________________________________
# launcher/utils/
# multisimulationconfig.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/26
#__________________________________________________
#
# functions related to the multi simulation configuration
#

from cast             import *
from simulationoutput import *

#__________________________________________________

def removeDisabledFilterFromConfig(t_config):
    # remove disabled filter from the given configuration
    flist = FilterFlavorList()
    for f in flist:
        if t_config.has_section(f):
            try:
                if not t_config.getboolean(f, 'enable'):
                    t_config.remove_section(f)
            except:
                t_config.remove_section(f)

#__________________________________________________

def availableENKFsFromConfig(t_config):
    # list of ENKFs enabled by config
    flist  = ENKFFlavorList()
    aflist = []
    for f in flist:
        if t_config.has_section(f) and t_config.getboolean(f, 'enable'):
            aflist.append(f)
    return aflist

#__________________________________________________

def availablePFsFromConfig(t_config):
    # list of PF enabled by config
    flist  = PFFlavorList()
    aflist = []
    for f in flist:
        if t_config.has_section(f) and t_config.getboolean(f, 'enable'):
            aflist.append(f)
    return aflist

#__________________________________________________

def availableFiltersFromConfig(t_config):
    # list of filters enabled by config
    flist  = FilterFlavorList()
    aflist = []
    for f in flist:
        if t_config.has_section(f) and t_config.getboolean(f, 'enable'):
            aflist.append(f)
    return aflist

#__________________________________________________

def checkDeterministicIntegration(t_config):
    # make sure zero integration variance is used with deterministic integrator

    integration_var = makeNumpyArray(eval(t_config.get('integration', 'variance')))
    if integration_var.size == 1 and ( integration_var[0] is None or integration_var[0] == 0.0 ):
        t_config.set('integration', 'class', 'Deterministic')

    integration_cls = t_config.get('integration', 'class')
    if integration_cls == 'Deterministic':
        t_config.set('integration', 'variance', '[0.0]')

    for f in availableFiltersFromConfig(t_config):

        integration_jit = makeNumpyArray(eval(t_config.get(f, 'integration_jitter')))
        if integration_jit.size == 1 and ( integration_jit[0] is None or integration_jit[0] == 0.0 ):
            t_config.set(f, 'integration_class', 'Deterministic')

        integration_jit_cls = t_config.get(f, 'integration_class')
        if integration_jit_cls == 'Deterministic':
            t_config.set(f, 'integration_jitter', '[0.0]')

#__________________________________________________

