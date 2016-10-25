#! /usr/bin/env python

#__________________________________________________
# pyLorenz/
# launchSimulation.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/20
#__________________________________________________
#
# Launcher for a simulation
#

from importall    import *
from ConfigParser import SafeConfigParser

# Read configuration
args       = extractArgv()
configFile = args['CONFIG_FILE']
config     = SafeConfigParser()
config.read(configFile)

# Dimensions
xDimension       = config.getint('dimensions', 'state')
yDimension       = config.getint('dimensions', 'observation')

# Initialiser
initialiser      = initialiserFromConfig(config, 'initialisation', xDimension)

# Model
model            = modelFromConfig(config, 'model')

# Integrator
integrator       = integratorFromConfig(config, 'integration', model)

# Observation operator
observation      = observationOperatorFromConfig(config, 'observation', yDimension)
observationTimes = observationTimesFromConfig(config, 'observation')

# Output
output           = outputFromConfig(config, 'output', 'observation', 'integration')

# Simulation
simulation       = Simulation(initialiser, integrator, observation, observationTimes, output)

# Filter
da_filter        = filterFromConfig(config, 'assimilation', 'integration', model, observation)
simulation.addFilter(da_filter)

# run
simulation.run()

# Copy config to output dir
output.copyConfigToOutputDir(config, da_filter.m_label)

