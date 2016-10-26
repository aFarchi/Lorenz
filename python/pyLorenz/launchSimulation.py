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
configFileNames = configFileNamesFromCommand()
config          = SafeConfigParser()
config.read(configFileNames)

# Simulation
simulation = simulationFromConfig(config)
simulation.run()

