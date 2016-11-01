#! /usr/bin/env python

#__________________________________________________
# launcher/
# ssmslauncher.py
#__________________________________________________
# author        : colonel
# last modified : 2016/11/1
#__________________________________________________
#
# Launcher for a single-stage-multi-simulation
#

from utils.ssmsconfiguration import SingleStageMultiSimulationConfiguration

# configuration from file
configuration = SingleStageMultiSimulationConfiguration()
# make configuration for all simulations
configuration.makeConfigurations()
# run simulations
configuration.runConfigurations()
# extract results
configuration.extractResuts()
# archive results
configuration.archiveHeavyOutput()

