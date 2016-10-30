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

from configuration import Configuration

# configuration from file
configuration = Configuration()
# simulation
simulation    = configuration.buildSimulation()
# run
simulation.run()

