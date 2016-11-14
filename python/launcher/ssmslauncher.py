#!/usr/bin/env python

#__________________________________________________
# launcher/
# ssmslauncher.py
#__________________________________________________
# author        : colonel
# last modified : 2016/11/14
#__________________________________________________
#
# Launcher for a single-stage-multi-simulation
#

from ssmsconfiguration import SingleStageMultiSimulationConfiguration

#__________________________________________________

def main():
    # configuration from file
    configuration = SingleStageMultiSimulationConfiguration()
    # make output directories
    configuration.makeOutputDirs()
    # run simulations
    configuration.runTruthSimulations()
    configuration.runFilterSimulations()
    # archive heavy output
    configuration.archiveHeavyOutput()

#__________________________________________________

if __name__ == '__main__':
    main()

#__________________________________________________

