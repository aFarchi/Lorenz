#! /usr/bin/env python

#__________________________________________________
# pyLorenz/
# launcher.py
#__________________________________________________
# author        : colonel
# last modified : 2016/11/1
#__________________________________________________
#
# Launcher for a simulation
#

from configuration import Configuration

def main():
    # configuration from file
    configuration = Configuration()
    # simulation
    simulation    = configuration.buildSimulation()
    # run
    simulation.run()

if __name__ == '__main__':
    main()

