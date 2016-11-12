#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/auxiliary/
# bash.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/23
#__________________________________________________
#
# bash utils
#

import sys

#__________________________________________________

def configFileNamesFromCommand():
    # extract the list of config files from the command used to launch the simulation
    cfn = list(sys.argv)
    cfn.pop(0)
    return cfn

#__________________________________________________

