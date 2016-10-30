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
from subprocess import check_output

#__________________________________________________

def createDir(t_dir):
    # create directory
    command = ['mkdir', '-p', t_dir]
    out     = check_output(command)
    if not out == '':
        print(out)

#__________________________________________________

def configFileNamesFromCommand():
    # extract the list of config files from the command used to launch the simulation
    cfn = list(sys.argv)
    cfn.pop(0)
    return cfn

#__________________________________________________

