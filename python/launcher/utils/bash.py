#! /usr/bin/env python

#__________________________________________________
# launcher/utils/
# bash.py
#__________________________________________________
# author        : colonel
# last modified : 2016/11/14
#__________________________________________________
#
# bash utils
# copied from toolbox
#

import sys
from contextlib import contextmanager
from path       import Path

#__________________________________________________

def configFileNamesFromCommand():
    # extract the list of config files from the command used to launch the simulation
    cfn = list(sys.argv)
    cfn.pop(0)
    return cfn

#__________________________________________________

@contextmanager
def workingDirectory(path):
    # execute some code with path as working directory
    pwd = Path.getcwd()
    path.cd()
    yield
    pwd.cd()

#__________________________________________________

