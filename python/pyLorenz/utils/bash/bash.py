#! /usr/bin/env python

#__________________________________________________
# pyLorenz/utils/bash/
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
    if out == '':
        print(out)

#__________________________________________________

def extractArgv():
    # extract argument used to call launcher
    args            = {}
    args['COMMAND'] = sys.argv.pop(0)

    for arg in sys.argv:
        members          = arg.split('=')
        try:
            args[members[0]] = members[1]
        except:
            args[members[0]] = None

    return args

#__________________________________________________

