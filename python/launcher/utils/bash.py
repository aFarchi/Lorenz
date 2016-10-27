#! /usr/bin/env python

#__________________________________________________
# launcher/utils/
# bash.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/26
#__________________________________________________
#
# bash utils
#

import os
import sys

from subprocess   import check_output
from subprocess   import call

#__________________________________________________

def createDir(t_dir):
    # create directory
    print('Making directory: '+t_dir)
    command = ['mkdir', '-p', t_dir]
    out     = check_output(command)
    if not out == '':
        print(out)

#__________________________________________________

def removeDir(t_dir):
    # remove directory
    print('Removing directory: '+t_dir)
    command = ['rm', '-rf', t_dir]
    out     = check_output(command)
    if not out == '':
        print(out)

#__________________________________________________

def tarDir(t_dir):
    # archive directory
    print('Archiving directory: '+t_dir)
    if t_dir.endswith('/'):
        t_dir = t_dir[:-1]
    tarFile = t_dir + '.tar.gz'
    command = ['tar', 'zcf', tarFile, t_dir]
    out     = check_output(command)
    if not out == '':
        print (out)

#__________________________________________________

def configFileNamesFromCommand():
    # extract the list of config files from the command used to launch the simulation
    cfn = list(sys.argv)
    cfn.pop(0)
    return cfn

#__________________________________________________

def runOneConfig(t_programm, t_config):
    # run the programm with the given config
    logFN   = t_config.replace('.cfg', '.log')
    command = []
    command.append(t_programm)
    command.append(t_config)

    logFile = open(logFN, 'w')
    status  = call(command, stdout = logFile)
    logFile.close()

    return status

#__________________________________________________

def runConfigList(t_programm, t_configList, t_nProcs):
    # run the programm for each config in configList with nProcs processors
    NTask = 0
    PIDs  = {}
    NC    = len(t_configList)

    # for all configs
    for nc in range(NC):

        # first select the processor to use

        # if all processors are already working
        if NTask == t_nProcs:

            # wait until a processor get available
            (PID_exit, status_exit) = os.wait()

            # find the number of the processors that just got available
            for nt in PIDs:
                if PIDs[nt] == PID_exit:
                    nTask = nt
                    break

        # else just use next processor
        else:
            NTask += 1
            nTask  = NTask

        # PID of the current processor
        PID = os.fork()

        if PID > 0:
            # record PID
            PIDs[nTask] = PID

        if PID == 0:
            # execute task
            print('Running config # '+str(nc+1)+' / '+str(NC))
            sys.exit(runOneConfig(t_programm, t_configList[nc]))

    # wait until all processors have finished
    for i in range(nTask):
        os.wait()

#__________________________________________________

