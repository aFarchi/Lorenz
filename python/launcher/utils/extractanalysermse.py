#! /usr/bin/env python

#__________________________________________________
# launcher/utils/
# extractanalysermse.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/26
#__________________________________________________
#
# extract analyse rmse from dict of config file names
#

import numpy as np

#__________________________________________________

def extractOneAnalyseRMSE(t_configFileName, t_ntBurnOut):
    aRMSEFileName = t_configFileName.replace('.cfg', '_analyseRMSE.bin')
    return np.fromfile(aRMSEFileName)[t_ntBurnOut:].mean()

#__________________________________________________

def extractAnalyseRMSE(t_configFileNames, t_ntBurnOut):
    aRMSE = {}
    for truthParameters in t_configFileNames:
        aRMSE[truthParameters] = {}
        for filterParameters in t_configFileNames[truthParameters]:
            aRMSE[truthParameters][filterParameters] = extractOneAnalyseRMSE(t_configFileNames[truthParameters][filterParameters], t_ntBurnOut)
    return aRMSE

#__________________________________________________

