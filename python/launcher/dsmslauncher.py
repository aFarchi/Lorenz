#! /usr/bin/env python

#__________________________________________________
# launcher/
# dsmslauncher.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/24
#__________________________________________________
#
# launch a double-stage multi-simulation
#

import cPickle as pck

from utils.cast                  import *
from utils.bash                  import *
from utils.preparefirststagerun  import *
from utils.preparesecondstagerun import *
from utils.multisimulationconfig import *
from utils.extractanalysermse    import *
from ConfigParser                import SafeConfigParser

#__________________________________________________

# Read configuration
configFileNames = configFileNamesFromCommand()
msConfig        = SafeConfigParser()
msConfig.read(configFileNames)

removeDisabledFilterFromConfig(msConfig)
checkDeterministicIntegration(msConfig)

# Program
program = msConfig.get('program', 'launcher')
nProcs  = int(eval(msConfig.get('program', 'nProcessors')))

# First stage run
fsc = prepareFirstStageRun(msConfig)
runConfigList(program, configDictToConfigList(fsc), nProcs)

# Second stage run
ssc = prepareSecondStageRun(msConfig)
runConfigList(program, configDictToConfigList(ssc), nProcs)

# Extract RMSE
fsBurnOut = int(eval(msConfig.get('first-stage', 'Nt_burnout')))
ntBurnOut = int(eval(msConfig.get('multi-simulation-output', 'Nt_burnout')))
fsRMSE    = extractAnalyseRMSE(fsc, fsBurnOut)
ssRMSE    = extractAnalyseRMSE(ssc, ntBurnOut)

resFile   = open(msConfig.get('output', 'directory')+'results.bin', 'wb')
p         = pck.Pickler(resFile, protocol = -1)
p.dump(fsRMSE)
p.dump(ssRMSE)
resFile.close()

# Archive results
for firstOrSecond in ['first', 'second']:
    fosDir = msConfig.get('output', 'directory') + firstOrSecond + 'Stage/'
    tarDir(fosDir)
    removeDir(fosDir)

#__________________________________________________

