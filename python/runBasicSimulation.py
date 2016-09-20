#! /usr/bin/env python

#__________________________________________________
# ./
# runBasicSimulation.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/20
#__________________________________________________
#
# script to launch a basic simulation
#

import numpy as np

from pyLorenz.simulation.basicSimulation          import BasicSimulation
from pyLorenz.model.lorenz63                      import Lorenz63Model
from pyLorenz.utils.random.independantGaussianRNG import IndependantGaussianRNG
from pyLorenz.utils.integration.eulerExplScheme   import EulerExplScheme
from pyLorenz.utils.integration.rk2Scheme         import RK2Scheme
from pyLorenz.utils.integration.rk4Scheme         import RK4Scheme
from pyLorenz.utils.output.basicOutputPrinter     import BasicOutputPrinter

# Integrator
#integrator = EulerExplScheme()
integrator = RK4Scheme()
dt         = 0.01
ie         = IndependantGaussianRNG()
ie_m       = np.zeros(3)
ie_v       = np.zeros(3)
ie.setParameters(ie_m, ie_v)
integrator.setParameters(dt)
integrator.setErrorGenerator(ie)

# Model
model = Lorenz63Model()
sigma = 10.0
beta  = 8.0 / 3.0
rho   = 28.0
me    = IndependantGaussianRNG()
me_m  = np.zeros(3)
me_v  = np.zeros(3)
me.setParameters(me_m, me_v)
model.setParameters(sigma, beta, rho)
model.setErrorGenerator(me)
model.setIntegrator(integrator)

# Initialiser
initialiser = IndependantGaussianRNG()
init_m      = np.array([2.0, 3.0, 4.0])
init_v      = np.zeros(3)
initialiser.setParameters(init_m, init_v)

# Output
outputPrinter = BasicOutputPrinter()
op_ntMod      = 100
op_ntFst      = 0
outputPrinter.setParameters(op_ntMod, op_ntFst)

# Number of time steps
Nt = 100000

# Output directory
outputDir = '/Users/aFarchi/Desktop/test/Lorenz/'

simulation = BasicSimulation()
simulation.setModel(model)
simulation.setInitialiser(initialiser)
simulation.setSimulationParameters(Nt)
simulation.setOutputPrinter(outputPrinter)

simulation.run(False)
simulation.recordToFile(outputDir)

