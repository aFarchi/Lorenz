#! /usr/bin/env python

#__________________________________________________
# ./
# runbasicdeterministicsimulation.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/20
#__________________________________________________
#
# script to launch a basic deterministic simulation
#

import numpy as np

from pyLorenz.simulation.basicsimulation            import BasicSimulation
from pyLorenz.model.lorenz63                        import DeterministicLorenz63Model
from pyLorenz.utils.random.independantgaussianrng   import IndependantGaussianRNG
from pyLorenz.utils.integration.eulerexplintegrator import DeterministicEulerExplIntegrator
from pyLorenz.utils.integration.rk2integrator       import DeterministicRK2Integrator
from pyLorenz.utils.integration.rk4integrator       import DeterministicRK4Integrator
from pyLorenz.utils.output.basicoutputprinter       import BasicOutputPrinter

# Output directory
outputDir = '/Users/aFarchi/Desktop/test/Lorenz/'

# Number of time steps
Nt = 1000

# Model
model = DeterministicLorenz63Model()
sigma = 10.0
beta  = 8.0 / 3.0
rho   = 28.0
model.setParameters(sigma, beta, rho)

# Integrator
#integrator = DeterministicEulerExplIntegrator()
#integrator = DeterministicRK2Integrator()
integrator = DeterministicRK4Integrator()
dt         = 0.01
integrator.setParameters(dt)

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

# Simulation
simulation = BasicSimulation()
simulation.setParameters(Nt)
simulation.setModel(model)
simulation.setIntegrator(integrator)
simulation.setInitialiser(initialiser)
simulation.setOutputPrinter(outputPrinter)

simulation.run()
simulation.recordToFile(outputDir)

