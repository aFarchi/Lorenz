#! /usr/bin/env python

#__________________________________________________
# ./
# runbasicdeterministicsimulation.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/21
#__________________________________________________
#
# script to launch a basic deterministic simulation
#

import numpy as np

from pyLorenz.simulation.basicsimulation            import BasicSimulation
from pyLorenz.model.lorenz63                        import DeterministicLorenz63Model
from pyLorenz.utils.random.independantgaussianrng   import IndependantGaussianRNG
from pyLorenz.utils.integration.eulerexplintegrator import DeterministicEulerExplIntegrator
from pyLorenz.utils.integration.kpintegrator        import DeterministicKPIntegrator
from pyLorenz.utils.integration.rk2integrator       import DeterministicRK2Integrator
from pyLorenz.utils.integration.rk4integrator       import DeterministicRK4Integrator
from pyLorenz.utils.output.basicoutputprinter       import BasicOutputPrinter

# Output directory
outputDir = '/Users/aFarchi/Desktop/test/Lorenz/'

# Number of time steps
Nt = 1000

# Model
sigma = 10.0
beta  = 8.0 / 3.0
rho   = 28.0
model = DeterministicLorenz63Model(sigma, beta, rho)

# Integrator
dt         = 0.01
integrator = DeterministicEulerExplIntegrator(dt, model)
#integrator = DeterministicKPIntegrator(dt, model)
#integrator = DeterministicRK2Integrator(dt, model)
#integrator = DeterministicRK4Integrator(dt, model)

# Initialiser
init_m      = np.array([2.0, 3.0, 4.0])
init_v      = np.zeros(3)
initialiser = IndependantGaussianRNG(init_m, init_v)

# Output
op_ntMod      = 100
op_ntFst      = 0
outputPrinter = BasicOutputPrinter(op_ntMod, op_ntFst)

# Simulation
simulation = BasicSimulation(Nt, integrator, initialiser, outputPrinter)

simulation.run()
simulation.recordToFile(outputDir)

