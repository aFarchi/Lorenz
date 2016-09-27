#! /usr/bin/env python

#__________________________________________________
# ./
# runoisirfiltersimulation.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/22
#__________________________________________________
#
# script to launch a OISIR filtering simulation
#

import numpy as np

from pyLorenz.simulation.multifiltersimulation             import MultiFilterSimulation
from pyLorenz.model.lorenz63                               import DeterministicLorenz63Model
from pyLorenz.filters.pf.oisir                             import OISIRPF
from pyLorenz.filters.kalman.stochasticenkf                import StochasticEnKF
from pyLorenz.observations.iobservations                   import StochasticIObservations
from pyLorenz.utils.random.independantgaussianrng          import IndependantGaussianRNG
from pyLorenz.utils.resampling.stochasticuniversalsampling import StochasticUniversalResampler
from pyLorenz.utils.integration.rk4integrator              import DeterministicRK4Integrator
from pyLorenz.utils.integration.rk4integrator              import StochasticRK4Integrator
from pyLorenz.utils.integration.rk4integrator              import MultiStochasticRK4Integrator
from pyLorenz.utils.integration.kpintegrator               import DeterministicKPIntegrator
from pyLorenz.utils.integration.kpintegrator               import StochasticKPIntegrator
from pyLorenz.utils.integration.kpintegrator               import MultiStochasticKPIntegrator
from pyLorenz.utils.output.basicoutputprinter              import BasicOutputPrinter

# Output directory
outputDir = '/Users/aFarchi/Desktop/test/Lorenz/'

# Number of time steps
Nt = 1200
# Number of particles
Ns = 10
# Observation times
ntObs = np.arange(Nt)

# Model
sigma = 10.0
beta  = 8.0 / 3.0
rho   = 28.0
model = DeterministicLorenz63Model(sigma, beta, rho)

# Integrator
dt         = 0.01
ie_m       = np.zeros(3)
ie_s       = 2.0 * dt * np.ones(3)
ie         = IndependantGaussianRNG(ie_m, ie_s)
#integrator = StochasticRK4Integrator(ie, dt, model)
integrator = StochasticKPIntegrator(ie, dt, model)

# Initialiser
init_m      = np.array([-5.91652, -5.52332, 24.5723])
init_v      = 0.1 * np.ones(3)
initialiser = IndependantGaussianRNG(init_m, init_v)

# Output
op_ntMod      = 100
op_ntFst      = 0
outputPrinter = BasicOutputPrinter(op_ntMod, op_ntFst)

# Observation operator
oe_m    = np.zeros(3)
oe_s    = 0.1 * np.ones(3)
oe      = IndependantGaussianRNG(oe_m, oe_s)
observe = StochasticIObservations(oe)

# Resampler
resampler = StochasticUniversalResampler()

# OI Filter
oVarInflation = 1.0
resThreshold  = 0.3
oisir         = OISIRPF(integrator, observe, resampler, oVarInflation, resThreshold)

# Kalman filter
enkf = StochasticEnKF(integrator, observe)

# Simulation
simulation = MultiFilterSimulation(Nt, integrator, initialiser, outputPrinter, Ns, ntObs, observe)
simulation.addFilter(oisir)
simulation.addFilter(enkf)

simulation.run()
simulation.recordToFile(outputDir)

simulation.computeFilterPerformance(200)
simulation.filterPerformanceToFile(outputDir)

