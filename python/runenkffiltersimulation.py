#! /usr/bin/env python

#__________________________________________________
# ./
# runenkffiltersimulation.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/22
#__________________________________________________
#
# script to launch an EnKF filtering simulation
#

import numpy as np

from pyLorenz.simulation.filtersimulation                  import FilterSimulation
from pyLorenz.model.lorenz63                               import DeterministicLorenz63Model
from pyLorenz.model.lorenz63                               import StochasticLorenz63Model
from pyLorenz.filters.kalman.stochasticenkf                import StochasticEnKF
from pyLorenz.observations.iobservations                   import StochasticIObservations
from pyLorenz.utils.random.independantgaussianrng          import IndependantGaussianRNG
from pyLorenz.utils.resampling.stochasticuniversalsampling import StochasticUniversalResampler
from pyLorenz.utils.integration.eulerexplintegrator        import DeterministicEulerExplIntegrator
from pyLorenz.utils.integration.eulerexplintegrator        import StochasticEulerExplIntegrator
from pyLorenz.utils.integration.rk2integrator              import DeterministicRK2Integrator
from pyLorenz.utils.integration.rk2integrator              import StochasticRK2Integrator
from pyLorenz.utils.integration.rk2integrator              import MultiStochasticRK2Integrator
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
Nt = 1000
# Number of particles
Ns = 10
# Observation times
ntObs = np.arange(Nt)

# Model
sigma = 10.0
beta  = 8.0 / 3.0
rho   = 28.0
#me_m  = np.zeros(3)
#me_s  = np.ones(3)
#me    = IndependantGaussianRNG(me_m, me_s)
model = DeterministicLorenz63Model(sigma, beta, rho)
#model = StochasticLorenz63Model(sigma, beta, rho, me)

# Integrator
dt         = 0.01
ie_m       = np.zeros(3)
ie_s       = 0.1 * np.ones(3)
ie         = IndependantGaussianRNG(ie_m, ie_s)
#integrator = DeterministicEulerExplIntegrator(dt, model)
#integrator = StochasticEulerExplIntegrator(ie, dt, model)
#integrator = DeterministicRK2Integrator(dt, model)
#integrator = StochasticRK2Integrator(ie, dt, model)
#integrator = MultiStochasticRK2Integrator(ie, dt, model)
#integrator = DeterministicRK4Integrator(dt, model)
integrator = StochasticRK4Integrator(ie, dt, model)
#integrator = MultiStochasticRK4Integrator(ie, dt, model)
#integrator = DeterministicKPIntegrator(dt, model)
#integrator = StochasticKPIntegrator(ie, dt, model)
#integrator = MultiStochasticKPIntegrator(ie, dt, model)

# Initialiser
init_m      = np.array([2.0, 3.0, 4.0])
init_v      = np.zeros(3)
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

# Filter
enkf = StochasticEnKF(integrator, observe)

# Simulation
simulation = FilterSimulation(Nt, integrator, initialiser, outputPrinter, Ns, ntObs, enkf, observe)

simulation.run()
simulation.recordToFile(outputDir)

simulation.computeFilterPerformance(0)
simulation.filterPerformanceToFile(outputDir)

