#! /usr/bin/env python

#__________________________________________________
# ./
# runsirfiltersimulation.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/20
#__________________________________________________
#
# script to launch a SIR filtering simulation
#

import numpy as np

from pyLorenz.simulation.filtersimulation                  import FilterSimulation
from pyLorenz.model.lorenz63                               import DeterministicLorenz63Model
from pyLorenz.model.lorenz63                               import StochasticLorenz63Model
from pyLorenz.filters.pf.sir                               import SIRPF
from pyLorenz.observations.iobservations                   import StochasticIObservations
from pyLorenz.utils.random.independantgaussianrng          import IndependantGaussianRNG
from pyLorenz.utils.resampling.stochasticuniversalsampling import StochasticUniversalResampler
from pyLorenz.utils.integration.eulerexplintegrator        import DeterministicEulerExplIntegrator
from pyLorenz.utils.integration.eulerexplintegrator        import StochasticEulerExplIntegrator
from pyLorenz.utils.integration.eulerexplintegrator        import MultiStochasticEulerExplIntegrator
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
Nt = 200
# Number of particles
Ns = 100
# Observation parameters
ntModObs = 1
ntFstObs = 0

# Model
model = DeterministicLorenz63Model()
#model = StochasticLorenz63Model()
sigma = 10.0
beta  = 8.0 / 3.0
rho   = 28.0
#me    = IndependantGaussianRNG()
#me_m  = np.zeros(3)
#me_s  = np.ones(3)
#me.setParameters(me_m, me_s)
model.setParameters(sigma, beta, rho)
#model.setErrorGenerator(me)

# Integrator
#integrator = DeterministicEulerExplIntegrator()
#integrator = StochasticEulerExplIntegrator()
#integrator = MultiStochasticEulerExplIntegrator()
#integrator = DeterministicRK2Integrator()
#integrator = StochasticRK2Integrator()
#integrator = MultiStochasticRK2Integrator()
#integrator = DeterministicRK4Integrator()
integrator = StochasticRK4Integrator()
#integrator = MultiStochasticRK4Integrator()
#integrator = DeterministicKPIntegrator()
#integrator = StochasticKPIntegrator()
#integrator = MultiStochasticKPIntegrator()
dt         = 0.01
ie         = IndependantGaussianRNG()
ie_m       = np.zeros(3)
ie_s       = 0.1 * np.ones(3)
ie.setParameters(ie_m, ie_s)
integrator.setParameters(dt)
integrator.setErrorGenerator(ie)

# Resampler
resampler = StochasticUniversalResampler()

# Observation operator
observe = StochasticIObservations()
oe      = IndependantGaussianRNG()
oe_m    = np.zeros(3)
oe_s    = 0.1 * np.ones(3)
oe.setParameters(oe_m, oe_s)
observe.setErrorGenerator(oe)

# Filter
sirfilter     = SIRPF()
oVarInflation = 50.0
resThreshold  = 0.3
sirfilter.setParameters(oVarInflation, resThreshold)
sirfilter.setModel(model)
sirfilter.setIntegrator(integrator)
sirfilter.setResampler(resampler)
sirfilter.setObservationOperator(observe)

# Initialiser
initialiser = IndependantGaussianRNG()
init_m      = np.array([2.0, 3.0, 4.0])
init_v      = np.ones(3)
initialiser.setParameters(init_m, init_v)

# Output
outputPrinter = BasicOutputPrinter()
op_ntMod      = 100
op_ntFst      = 0
outputPrinter.setParameters(op_ntMod, op_ntFst)

# Simulation
simulation = FilterSimulation()
simulation.setParameters(Nt, Ns, ntModObs, ntFstObs)
simulation.setFilter(sirfilter)
simulation.setModel(model)
simulation.setIntegrator(integrator)
simulation.setInitialiser(initialiser)
simulation.setObservationOperator(observe)
simulation.setOutputPrinter(outputPrinter)

simulation.run()
simulation.recordToFile(outputDir)

