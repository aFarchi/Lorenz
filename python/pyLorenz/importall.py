#! /usr/bin/env python

#__________________________________________________
# pyLorenz/
# importall.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/18
#__________________________________________________
#
# define all the packages to import
#

import numpy as np

from utils.random.independantgaussianerrorgenerator  import IndependantGaussianErrorGenerator
from utils.random.stochasticuniversalresampler       import StochasticUniversalResampler
from utils.random.directresampler                    import DirectResampler

from utils.integration.kpintegrationstep             import KPIntegrationStep
from utils.integration.rk4integrationstep            import RK4IntegrationStep
from utils.integration.integrator                    import BasicStochasticIntegrator
from utils.integration.integrator                    import DeterministicIntegrator

from utils.initialisation.randominitialiser          import RandomInitialiser

from utils.output.basicoutputprinter                 import BasicOutputPrinter

from utils.trigger.thresholdtrigger                  import ThresholdTrigger

from utils.minimisation.newtonminimiser              import NewtonMinimiser
from utils.minimisation.goldensectionsearchminimiser import GoldenSectionMinimiser

from model.lorenz63                                  import Lorenz63Model

from observations.observealloperator                 import ObserveAllOperator

from simulation.multifiltersimulation                import MultiFilterSimulation

from filters.kalman.stochasticenkf                   import StochasticEnKF
from filters.kalman.entkf                            import EnTKF
from filters.kalman.entkfn                           import EnTKF_N

from filters.pf.sir                                  import SIRPF
from filters.pf.oisir                                import OISIRPF_diag
from filters.pf.asir                                 import ASIRPF
from filters.pf.msir                                 import MSIRPF
from filters.pf.amsir                                import AMSIRPF

#__________________________________________________

def standardLorenz63DeterministicSimulation(t_observation_Dt, t_observation_Nt, t_observation_variance):
    # build a simulation object for a standard Lorenz 63 simulation with no model error

    # Standard Lorenz 63 Model
    model_sigma = 10.0
    model_beta  = 8.0 / 3.0
    model_rho   = 28.0
    model       = Lorenz63Model(model_sigma, model_beta, model_rho)

    # Integrator without model error
    integrator_dt   = 0.01
    integrator_step = RK4IntegrationStep(integrator_dt, model)
    integrator      = DeterministicIntegrator(integrator_step)

    # Initialiser
    initialiser_truth = np.array([-5.91652, -5.52332, 24.5723])
    initialiser_std   = np.sqrt( 1.0 ) * np.ones(3)
    initialiser_eg    = IndependantGaussianErrorGenerator(initialiser_std)
    initialiser       = RandomInitialiser(initialiser_truth, initialiser_eg)

    # Output
    output_ntMod = min(max(t_observation_Nt/100, 500), 5000)
    output_ntFst = 0
    output       = BasicOutputPrinter(output_ntMod, output_ntFst)

    # Observation operator
    observation_std = np.sqrt(t_observation_variance) * np.ones(3)
    observation_eg  = IndependantGaussianErrorGenerator(observation_std)
    observation     = ObserveAllOperator(observation_eg)

    # Observation times
    observationTimes = np.arange(t_observation_Nt) * t_observation_Dt

    return MultiFilterSimulation(integrator, initialiser, output, observationTimes, observation)

#__________________________________________________

def standardMinimisers():
    # build standard minimiser objects

    # Newton minimiser
    newton_dx        = 1.0e-5
    newton_maxIt     = 100
    newton_tolerance = 1.0e-8
    newton           = NewtonMinimiser(newton_dx, newton_maxIt, newton_tolerance)

    # Golden section search minimiser
    gss_maxIt     = 100
    gss_tolerance = 1.0e-4
    gss           = GoldenSectionMinimiser(gss_maxIt, gss_tolerance)

    return [newton, gss]

#__________________________________________________

