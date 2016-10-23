#! /usr/bin/env python

#__________________________________________________
# pyLorenz/
# importall.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/23
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

from utils.output.defaultouput                       import DefaultOutput
from utils.output.onlyrmseoutput                     import OnlyRMSEOutput

from utils.trigger.thresholdtrigger                  import ThresholdTrigger
from utils.trigger.modulustrigger                    import ModulusTrigger

from utils.minimisation.newtonminimiser              import NewtonMinimiser
from utils.minimisation.goldensectionsearchminimiser import GoldenSectionMinimiser

from model.lorenz63                                  import Lorenz63Model

from observations.observealloperator                 import ObserveAllOperator

from simulation.simulation                           import Simulation

from filters.kalman.stochasticenkf                   import StochasticEnKF
from filters.kalman.entkf                            import EnTKF
from filters.kalman.entkfn                           import EnTKF_N_dual

from filters.pf.sir                                  import SIRPF
from filters.pf.oisir                                import OISIRPF_diag
from filters.pf.asir                                 import ASIRPF
from filters.pf.msir                                 import MSIRPF
from filters.pf.amsir                                import AMSIRPF

#__________________________________________________

def standardOutput(t_outputDir, t_Nt):
    # build a standard output object
    trigger = ModulusTrigger(500, 0)
    return OnlyRMSEOutput(trigger, t_outputDir, t_Nt/2)

#__________________________________________________

def standardLorenz63DeterministicSimulation(t_outputDir, t_observation_Dt, t_observation_Nt, t_observation_variance, t_integrationJitter_variance = None):
    # build a simulation object for a standard Lorenz 63 simulation with no model error

    # Initialiser
    initialiser_truth = np.array([-5.91652, -5.52332, 24.5723])
    initialiser_std   = np.sqrt( 1.0 ) * np.ones(3)
    initialiser_eg    = IndependantGaussianErrorGenerator(initialiser_std)
    initialiser       = RandomInitialiser(initialiser_truth, initialiser_eg)

    # Standard Lorenz 63 Model
    model_sigma = 10.0
    model_beta  = 8.0 / 3.0
    model_rho   = 28.0
    model       = Lorenz63Model(model_sigma, model_beta, model_rho)

    # Integrator without model error
    integrator_dt      = 0.01
    if t_integrationJitter_variance is not None:
        integrator_std = np.sqrt( t_integrationJitter_variance ) * np.ones(3)
        integrator_eg  = IndependantGaussianErrorGenerator(integrator_std)
    else:
        integrator_eg  = None
    integrator_step    = RK4IntegrationStep(integrator_dt, model, integrator_eg)
    integrator         = DeterministicIntegrator(integrator_step)

    # Observation operator
    observation_std = np.sqrt(t_observation_variance) * np.ones(3)
    observation_eg  = IndependantGaussianErrorGenerator(observation_std)
    observation     = ObserveAllOperator(observation_eg)

    # Observation times
    observationTimes = np.arange(t_observation_Nt) * t_observation_Dt

    # Output
    output = standardOutput(t_outputDir, t_observation_Nt)

    return Simulation(initialiser, integrator, observation, observationTimes, output)

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

def standardResamplers():
    # build standard resampler objects

    direct = DirectResampler()
    sus    = StochasticUniversalResampler()

    return [direct, sus]

#__________________________________________________

