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
# and build objects from config
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

from utils.output.onlyrmseoutput                     import OnlyRMSEOutput
from utils.output.defaultouput                       import DefaultOutput
from utils.output.fileNames                          import *

from utils.trigger.thresholdtrigger                  import ThresholdTrigger
from utils.trigger.modulustrigger                    import ModulusTrigger

from utils.minimisation.newtonminimiser              import NewtonMinimiser
from utils.minimisation.goldensectionsearchminimiser import GoldenSectionMinimiser

from utils.bash.bash                                 import *

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

def checkDeterministicIntegration(t_config):
    # make sure zero integration variance is used with deterministic integrator
    integration_var = eval(t_config.get('integration', 'variance'))
    if integration_var is None or integration_var == 0.0:
        t_config.set('integration', 'class', 'Deterministic')

    integration_cls = t_config.get('integration', 'class')
    if integration_cls == 'Deterministic':
        t_config.set('integration', 'variance', '0.0')

    integration_jit = eval(t_config.get('assimilation', 'integration_jitter'))
    if integration_jit is None or integration_jit == 0.0:
        t_config.set('assimilation', 'integration_class', 'Deterministic')

    integration_jit_cls = t_config.get('assimilation', 'integration_class')
    if integration_jit_cls == 'Deterministic':
        t_config.set('assimilation', 'integration_jitter', '0.0')

#__________________________________________________

def numpyArrayFromStr(t_str):
    # cast string to numpy array
    array = eval(t_str)
    if isinstance(array, np.ndarray):
        return array
    elif isinstance(array, list):
        return np.array(array)
    else:
        return np.array([array])

#__________________________________________________

def initialiserFromConfig(t_config):
    # build initialiser from config
    initialiser_truth = numpyArrayFromStr(t_config.get('initialisation', 'truth'))
    initialiser_std   = np.sqrt(eval(t_config.get('initialisation', 'variance'))) * np.ones(initialiser_truth.size)
    initialiser_eg    = IndependantGaussianErrorGenerator(initialiser_std)
    return RandomInitialiser(initialiser_truth, initialiser_eg)

#__________________________________________________

def Lorenz63ModelFromConfig(t_config):
    # build Lorenz63 model from config
    model_sigma = eval(t_config.get('model', 'sigma'))
    model_beta  = eval(t_config.get('model', 'beta'))
    model_rho   = eval(t_config.get('model', 'rho'))
    return Lorenz63Model(model_sigma, model_beta, model_rho)

#__________________________________________________

def modelFromConfig(t_config):
    # build model from config
    model_name = t_config.get('model', 'name')
    if model_name == 'Lorenz63':
        return Lorenz63ModelFromConfig(t_config)

#__________________________________________________

def integratorFromConfig(t_config, t_truthOrFilter, t_model):
    # build integrator from config
    integrator_dt  = eval(t_config.get('integration', 'dt'))
    integrator_stc = t_config.get('integration', 'step')

    if t_truthOrFilter == 'truth':
        integrator_cls = t_config.get('integration', 'class')
        integrator_var = eval(t_config.get('integration', 'variance'))
    else:
        integrator_cls = t_config.get('assimilation', 'integration_class')
        integrator_var = eval(t_config.get('assimilation', 'integration_jitter'))

    # error generator
    if integrator_cls == 'Deterministic':
        integrator_eg  = None
    else:
        integrator_std = np.sqrt(integrator_var) * np.ones(t_model.m_spaceDimension)
        integrator_eg  = IndependantGaussianErrorGenerator(integrator_std)

    # integration step
    if integrator_stc == 'EulerExpl':
        integrator_stp = EulerExplIntegrationStep(integrator_dt, t_model, integrator_eg)
    elif integrator_stc == 'KP':
        integrator_stp = KPIntegrationStep(integrator_dt, t_model, integrator_eg)
    elif integrator_stc == 'RK2':
        integrator_stp = RK2IntegrationStep(integrator_dt, t_model, integrator_eg)
    elif integrator_stc == 'RK4':
        integrator_stp = RK4IntegrationStep(integrator_dt, t_model, integrator_eg)

    # integrator
    if integrator_cls == 'Deterministic':
        return DeterministicIntegrator(integrator_stp)
    elif integrator_cls == 'BasicStochastic':
        return BasicStochasticIntegrator(integrator_stp)
    elif integrator_cls == 'Stochastic':
        return StochasticIntegrator(integrator_stp)

#__________________________________________________

def observationOperatorFromConfig(t_config, t_observationDimension):
    # build observation from config
    observation_var = eval(t_config.get('observation', 'variance'))
    observation_std = np.sqrt(observation_var) * np.ones(t_observationDimension)
    observation_eg  = IndependantGaussianErrorGenerator(observation_std)
    observation_op  = t_config.get('observation', 'operator')
    if observation_op == 'ObserveAll':
        return ObserveAllOperator(observation_eg)
    elif observation_op == 'ObserveNFirst':
        return ObserveNFirstOperator(observation_eg)

#__________________________________________________

def observationTimesFromConfig(t_config):
    # build observation times array from config
    observation_dt = eval(t_config.get('observation', 'dt'))
    observation_Nt = int(eval(t_config.get('observation', 'Nt')))
    return observation_dt * np.arange(observation_Nt)

#__________________________________________________

def outputFromConfig(t_config):
    # build output from config
    output_dir = outputSubDir(t_config)
    output_nwt = int(eval(t_config.get('output', 'nWrite')))
    output_trg = ModulusTrigger(int(eval(t_config.get('output', 'nPrint'))), 0)
    output_cls = t_config.get('output', 'class')

    if output_cls == 'OnlyRMSE':
        return OnlyRMSEOutput(output_trg, output_dir, output_nwt)
    elif output_cls == 'Default':
        return DefaultOutput(output_trg, output_dir, output_nwt)
    elif output_cls == 'WriteAll':
        return WriteAllOutput(output_trg, output_dir, output_nwt)

#__________________________________________________

def GoldenSectionMinimiserFromConfig(t_config):
    # build one-dimensional golden section minimiser from config
    minimiser_maxIt = int(eval(t_config.get('assimilation', 'min1d_maxIt')))
    minimiser_tol   = eval(t_config.get('assimilation', 'min1d_tol'))
    return GoldenSectionMinimiser(minimiser_maxIt, minimiser_tol)

#__________________________________________________

def NewtonMinimiserFromConfig(t_config):
    # build one-dimensional newton minimiser from config
    minimiser_dx    = eval(t_config.get('assimilation', 'min1d_dx'))
    minimiser_maxIt = int(eval(t_config.get('assimilation', 'min1d_maxIt')))
    minimiser_tol   = eval(t_config.get('assimilation', 'min1d_tol'))
    return NewtonMinimiser(minimiser_dx, minimiser_maxIt, minimiser_tol)

#__________________________________________________

def minimiser1DFromConfig(t_config):
    # build one-dimensional minimiser from config
    minimiser_cls = t_config.get('assimilation', 'min1d_class')

    # minimiser have different 'builder'-functions since they can have different parameters
    if minimiser_cls == 'GoldenSection':
        return GoldenSectionMinimiserFromConfig(t_config)
    elif minimiser_cls == 'Newton':
        return NewtonMinimiserFromConfig(t_config)

#__________________________________________________

def resamplerFromConfig(t_config):
    # build resampler from config
    resampler_cls = t_config.get('assimilation', 'resampler')

    if resampler_cls == 'StochasticUniversal':
        return StochasticUniversalResampler()
    elif resampler_cls == 'Direct':
        return DirectResampler()

#__________________________________________________

def EnKFFromConfig(t_config, t_model, t_observation):
    # build EnKF from config
    filter_cls = t_config.get('assimilation', 'filter')
    filter_Ns  = int(eval(t_config.get('assimilation', 'Ns')))
    filter_ifl = eval(t_config.get('assimilation', 'inflation'))
    filter_int = integratorFromConfig(t_config, 'filter', t_model)
    filter_lbl = EnKFLabel(t_config)

    if filter_cls == 'senkf':
        return StochasticEnKF(filter_lbl, filter_int, t_observation, filter_Ns, filter_ifl)

    elif filter_cls == 'entkf':
        U = np.eye(filter_Ns)
        return EnTKF(filter_lbl, filter_int, t_observation, filter_Ns, filter_ifl, U)

    elif filter_cls == 'entkfn-dual-capped':
        U         = np.eye(filter_Ns)
        epsilon   = filter_Ns / ( filter_Ns - 1.0 )
        maxZeta   = filter_Ns - 1.0 
        minimiser = minimiser1DFromConfig(t_config)
        return EnTKF_N_dual(filter_lbl, filter_int, t_observation, filter_Ns, filter_ifl, minimiser, epsilon, maxZeta, U)

    elif filter_cls == 'entkfn-dual':
        U         = np.eye(filter_Ns)
        epsilon   = ( filter_Ns + 1.0 ) / filter_Ns
        maxZeta   = float(filter_Ns)
        minimiser = minimiser1DFromConfig(t_config)
        return EnTKF_N_dual(filter_lbl, filter_int, t_observation, filter_Ns, filter_ifl, minimiser, epsilon, maxZeta, U)

#__________________________________________________

def PFFromConfig(t_config, t_model, t_observation):
    # build PF from config
    filter_cls = t_config.get('assimilation', 'filter')
    filter_Ns  = int(eval(t_config.get('assimilation', 'Ns')))
    filter_rt  = eval(t_config.get('assimilation', 'resampling_thr'))
    filter_int = integratorFromConfig(t_config, 'filter', t_model)
    filter_lbl = PFLabel(t_config)

    resampler  = resamplerFromConfig(t_config)
    trigger    = ThresholdTrigger(filter_rt)

    if filter_cls == 'sirpf':
        return SIRPF(filter_lbl, filter_int, t_observation, filter_Ns, resampler, trigger)

    elif filter_cls == 'asirpf':
        return ASIRPF(filter_lbl, filter_int, t_observation, filter_Ns, resampler, trigger)

    elif filter_cls == 'oisirpf':
        return OISIRPF_diag(filter_lbl, filter_int, t_observation, filter_Ns, resampler, trigger)

#__________________________________________________

def filterFromConfig(t_config, t_model, t_observation):
    # build filter from config
    filter_cls = t_config.get('assimilation', 'filter')

    # filters have different 'builder'-functions since they can have different parameters
    if 'en' in filter_cls and 'kf' in filter_cls:
        return EnKFFromConfig(t_config, t_model, t_observation)
    elif 'pf' in filter_cls:
        return PFFromConfig(t_config, t_model, t_observation)

#__________________________________________________

def simulationFromConfig(t_config):
    # build simulation from config

    checkDeterministicIntegration(t_config)

    # dimensions
    xDimension       = int(eval(t_config.get('dimensions', 'state')))
    yDimension       = int(eval(t_config.get('dimensions', 'observation')))

    # initialiser
    initialiser      = initialiserFromConfig(t_config)

    # model
    model            = modelFromConfig(t_config)

    # integrator
    integrator       = integratorFromConfig(t_config, 'truth', model)

    # observation
    observation      = observationOperatorFromConfig(t_config, yDimension)
    observationTimes = observationTimesFromConfig(t_config)

    # output
    output           = outputFromConfig(t_config)

    # filter
    assimilation     = filterFromConfig(t_config, model, observation)

    # simulation
    simulation       = Simulation(initialiser, integrator, observation, observationTimes, output)
    simulation.addFilter(assimilation)

    return simulation

#__________________________________________________

