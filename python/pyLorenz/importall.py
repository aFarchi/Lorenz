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

def initialiserFromConfig(t_config, t_section, t_stateDimension):
    # build initialiser object from section of config
    initialiser_truth = np.array(eval(t_config.get(t_section, 'truth')))
    initialiser_std   = np.sqrt(t_config.getfloat(t_section, 'variance')) * np.ones(t_stateDimension)
    initialiser_eg    = IndependantGaussianErrorGenerator(initialiser_std)
    return RandomInitialiser(initialiser_truth, initialiser_eg)

#__________________________________________________

def modelFromConfig(t_config, t_section):
    # build model object from section of config
    model_name = t_config.get(t_section, 'name')
    if model_name == 'Lorenz63':
        model_sigma = eval(t_config.get(t_section, 'sigma'))
        model_beta  = eval(t_config.get(t_section, 'beta'))
        model_rho   = eval(t_config.get(t_section, 'rho'))
        return Lorenz63Model(model_sigma, model_beta, model_rho)

#__________________________________________________

def buildIntegrator(t_integrator_cls, t_integrator_stp, t_integrator_dt, t_integrator_var, t_model):
    # build integrator object from parameters

    # error generator
    if t_integrator_var is None or t_integrator_cls == 'Deterministic':
        integrator_eg  = None
    else:
        integrator_var = eval(t_integrator_var)
        if integrator_var > 0.0:
            integrator_std = np.sqrt(eval(t_integrator_var)) * np.ones(t_model.m_spaceDimension)
            integrator_eg  = IndependantGaussianErrorGenerator(integrator_std)
        else:
            integrator_eg  = None

    # integration step
    if t_integrator_stp == 'EulerExpl':
        integrator_stp = EulerExplIntegrationStep(t_integrator_dt, t_model, integrator_eg)
    elif t_integrator_stp == 'KP':
        integrator_stp = KPIntegrationStep(t_integrator_dt, t_model, integrator_eg)
    elif t_integrator_stp == 'RK2':
        integrator_stp = RK2IntegrationStep(t_integrator_dt, t_model, integrator_eg)
    elif t_integrator_stp == 'RK4':
        integrator_stp = RK4IntegrationStep(t_integrator_dt, t_model, integrator_eg)

    # integrator
    if integrator_eg is None:
        return DeterministicIntegrator(integrator_stp)
    elif t_integrator_cls == 'BasicStochastic':
        return BasicStochasticIntegrator(integrator_stp)
    elif t_integrator_cls == 'Stochastic':
        return StochasticIntegrator(integrator_stp)

#__________________________________________________

def integratorFromConfig(t_config, t_section, t_model):
    # build integrator object from section of config
    integrator_dt  = t_config.getfloat(t_section, 'dt')
    integrator_var = t_config.get(t_section, 'variance')
    integrator_stp = t_config.get(t_section, 'step')
    integrator_cls = t_config.get(t_section, 'class')
    return buildIntegrator(integrator_cls, integrator_stp, integrator_dt, integrator_var, t_model)

#__________________________________________________

def observationOperatorFromConfig(t_config, t_section, t_observationDimension):
    # build observation object from section of config
    observation_var = eval(t_config.get(t_section, 'variance'))
    observation_std = np.sqrt(observation_var) * np.ones(t_observationDimension)
    observation_eg  = IndependantGaussianErrorGenerator(observation_std)
    observation_op  = t_config.get(t_section, 'operator')
    if observation_op == 'ObserveAll':
        return ObserveAllOperator(observation_eg)
    elif observation_op == 'ObserveNFirst':
        return ObserveNFirstOperator(observation_eg)

#__________________________________________________

def observationTimesFromConfig(t_config, t_section):
    # build observation times array from config
    observation_dt = t_config.getfloat(t_section, 'dt')
    observation_Nt = t_config.getint(t_section, 'Nt')
    return observation_dt * np.arange(observation_Nt)
#__________________________________________________

def outputFromConfig(t_config, t_section, t_sectionObservation, t_sectionIntegration):
    # build output object from section of config
    obs_dt     = t_config.getfloat(t_sectionObservation, 'dt')
    obs_var    = eval(t_config.get(t_sectionObservation, 'variance'))
    int_var    = t_config.get(t_sectionIntegration, 'variance')
    int_cls    = t_config.get(t_sectionIntegration, 'class')
    if int_cls == 'Deterministic' or int_var is None:
        int_var = 0.0
    else:
        int_var = eval(int_var)
    output_dir = outputSubDir(t_config.get(t_section, 'directory'), obs_dt, obs_var, int_var)
    output_trg = ModulusTrigger(t_config.getint(t_section, 'nPrint'), 0)
    output_nwt = t_config.getint(t_section, 'nWrite')
    output_cls = t_config.get(t_section, 'class')
    if output_cls == 'OnlyRMSE':
        return OnlyRMSEOutput(output_trg, output_dir, output_nwt)
    elif output_cls == 'Default':
        return DefaultOutput(output_trg, output_dir, output_nwt)
    elif output_cls == 'WriteAll':
        return WriteAllOutput(output_trg, output_dir, output_nwt)

#__________________________________________________

def minimiser1DFromConfig(t_config, t_section):
    # build one-dimensional minimiser object from section of config
    min_cls = t_config.get(t_section, 'min1d_cls')

    if min_cls == 'GoldenSection':
        min_maxIt = t_config.getint(t_section, 'min1d_maxIt')
        min_tol   = eval(t_config.get(t_section, 'min1d_tol'))
        return GoldenSectionMinimiser(min_maxIt, min_tol)

    elif min_cls == 'Newton':
        min_dx    = eval(t_config.get(t_section, 'min1d_dx'))
        min_maxIt = t_config.getint(t_section, 'min1d_maxIt')
        min_tol   = eval(t_config.get(t_section, 'min1d_tol'))
        return NewtonMinimiser(min_dx, min_maxIt, min_tol)

#__________________________________________________

def resamplerFromConfig(t_config, t_section):
    # build resampler object from section of config
    res_cls = t_config.get(t_section, 'resampler')

    if res_cls == 'StochasticUniversal':
        return StochasticUniversalResampler()
    elif res_cls == 'Direct':
        return DirectResampler()

#__________________________________________________

def filterFromConfig(t_config, t_section, t_sectionIntegration, t_model, t_observation):
    # build filter object from section of config
    filter_cls  = t_config.get(t_section, 'filter')
    filter_Ns   = t_config.getint(t_section, 'Ns')

    try:
        filter_infl = eval(t_config.get(t_section, 'inflation'))
    except:
        filter_infl = 1.0

    int_dt      = t_config.getfloat(t_sectionIntegration, 'dt')
    int_var     = t_config.get(t_section, 'int_jitter')
    int_stp     = t_config.get(t_sectionIntegration, 'step')
    int_cls     = t_config.get(t_section, 'int_cls')
    filter_int  =  buildIntegrator(int_cls, int_stp, int_dt, int_var, t_model)

    try:
        filter_rt = eval(t_config.get(t_section, 'res_thr'))
    except:
        filter_rt = 0.5

    filter_lbl  = filterLabel(filter_cls, filter_Ns, filter_infl, int_var, filter_rt)

    if filter_cls == 'senkf':
        return StochasticEnKF(filter_lbl, filter_int, t_observation, filter_Ns, filter_infl)

    elif filter_cls == 'entkf':
        U = np.eye(filter_Ns)
        return EnTKF(filter_lbl, filter_int, t_observation, filter_Ns, filter_infl, U)

    elif filter_cls == 'entkfn-dual-capped':
        U         = np.eye(filter_Ns)
        epsilon   = filter_Ns / ( filter_Ns - 1.0 )
        maxZeta   = filter_Ns - 1.0 
        minimiser = minimiserFromConfig(t_config, t_section)
        return EnTKF_N_dual(filter_lbl, filter_int, t_observation, filter_Ns, filter_infl, minimiser, epsilon, maxZeta, U)

    elif filter_cls == 'entkfn-dual':
        U         = np.eye(filter_Ns)
        epsilon   = ( filter_Ns + 1.0 ) / filter_Ns
        maxZeta   = float(filter_Ns)
        minimiser = minimiser1DFromConfig(t_config, t_section)
        return EnTKF_N_dual(filter_lbl, filter_int, t_observation, filter_Ns, filter_infl, minimiser, epsilon, maxZeta, U)

    elif filter_cls == 'sirpf':
        resampler = resamplerFromConfig(t_config, t_section)
        trigger   = ThresholdTrigger(filter_rt)
        return SIRPF(filter_lbl, filter_int, t_observation, filter_Ns, resampler, trigger)

    elif filter_cls == 'asirpf':
        resampler = resamplerFromConfig(t_config, t_section)
        trigger   = ThresholdTrigger(filter_rt)
        return ASIRPF(filter_lbl, filter_int, t_observation, filter_Ns, resampler, trigger)

    elif filter_cls == 'oisirpf':
        resampler = resamplerFromConfig(t_config, t_section)
        trigger   = ThresholdTrigger(filter_rt)
        return OISIRPF(filter_lbl, filter_int, t_observation, filter_Ns, resampler, trigger)

#__________________________________________________

