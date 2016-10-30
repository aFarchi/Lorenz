#! /usr/bin/env python

#__________________________________________________
# pyLorenz/
# configuration.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/30
#__________________________________________________
#
# configuration for a simulation
#

import numpy as np

from ConfigParser                                    import SafeConfigParser

from utils.auxiliary.bash                            import configFileNamesFromCommand
from utils.auxiliary.dictutils                       import makeKeyListDict

from utils.random.independantgaussianerrorgenerator  import IndependantGaussianErrorGenerator
from utils.random.stochasticuniversalresampler       import StochasticUniversalResampler
from utils.random.directresampler                    import DirectResampler

from utils.integration.kpintegrationstep             import KPIntegrationStep
from utils.integration.rk4integrationstep            import RK4IntegrationStep
from utils.integration.integrator                    import BasicStochasticIntegrator
from utils.integration.integrator                    import DeterministicIntegrator

from utils.initialisation.randominitialiser          import RandomInitialiser

from utils.output.output                             import Output

from utils.trigger.thresholdtrigger                  import ThresholdTrigger

from utils.minimisation.newtonminimiser              import NewtonMinimiser
from utils.minimisation.goldensectionsearchminimiser import GoldenSectionMinimiser

from model.lorenz63                                  import Lorenz63Model

from observations.observealloperator                 import ObserveAllOperator
from observations.regularobservationtimes            import RegularObservationTimes

from simulation.truth                                import Truth
from simulation.run                                  import Simulation

from filters.kalman.stochasticenkf                   import StochasticEnKF
from filters.kalman.entkf                            import EnTKF
from filters.kalman.entkfn                           import EnTKF_N_dual

from filters.pf.sir                                  import SIRPF
from filters.pf.oisir                                import OISIRPF_diag
from filters.pf.asir                                 import ASIRPF

#__________________________________________________

def filterClassHierarchy():
    # hierarchy of implemented filter classes
    fch                = {}
    fch['EnF']         = {}
    fch['EnF']['EnKF'] = ['senkf', 'entkf', 'entkfn-dual', 'entkfn-dual-capped']
    fch['EnF']['PF']   = ['sir', 'asir', 'oisir']
    return fch

#__________________________________________________

def transformedFilterClassHierarchy():
    # tranform class hierarchy to make it useful
    # each key is now a filter class
    # and is associated with its inheritance hierarchy
    return makeKeyListDict(filterClassHierarchy())

#__________________________________________________

def EnKFLabel(t_flavor, t_Ns, t_inflation, t_jitter):
    return ( t_flavor +
            '_' + str(t_Ns) +
            '_' + str(t_inflation).replace('.', 'p') +
            '_' + str(t_jitter).replace('.', 'p') )

#__________________________________________________

def PFLabel(t_flavor, t_Ns, t_resampling_thr, t_jitter):
    return ( t_flavor +
            '_' + str(t_Ns) +
            '_' + str(t_resampling_thr).replace('.', 'p') +
            '_' + str(t_jitter).replace('.', 'p') )

#__________________________________________________

class Configuration(object):

    #_________________________

    def __init__(self):
        # read command
        configFileNames = configFileNamesFromCommand()
        # config parser
        self.m_config   = SafeConfigParser()
        self.m_config.read(configFileNames)

        self.checkDeterministicIntegration()

    #_________________________

    def getString(self, t_section, t_option):
        return self.m_config.get(t_section, t_option)

    #_________________________

    def getInt(self, t_section, t_option):
        return int(eval(self.m_config.get(t_section, t_option)))

    #_________________________

    def getFloat(self, t_section, t_option):
        return float(eval(self.m_config.get(t_section, t_option)))

    #_________________________

    def getStringList(self, t_section, t_option):
        s = self.m_config.get(t_section, t_option)
        if s == '':
            return []
        if s[0] == '[':
            s = s[1:]
        if s[-1] == ']':
            s = s[:-1]
        l = s.split(',')
        return [e.strip() for e in l]

    #_________________________

    def getNPArray(self, t_section, t_option):
        a = eval(self.m_config.get(t_section, t_option))
        if isinstance(a, np.ndarray):
            return a
        elif isinstance(a, list):
            return np.array(a)
        else:
            return np.array([a])

    #_________________________

    def checkDeterministicIntegration(self):
        # make sure zero integration variance is used with deterministic integrator
        integration_var = self.getFloat('integration', 'variance')
        if integration_var is None or integration_var == 0.0:
            self.m_config.set('integration', 'class', 'Deterministic')

        if self.getString('integration', 'class') == 'Deterministic':
            self.m_config.set('integration', 'variance', '0.0')

        integration_jit = self.getFloat('assimilation', 'integration_jitter')
        if integration_jit is None or integration_jit == 0.0:
            self.m_config.set('assimilation', 'integration_class', 'Deterministic')

        if self.getString('assimilation', 'integration_class') == 'Deterministic':
            self.m_config.set('assimilation', 'integration_jitter', '0.0')

    #_________________________

    def initialiser(self):
        # build initialiser
        initialiser_truth = self.getNPArray('initialisation', 'truth')
        initialiser_std   = np.sqrt(self.getFloat('initialisation', 'variance')) * np.ones(initialiser_truth.size)
        initialiser_eg    = IndependantGaussianErrorGenerator(initialiser_std)
        return RandomInitialiser(initialiser_truth, initialiser_eg)

    #_________________________

    def Lorenz63Model(self):
        # build Lorenz63 model
        model_sigma = self.getFloat('model', 'sigma')
        model_beta  = self.getFloat('model', 'beta')
        model_rho   = self.getFloat('model', 'rho')
        return Lorenz63Model(model_sigma, model_beta, model_rho)

    #_________________________

    def model(self):
        # build model
        model_class = self.getString('model', 'class')
        if model_class == 'Lorenz63':
            return self.Lorenz63Model()

    #_________________________

    def integrator(self, t_truthOrFilter, t_model):
        # build integrator
        integrator_dt  = self.getFloat('integration', 'dt')
        integrator_stc = self.getString('integration', 'step')

        if t_truthOrFilter == 'truth':
            integrator_cls = self.getString('integration', 'class')
            integrator_var = self.getFloat('integration', 'variance')
        else:
            integrator_cls = self.getString('assimilation', 'integration_class')
            integrator_var = self.getFloat('assimilation', 'integration_jitter')

        # error generator
        if integrator_cls == 'Deterministic':
            integrator_eg  = None
        else:
            integrator_std = np.sqrt(integrator_var) * np.ones(self.getInt('dimensions', 'state'))
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

    #_________________________

    def observationOperator(self):
        # build observation operator
        observation_var = self.getFloat('observation-operator', 'variance')
        observation_std = np.sqrt(observation_var) * np.ones(self.getInt('dimensions', 'observations'))
        observation_eg  = IndependantGaussianErrorGenerator(observation_std)
        observation_cls = self.getString('observation-operator', 'class')
        if observation_cls == 'ObserveAll':
            return ObserveAllOperator(observation_eg)
        elif observation_cls == 'ObserveNFirst':
            return ObserveNFirstOperator(observation_eg)

    #_________________________

    def regularObservationTimes(self):
        # build regular observation times
        observation_dt = self.getFloat('observation-times', 'dt')
        observation_Nt = self.getInt('observation-times', 'Nt')
        return RegularObservationTimes(observation_dt, observation_Nt)

    #_________________________

    def observationTimes(self):
        # build observation times
        observationTimes_cls = self.getString('observation-times', 'class')
        if observationTimes_cls == 'Regular':
            return self.regularObservationTimes()

    #_________________________

    def output(self):
        # build output
        output_dir = self.getString('output', 'directory')
        output_mw  = self.getInt('output', 'modWrite')
        output_mp  = self.getInt('output', 'modPrint')
        return Output(output_dir, output_mw, output_mp)

    #_________________________

    def truth(self, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output):
        # build truth
        truthOutput        = self.getStringList('output', 'truth')
        observationsOutput = self.getStringList('output', 'observations')
        return Truth(t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, truthOutput, observationsOutput)

    #_________________________

    def GoldenSectionMinimiser(self, t_prefix):
        # build golden section minimiser from config
        minimiser_maxIt = self.getInt('assimilation', t_prefix+'_maxIt')
        minimiser_tol   = self.getFloat('assimilation', t_prefix+'_tol')
        return GoldenSectionMinimiser(minimiser_maxIt, minimiser_tol)

    #_________________________

    def NewtonMinimiser(self, t_prefix):
        # build newton minimiser from config
        minimiser_dx    = self.getFloat('assimilation', t_prefis+'_dx')
        minimiser_maxIt = self.getInt('assimilation', t_prefix+'_maxIt')
        minimiser_tol   = self.getFloat('assimilation', t_prefix+'_tol')
        return NewtonMinimiser(minimiser_dx, minimiser_maxIt, minimiser_tol)

    #_________________________

    def minimiser(self, t_prefix):
        # build minimiser from config
        minimiser_cls = self.getString('assimilation', t_prefix+'_class')
        # note: minimiser have different 'builder'-functions since they can have different parameters
        if minimiser_cls == 'GoldenSection':
            return GoldenSectionMinimiser(self, t_prefix)
        elif minimiser_cls == 'Newton':
            return NewtonMinimiser(self, t_prefix)

    #_________________________

    def resampler(self):
        # build resampler from config
        resampler_cls = self.getString('assimilation', 'resampler')
        if resampler_cls == 'StochasticUniversal':
            return StochasticUniversalResampler()
        elif resampler_cls == 'Direct':
            return DirectResampler()

    #_________________________

    def EnKF(self, t_class, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, t_Ns, t_outputFields):
        # build EnKF
        filter_ifl = self.getFloat('assimilation', 'inflation')
        filter_jit = self.getFloat('assimilation', 'integration_jitter')
        filter_lbl = EnKFLabel(t_class, t_Ns, filter_ifl, filter_jit)

        if t_class == 'senkf':
            return StochasticEnKF(t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output,
                    filter_lbl, t_Ns, t_outputFields, filter_ifl)

        elif t_class == 'entkf':
            U = np.eye(t_Ns)
            return EnTKF(t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output,
                    filter_lbl, t_Ns, t_outputFields, filter_ifl, U)

        elif t_class == 'entkfn-dual-capped':
            U         = np.eye(t_Ns)
            epsilon   = t_Ns / ( t_Ns - 1.0 )
            maxZeta   = t_Ns - 1.0 
            minimiser = self.minimiser('minimiser_1d')
            return EnTKF_N_dual(t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output,
                    filter_lbl, t_Ns, t_outputFields, filter_ifl, minimiser, epsilon, maxZeta, U)

        elif t_class == 'entkfn-dual':
            U         = np.eye(t_Ns)
            epsilon   = ( t_Ns + 1.0 ) / t_Ns
            maxZeta   = float(t_Ns)
            minimiser = self.minimiser('minimiser_1d')
            return EnTKF_N_dual(t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output,
                    filter_lbl, t_Ns, t_outputFields, filter_ifl, minimiser, epsilon, maxZeta, U)

    #_________________________

    def PF(self, t_class, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, t_Ns, t_outputFields):
        # build PF
        filter_rt  = self.getFloat('assimilation', 'resampling_thr')
        filter_jit = self.getFloat('filter', 'integration_jitter')
        filter_lbl = PFLabel(t_class, t_Ns, filter_rt, filter_jit)

        resampler  = self.resampler()
        trigger    = ThresholdTrigger(filter_rt)

        if t_class == 'sirpf':
            return SIRPF(t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, 
                    filter_lbl, t_Ns, t_outputFields, resampler, trigger)

        elif t_class == 'asirpf':
            return ASIRPF(t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, 
                    filter_lbl, t_Ns, t_outputFields, resampler, trigger)

        elif t_class == 'oisirpf':
            return OISIRPF_diag(t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, 
                    filter_lbl, t_Ns, t_outputFields, resampler, trigger)

    #_________________________

    def EnF(self, t_classInh, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output):
        # build EnF
        filter_Ns  = self.getInt('assimilation', 'Ns')
        filter_out = self.getStringList('assimilation', 'output')
        filter_cls = t_classInh.pop()

        if filter_cls == 'EnKF':
            return self.EnKF(t_classInh[0], t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, filter_Ns, filter_out)
        elif filter_cls == 'PF':
            return self.PF(t_classInh[0], t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, filter_Ns, filter_out)

    #_________________________

    def filter(self, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output):
        # build filter
        fch        = transformedFilterClassHierarchy()
        filter_cls = self.getString('assimilation', 'filter')
        fch        = transformedFilterClassHierarchy()
        filter_inh = fch[filter_cls]

        top_cls    = filter_inh.pop()
        if top_cls == 'EnF':
            return self.EnF(filter_inh, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output)

    #_________________________

    def buildSimulation(self):
        # build simulation
        # initialiser
        initialiser         = self.initialiser()
        # model
        model               = self.model()
        # observation operator
        observationOperator = self.observationOperator()
        # observation times
        observationTimes    = self.observationTimes()
        # output
        output              = self.output()
        # truth integrator
        truthIntegrator     = self.integrator('truth', model)
        # truth
        truth               = self.truth(initialiser, truthIntegrator, observationOperator, observationTimes, output)
        # filter integrator
        filterIntegrator    = self.integrator('filter', model)
        # filter
        filter              = self.filter(initialiser, filterIntegrator, observationOperator, observationTimes, output)
        # simulation
        self.m_simulation   = Simulation(truth, filter, output, observationTimes)
        return self.m_simulation

#__________________________________________________

