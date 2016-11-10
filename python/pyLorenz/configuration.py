#! /usr/bin/env python

#__________________________________________________
# pyLorenz/
# configuration.py
#__________________________________________________
# author        : colonel
# last modified : 2016/11/5
#__________________________________________________
#
# configuration for a simulation
#

import numpy as np

from numpy.random                                    import RandomState

from utils.auxiliary.bash                            import configFileNamesFromCommand
from utils.auxiliary.configparser                    import ConfigParser
from utils.auxiliary.dictutils                       import makeKeyDictRecListDict

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

from model.nomodel                                   import NoModel
from model.lorenz63                                  import Lorenz63Model
from model.lorenz95                                  import Lorenz95Model

from observations.observealloperator                 import ObserveAllOperator
from observations.regularobservationtimes            import RegularObservationTimes

from truth.simulatedtruth                            import SimulatedTruth
from truth.truthfromfile                             import TruthFromFile

from filters.kalman.stochasticenkf                   import StochasticEnKF
from filters.kalman.entkf                            import EnTKF
from filters.kalman.entkfn                           import EnTKF_N_dual

from filters.pf.sir                                  import SIRPF
from filters.pf.oisir                                import OISIRPF_diag
from filters.pf.asir                                 import ASIRPF

#from simulation.simulation_debug                     import Simulation
from simulation.simulation                           import Simulation

#__________________________________________________

def filterClassHierarchy():
    # hierarchy of implemented filter classes
    fch                = {}
    fch['EnF']         = {}
    fch['EnF']['EnKF'] = ['StoEnKF', 'ETKF', 'ETKF-N-dual']
    fch['EnF']['PF']   = ['SIR', 'ASIR', 'OISIR']
    return fch

#__________________________________________________

def transformedFilterClassHierarchy():
    # tranform class hierarchy to make it 'useful'
    # each key is now a filter class
    # and is associated with its inheritance hierarchy
    return makeKeyDictRecListDict(filterClassHierarchy())

#__________________________________________________

class Configuration(object):

    #_________________________

    def __init__(self):
        # config parser
        self.m_config   = ConfigParser()
        self.m_config.readfiles(configFileNamesFromCommand())
        # list of filters
        self.m_filters  = self.filterList()
        # transformed filter class hierarchy
        self.m_tfch     = transformedFilterClassHierarchy()

        # make some test to configuration
        self.checkFromTruth()
        self.checkDeterministicIntegration()

    #_________________________

    def filterList(self):
        # list of filters
        sections   = self.m_config.options()
        filterList = []
        for section in sections:
            if not section in ['dimensions', 'model', 'observation-times', 'output', 'truth']:
                filterList.append(section)
        return filterList

    #_________________________

    def checkFromTruth(self):
        # replace 'from truth' occurences by their values
        for f in self.m_filters:

            if self.m_config.get(f, 'initialisation', 'mean') == 'from truth':
                self.m_config.set(f, 'initialisation', 'mean', self.m_config.get('truth', 'initialisation', 'mean'))
            if self.m_config.get(f, 'initialisation', 'variance') == 'from truth':
                self.m_config.set(f, 'initialisation', 'variance', self.m_config.get('truth', 'initialisation', 'variance'))

            if self.m_config.get(f, 'integration', 'class') == 'from truth':
                self.m_config.set(f, 'integration', 'class', self.m_config.get('truth', 'integration', 'class'))
            if self.m_config.get(f, 'integration', 'step') == 'from truth':
                self.m_config.set(f, 'integration', 'step', self.m_config.get('truth', 'integration', 'step'))
            if self.m_config.get(f, 'integration', 'dt') == 'from truth':
                self.m_config.set(f, 'integration', 'dt', self.m_config.get('truth', 'integration', 'dt'))
            if self.m_config.get(f, 'integration', 'variance') == 'from truth':
                self.m_config.set(f, 'integration', 'variance', self.m_config.get('truth', 'integration', 'variance'))

            if self.m_config.get(f, 'observation-operator', 'class') == 'from truth':
                self.m_config.set(f, 'observation-operator', 'class', self.m_config.get('truth', 'observation-operator', 'class'))
            if self.m_config.get(f, 'observation-operator', 'variance') == 'from truth':
                self.m_config.set(f, 'observation-operator', 'variance', self.m_config.get('truth', 'observation-operator', 'variance'))

    #_________________________

    def checkDeterministicIntegration(self):
        # make sure zero integration variance is used with deterministic integrator
        integration_var = self.m_config.getFloat('truth', 'integration', 'variance')
        if integration_var is None or integration_var == 0.0:
            self.m_config.set('truth', 'integration', 'class', 'Deterministic')

        if self.m_config.get('truth', 'integration', 'class') == 'Deterministic':
            self.m_config.set('truth', 'integration', 'variance', '0.0')

        for f in self.m_filters:
            integration_jit = self.m_config.getFloat(f, 'integration', 'variance')
            if integration_jit is None or integration_jit == 0.0:
                self.m_config.set(f, 'integration', 'class', 'Deterministic')

            if self.m_config.get(f, 'integration', 'class') == 'Deterministic':
                self.m_config.set(f, 'integration', 'variance', '0.0')

    #_________________________

    def initialiser(self, t_truthOrFilter):
        # build initialiser
        initialiser_mean = self.m_config.getNumpyArray(t_truthOrFilter, 'initialisation', 'mean')
        initialiser_std  = np.sqrt(self.m_config.getFloat(t_truthOrFilter, 'initialisation', 'variance')) * np.ones(self.m_config.getInt('dimensions', 'state'))
        try:
            seed         = self.m_config.getInt(t_truthOrFilter, 'initialisation', 'seed')
        except:
            seed         = None
        initialiser_rng  = RandomState(seed)
        initialiser_eg   = IndependantGaussianErrorGenerator(initialiser_std, initialiser_rng)
        return RandomInitialiser(initialiser_mean, initialiser_eg)

    #_________________________

    def Lorenz63Model(self):
        # build Lorenz63 model
        model_sigma = self.m_config.getFloat('model', 'sigma')
        model_beta  = self.m_config.getFloat('model', 'beta')
        model_rho   = self.m_config.getFloat('model', 'rho')
        return Lorenz63Model(model_sigma, model_beta, model_rho)

    #_________________________

    def Lorenz95Model(self):
        # build Lorenz95 model
        model_d = self.m_config.getInt('dimensions', 'state')
        model_f = self.m_config.getFloat('model', 'f')
        return Lorenz95Model(model_d, model_f)

    #_________________________

    def NoModel(self):
        # build Lorenz95 model
        model_d = self.m_config.getInt('dimensions', 'state')
        return NoModel(model_d)

    #_________________________

    def model(self):
        # build model
        model_class = self.m_config.get('model', 'class')
        if model_class == 'Lorenz63':
            return self.Lorenz63Model()
        elif model_class == 'Lorenz95':
            return self.Lorenz95Model()
        elif model_class == 'NoModel':
            return self.NoModel()

    #_________________________

    def integrator(self, t_truthOrFilter, t_model):
        # build integrator
        integrator_dt  = self.m_config.getFloat(t_truthOrFilter, 'integration', 'dt')
        integrator_stc = self.m_config.get(t_truthOrFilter, 'integration', 'step')
        integrator_cls = self.m_config.get(t_truthOrFilter, 'integration', 'class')
        integrator_var = self.m_config.getFloat(t_truthOrFilter, 'integration', 'variance')
        try:
            seed       = self.m_config.getInt(t_truthOrFilter, 'integration', 'seed')
        except:
            seed       = None
        integrator_rng = RandomState(seed)

        # error generator
        if integrator_cls == 'Deterministic':
            integrator_eg  = None
        else:
            integrator_std = np.sqrt(integrator_var) * np.ones(self.m_config.getInt('dimensions', 'state'))
            integrator_eg  = IndependantGaussianErrorGenerator(integrator_std, integrator_rng)

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

    def observationOperator(self, t_truthOrFilter):
        # build observation operator
        observation_var = self.m_config.getFloat(t_truthOrFilter, 'observation-operator', 'variance')
        observation_std = np.sqrt(observation_var) * np.ones(self.m_config.getInt('dimensions', 'observations'))
        try:
            seed        = self.m_config.getInt(t_truthOrFilter, 'observation-operator', 'seed')
        except:
            seed        = None
        observation_rng = RandomState(seed)
        observation_eg  = IndependantGaussianErrorGenerator(observation_std, observation_rng)
        observation_cls = self.m_config.get(t_truthOrFilter, 'observation-operator', 'class')
        if observation_cls == 'ObserveAll':
            return ObserveAllOperator(observation_eg)
        elif observation_cls == 'ObserveNFirst':
            return ObserveNFirstOperator(observation_eg)

    #_________________________

    def regularObservationTimes(self):
        # build regular observation times
        observation_dt = self.m_config.getFloat('observation-times', 'dt')
        observation_Nt = self.m_config.getInt('observation-times', 'Nt')
        return RegularObservationTimes(observation_dt, observation_Nt)

    #_________________________

    def observationTimes(self):
        # build observation times
        observationTimes_cls = self.m_config.get('observation-times', 'class')
        if observationTimes_cls == 'Regular':
            return self.regularObservationTimes()

    #_________________________

    def output(self):
        # build output
        output_dir = self.m_config.get('output', 'directory')
        if len(output_dir) == 0:
            output_dir = './'
        if not output_dir[-1] == '/':
            output_dir += '/'
        output_mw  = self.m_config.getInt('output', 'modWrite')
        output_mp  = self.m_config.getInt('output', 'modPrint')
        output_lbl = self.m_config.get('output', 'label')
        return Output(output_dir, output_mw, output_mp, output_lbl)

    #_________________________

    def truthFromFile(self, t_observationTimes, t_output, t_truthOutputFields):
        # build truth from file
        truthFile        = self.m_config.get('truth', 'files', 'truth')
        observationsFile = self.m_config.get('truth', 'files', 'observations')
        bufferSize       = self.m_config.getInt('truth', 'files', 'buffer_size')
        xDimension       = self.m_config.getInt('dimensions', 'state')
        yDimension       = self.m_config.getInt('dimensions', 'observations')
        return TruthFromFile(truthFile, observationsFile, bufferSize, xDimension, yDimension, t_observationTimes, t_output, t_truthOutputFields)

    #_________________________

    def simulatedTruth(self, t_model, t_observationTimes, t_output, t_truthOutputFields):
        # build simulated truth
        initialiser = self.initialiser('truth')
        integrator  = self.integrator('truth', t_model)
        observation = self.observationOperator('truth')
        return SimulatedTruth(initialiser, integrator, observation, t_observationTimes, t_output, t_truthOutputFields)

    #_________________________

    def truth(self, t_model, t_observationTimes, t_output):
        # build truth
        truth_cls = self.m_config.get('truth', 'class')
        truth_out = self.m_config.getStringList('truth', 'output', 'fields')
        if truth_cls == 'Simulated':
            return self.simulatedTruth(t_model, t_observationTimes, t_output, truth_out)
        elif truth_cls == 'FromFile':
            return self.truthFromFile(t_observationTimes, t_output, truth_out)

    #_________________________

    def GoldenSectionMinimiser(self, *t_options):
        # build golden section minimiser
        opt             = list(t_options)
        minimiser_maxIt = self.m_config.getInt(*(opt+['max_it']))
        minimiser_tol   = self.m_config.getFloat(*(opt+['tolerance']))
        return GoldenSectionMinimiser(minimiser_maxIt, minimiser_tol)

    #_________________________

    def NewtonMinimiser(self, *t_options):
        # build newton minimiser
        opt             = list(t_options)
        minimiser_dx    = self.m_config.getFloat(*(opt+['dx']))
        minimiser_maxIt = self.m_config.getInt(*(opt+['max_it']))
        minimiser_tol   = self.m_config.getFloat(*(opt+['tolerance']))
        return NewtonMinimiser(minimiser_dx, minimiser_maxIt, minimiser_tol)

    #_________________________

    def minimiser(self, *t_options):
        # build minimiser
        opt           = list(t_options)
        opt.append('class')
        minimiser_cls = self.m_config.get(*opt)
        if minimiser_cls == 'GoldenSection':
            return self.GoldenSectionMinimiser(*t_options)
        elif minimiser_cls == 'Newton':
            return self.NewtonMinimiser(*t_options)

    #_________________________

    def resampler(self, t_filter):
        # build resampler
        resampler_cls = self.m_config.get(t_filter, 'resampling', 'class')
        try:
            seed      = self.m_config.getInt(t_filter, 'resampling', 'seed')
        except:
            seed      = None
        resampler_rng = RandomState(seed)
        if resampler_cls == 'StochasticUniversal':
            return StochasticUniversalResampler(resampler_rng)
        elif resampler_cls == 'Direct':
            return DirectResampler(resampler_rng)

    #_________________________

    def EnKF(self, t_filter, t_class, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, t_Ns, t_outputFields):
        # build EnKF
        filter_ifl = self.m_config.getFloat(t_filter, 'ensemble', 'inflation')

        if t_class == 'StoEnKF':
            return StochasticEnKF(t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output,
                    t_filter, t_Ns, t_outputFields, filter_ifl)

        elif t_class == 'ETKF':
            U = np.eye(t_Ns)
            return EnTKF(t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output,
                    t_filter, t_Ns, t_outputFields, filter_ifl, U)

        elif t_class == 'ETKF-N-dual':
            U         = np.eye(t_Ns)
            epsilon   = self.m_config.getFloat(t_filter, 'dual-minimisation', 'epsilon')
            maxZeta   = self.m_config.getFloat(t_filter, 'dual-minimisation', 'maxZeta')
            minimiser = self.minimiser(t_filter, 'dual-minimisation', 'minimiser')
            return EnTKF_N_dual(t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output,
                    t_filter, t_Ns, t_outputFields, filter_ifl, minimiser, epsilon, maxZeta, U)

    #_________________________

    def PF(self, t_filter, t_class, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, t_Ns, t_outputFields):
        # build PF
        filter_rt  = self.m_config.getFloat(t_filter, 'resampling', 'threshold')
        resampler  = self.resampler(t_filter)
        trigger    = ThresholdTrigger(filter_rt)

        if t_class == 'SIR':
            return SIRPF(t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, 
                    t_filter, t_Ns, t_outputFields, resampler, trigger)

        elif t_class == 'ASIR':
            return ASIRPF(t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, 
                    t_filter, t_Ns, t_outputFields, resampler, trigger)

        elif t_class == 'OISIR':
            try:
                filter_rng = t_integrator.m_integrationStep.m_errorGenerator.m_rng
            except:
                try:
                    seed   = self.m_config.getInt(t_filter, 'integration', 'seed')
                except:
                    seed   = None
                filter_rng = RandomState(seed)
            return OISIRPF_diag(t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, 
                    t_filter, t_Ns, t_outputFields, resampler, trigger, filter_rng)

    #_________________________

    def EnF(self, t_filter, t_classInh, t_filter_cls, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output):
        # build EnF
        filter_Ns  = self.m_config.getInt(t_filter, 'ensemble', 'Ns')
        filter_out = self.m_config.getStringList(t_filter, 'output', 'fields')
        filter_tcl = t_classInh.pop(0)

        if filter_tcl == 'EnKF':
            return self.EnKF(t_filter, t_filter_cls, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, filter_Ns, filter_out)
        elif filter_tcl == 'PF':
            return self.PF(t_filter, t_filter_cls, t_initialiser, t_integrator, t_observationOperator, t_observationTimes, t_output, filter_Ns, filter_out)

    #_________________________

    def filter(self, t_filter, t_model, t_observationTimes, t_output):
        # build filter
        filter_cls  = self.m_config.get(t_filter, 'filter')
        filter_inh  = list(self.m_tfch[filter_cls])

        initialiser = self.initialiser(t_filter)
        integrator  = self.integrator(t_filter, t_model)
        observation = self.observationOperator(t_filter)

        top_cls    = filter_inh.pop(0)
        if top_cls == 'EnF':
            return self.EnF(t_filter, filter_inh, filter_cls, initialiser, integrator, observation, t_observationTimes, t_output)

    #_________________________

    def filters(self, t_model, t_observationTimes, t_output):
        # build all filters
        filters = []

        for f in self.m_filters:
            filters.append(self.filter(f, t_model, t_observationTimes, t_output))

        return filters

    #_________________________

    def buildSimulation(self):
        # model
        model             = self.model()
        # observation times
        observationTimes  = self.observationTimes()
        # output
        output            = self.output()
        # truth
        truth             = self.truth(model, observationTimes, output)
        # filters
        filters           = self.filters(model, observationTimes, output)

        # simulation
        self.m_simulation = Simulation(truth, filters, output, observationTimes)

        return self.m_simulation

#__________________________________________________

