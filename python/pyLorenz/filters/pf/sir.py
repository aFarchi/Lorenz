#! /usr/bin/env python

#__________________________________________________
# pyLorenz/filters/pf
# sir.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/20
#__________________________________________________
#
# class to handle a SIR particle filter
#

import numpy as np

#__________________________________________________

class SIRPF:

    #_________________________

    def __init__(self):
        # constructor
        self.m_weightsTolerance = 1.0e-8
        self.m_resampled        = []

    #_________________________

    def setParameters(self, t_observationVarianceInflation = 10.0, t_resamplingThreshold = 0.3):
        # parameters of the SIR algorithm
        self.m_observationVarianceInflation = t_observationVarianceInflation
        self.m_resamplingThreshold          = t_resamplingThreshold

    #_________________________

    def setModel(self, t_model):
        # set model
        self.m_model = t_model

    #_________________________

    def setIntegrator(self, t_integrator):
        # set integrator
        self.m_integrator = t_integrator

    #_________________________

    def setResampler(self, t_resampler):
        # set resampler
        self.m_resampler = t_resampler

    #_________________________

    def setObservationOperator(self, t_obsOp):
        # set observation operator
        self.m_observationOperator = t_obsOp

    #_________________________

    def initialise(self, t_x):
        # particles / samples
        self.m_x  = t_x
        # number of particles / samples
        self.m_Ns = t_x.shape[0]
        # relative weights
        self.m_w  = np.ones(self.m_Ns) / self.m_Ns

    #_________________________

    def Neff(self):
        # empirical effective relative sample size
        return 1.0 / ( np.power(self.m_w, 2).sum() * self.m_Ns )

    #_________________________

    def resampledSteps(self):
        return 1.0 * np.array(self.m_resampled)

    #_________________________

    def analyse(self, t_nt, t_obs):
        # observation weights
        w = self.m_observationOperator.observationPDF(t_obs, self.m_x, self.m_observationVarianceInflation)

        if w.max() < self.m_weightsTolerance:
            ###_____________________________
            ### --->>> TO IMPROVE <<<--- ###
            ###_____________________________
            # filter has diverged from the truth...
            # ignore observation
            print('filter divergence, nt='+str(t_nt))
            w = 1.0

        # reweight ensemble
        self.m_w *= w
        # normalize weights
        self.m_w /= self.m_w.sum()
        # resample if needed
        if self.Neff() < self.m_resamplingThreshold:
            ###_______________________________
            ### --->>> REMOVE PRINT <<<--- ###
            ###_______________________________
            print('resampling, nt='+str(t_nt))
            (self.m_w, self.m_x) = self.m_resampler.resample(self.m_w, self.m_x)
            self.m_resampled.append(t_nt)

    #_________________________

    def forecast(self):
        # integrate particles
        self.m_x = self.m_integrator.process(self.m_model, self.m_x)

    #_________________________

    def estimate(self):
        # mean of x
        return np.average(self.m_x, axis = 0, weights = self.m_w)

#__________________________________________________

