#! /usr/bin/env python

#__________________________________________________
# pyLorenz/filters/pf/
# rmimplicitpf.py
#__________________________________________________
# author        : colonel
# last modified : 2016/9/28
#__________________________________________________
#
# class to handle a SIR particle filter that uses the optimal importance function
# assuming noise are additive (i.e. StochasticProcess class is used) gaussian and independant
# for both observation operator and integrator
#

import numpy as np
import numpy.random as rnd

from sir                                             import SIRPF
from ...utils.integration.rk4integrator              import DeterministicRK4Integrator
from ...observations.iobservations                   import StochasticIObservations
from ...utils.resampling.stochasticuniversalsampling import StochasticUniversalResampler
from ...utils.minimisation.newtonminimiser           import NewtonMinimiser
from ...utils.trigger.thresholdtrigger               import ThresholdTrigger

#__________________________________________________

class RMImplicitPF(SIRPF):

    #_________________________

    def __init__(self, t_integrator = DeterministicRK4Integrator(), t_obsOp = StochasticIObservations(), t_resampler = StochasticUniversalResampler(), 
            t_observationVarianceInflation = 1.0, t_resamplingTrigger = ThresholdTrigger(0.3), t_Ns = 10, t_minimiser = NewtonMinimiser(), t_perturbationLambda = 1.0e-5):
        SIRPF.__init__(self, t_integrator, t_obsOp, t_resampler, t_observationVarianceInflation, t_resamplingTrigger, t_Ns)
        self.setRMImplicitPFParameters(t_minimiser, t_perturbationLambda)
        self.m_deterministicIntegrator          = self.m_integrator.deterministicIntegrator()
        self.m_deterministicObservationOperator = self.m_observationOperator.deterministicObservationOperator()

    #_________________________

    def setRMImplicitPFParameters(self, t_minimiser = NewtonMinimiser(), t_perturbationLambda = 1.0e-5):
        self.m_minimiser          = t_minimiser
        self.m_perturbationLambda = t_perturbationLambda

    #_________________________

    def forecast(self, t_ntStart, t_ntEnd, t_observation):
        # integrate particles from ntStart to ntEnd, given the observation at ntEnd
        # this includes analyse step for conveniance

        # shortcuts
        r       = t_ntEnd - t_ntStart                                 # number of steps to predict
        d       = self.m_integrator.m_model.m_stateDimension          # state dimension
        sigma_m = self.m_integrator.m_errorGenerator.m_sigma          # model error covariance
        sigma_o = self.m_observationOperator.m_errorGenerator.m_sigma # observation error covariance

        trajectory = np.zeros((self.m_Ns, r, d)) # prediction
        w          = np.zeros(self.m_Ns) # associated weights

        for i in np.arange(self.m_Ns): # for each particle

            def Fi(Xi):
                # cost function for particle i
                # Fi(Xi) = Fi(xi[n0+1]...xi[n0+r]) = - 2 * log ( p ( xi[n0+1] | xi[n0] ) * ... * p ( xi[n0+r] | xi[n0+r-1] ) * p ( y[n0+r] | xi[n0+r] ) )
                # here Xi is expected as a d*r-vector
                # with Xi[d*nt:d*(nt+1)] = xi[nt]
                #
                # note that n0 = t_ntStart, y[n0+r] and xi[n0] are given

                me = Xi[:d] - self.m_deterministicIntegrator.process(self.m_x[i], t_ntStart) # model error when integrating xi[n0]
                p  = ( me * ( 1.0 / sigma_m ) * me ).sum() # - 2 * log ( p ( xi[n0+1] | xi[n0] ) )
                for nt in np.arange(r-1):
                    me   = Xi[d*(nt+1):d*(nt+2)] - self.m_deterministicIntegrator.process(Xi[d*nt:d*(nt+1)], t_ntStart+nt+1) # model error when integrating xi[n0+nt+1]
                    p   += ( me * ( 1.0 / sigma_m ) * me ).sum() # - 2 * log ( p ( xi[n0+nt+2] | xi[n0+nt+1] ) )

                oe  = t_observation - self.m_deterministicObservationOperator.process(Xi[d*(r-1):], t_ntEnd*self.m_integrator.m_dt) # observation error at n0+r
                p  += ( oe * ( 1.0 / sigma_o ) * oe ).sum() # - 2 * log ( p ( y[n0+r] | xi[n0+r] ) )

                return p / 2.0

            # We first want to minimise Fi
            # for that we use Newton's method
            # and we initialise the algorihtm with a model run
            Xi0     = np.zeros(d*r)
            Xi0[:d] = self.m_deterministicIntegrator.process(self.m_x[i], t_ntStart)
            for nt in np.arange(r-1):
                Xi0[d*(nt+1):d*(nt+2)] = self.m_deterministicIntegrator.process(Xi0[d*nt:d*(nt+1)], t_ntStart+nt+1)

            # minimisation of Fi
            # mui = argmin(Fi)
            # Hi  = hessian of Fi at mui
            # nit = number of iterations necessary for Newton's method to converge
            # phi = Fi(mui) = min(Fi)
            (mui, Hi, nit) = self.m_minimiser.minimise(Fi, Xi0)
            #---------------------------------------
            # print('minimisation, nit = '+str(nit))
            #---------------------------------------
            phi = Fi(mui)

            # Cholesky decomposition of Hi
            Li = np.linalg.cholesky(Hi)

            # Random sample
            zetai = np.random.normal(np.zeros(d*r), np.ones(d*r)) # zetai ~ N(O,I)
            rhoi  = ( zetai * zetai ).sum()

            # We now want to solve Fi ( mui + lambdai * transpose(Li) * zetai / sqrt(rohi) ) = phi + 0.5 * rhoi
            # for that we once again use Newton's method
            # and we initialise the algorithm with lambadi0 = sqrt(rhoi)
            lambdai0 = np.sqrt(rhoi)
            diri     = np.dot(np.transpose(Li), zetai) / lambdai0

            def FiLambda(lambdai):
                return Fi ( mui + lambdai * diri )
            leveli = phi + rhoi / 2.0

            # finding the level of FiLambda
            # FiLambda(lambdai) = phi + rho
            # nit = number of iterations necessary for Newton's method to converge
            (lambdai, nit) = self.m_minimiser.findLevel(FiLambda, leveli, lambdai0)
            #---------------------------------
            # print('levels, nit = '+str(nit))
            #---------------------------------

            # Record the computed trajectory
            trajectory[i] = ( mui + lambdai * diri ).reshape((r,d))

            # Finally we compute the updated weights
            # for that we need the value of the jacobian of the map zetai -> Xi
            # which depends on dlambda / drho

            # compute dlambda / drho using finite differences
            dlambda = self.m_perturbationLambda * lambdai0
            rhopp   = FiLambda ( lambdai + dlambda )
            rhomm   = FiLambda ( lambdai - dlambda )
            dldrho  = 2.0 * dlambda / ( rhopp - rhomm )

            # compute the jacobian of the map zetai -> Xi
            logJi = np.log(np.abs(np.linalg.det(Li))) + ( 1.0 - r * d / 2.0 ) * np.log(rhoi) + ( r * d - 1.0 ) * np.log(np.abs(lambdai)) + np.log(np.abs(dldrho))

            # update weight in log scale
            w[i] = logJi - phi

        # classic analyse process
        self.m_w += w
        self.normaliseWeights()
        self.resample(t_ntEnd)

        # estimation of the trajectory after the analyse
        for nt in np.arange(r):
            self.m_estimate[t_ntStart+nt+1] = np.average(trajectory[:,nt,:], axis = 0, weights = np.exp(self.m_w))
        self.m_neff[t_ntStart+1:t_ntEnd+1] = self.Neff()

        # update model state
        self.m_x[:,:] = trajectory[:,r-1,:]

    #_________________________

    def analyse(self, t_nt, t_obs):
        if t_nt == 0:
            SIRPF.analyse(self, t_nt, t_obs)

#__________________________________________________

