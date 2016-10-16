#! /usr/bin/env python

#__________________________________________________
# pyLorenz/observations/
# abstractobservationoperator.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/13
#__________________________________________________
#
# abstract class to handle an observation operator
#

#__________________________________________________

class AbstractObservationOperator(object):

    #_________________________

    def __init__(self, t_errorGenerator):
        self.setAbstractObservationOperatorParameters(t_errorGenerator)

    #_________________________

    def setAbstractObservationOperatorParameters(self, t_errorGenerator):
        # space dimension
        self.m_spaceDimension = t_errorGenerator.m_spaceDimension
        # error generator
        self.m_errorGenerator = t_errorGenerator

    #_________________________

    def deterministicObserve(self, t_x, t_t):
        # deterministic observation
        raise NotImplementedError

    #_________________________

    def observe(self, t_x, t_t):
        # deterministic observation + observation errors
        y = self.deterministicObserve(t_x, t_t)
        return y + self.drawErrorSamples(t_t, y.shape)

    #_________________________

    def isLinear(self):
        # return true if, and ony if deterministicProcess() is a linear function of x
        raise NotImplementedError

    #_________________________

    def differential(self, t_x, t_t):
        # return the matrix of the differential of deterministicProcess about (x, nt) at point (x, nt)
        raise NotImplementedError

    #_________________________

    def differential_diag(self, t_x, t_t):
        # return the diagonal of the differential matrix
        raise NotImplementedError

    #_________________________

    def drawErrorSamples(self, t_t, t_shape):
        # draw samples from error generator at time t
        return self.m_errorGenerator.drawSamples(t_t, t_shape)

    #_________________________

    def pdf(self, t_observation, t_x, t_t):
        # observation pdf in log scale : log ( p ( observation | x ) )
        # in this case it is the error generator pdf at time t taken at point ( observation - H ( x ) )
        return self.m_errorGenerator.pdf(t_observation-self.deterministicObserve(t_x, t_t), t_t)

    #_________________________

    def errorCovarianceMatrix_diag(self, t_t, t_spaceDimension):
        # return the diagonal of the covariance matrix
        # which is then casted in state space (whose dimension is t_spaceDimension)
        raise NotImplementedError

    #_________________________

    def errorStdDevMatrix_diag(self, t_t, t_spaceDimension):
        # return the diagonal of the standard deviation matrix
        # which is then casted in state space (whose dimension is t_spaceDimension)
        raise NotImplementedError

    #_________________________

    def castObservationToStateSpace(self, t_observation, t_t, t_spaceDimension):
        # cast observation in state space (whose dimension is t_spaceDimension)
        raise NotImplementedError

#__________________________________________________

