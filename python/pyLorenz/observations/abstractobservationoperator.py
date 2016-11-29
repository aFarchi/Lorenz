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

    def deterministicObserve(self, t_x, t_t, t_y):
        # deterministic observation
        raise NotImplementedError

    #_________________________

    def observe(self, t_x, t_t, t_y):
        # deterministic observation + observation errors
        self.deterministicObserve(t_x, t_t, t_y)
        t_y += self.drawErrorSamples(t_t, t_y.shape)

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

    def pdf(self, t_observation, t_Hx, t_t):
        # observation pdf in log scale : log ( p ( observation | x ) )
        # in this case it is the error generator pdf at time t taken at point ( observation - H ( x ) )
        return self.m_errorGenerator.pdf(t_observation-t_Hx, t_t)

    #_________________________

    def applyLeftErrorCovMatrix_inv(self, t_x):
        # return R^(-1) . x
        return self.m_errorGenerator.applyLeftCovMatrix_inv(t_x)

    #_________________________

    def applyLeftErrorStdDevMatrix_inv(self, t_x):
        # return R^(-1/2) . x
        return self.m_errorGenerator.applyLeftStdDevMatrix_inv(t_x)

    #_________________________

    def applyLeftErrorStdDevMatrix_inv_local(self, t_x, t_dimensions):
        # return R^(-1/2)[dimensions, dimensions] . x[dimensions, ...]
        return self.m_errorGenerator.applyLeftStdDevMatrix_inv_local(t_x, t_dimensions)

    #_________________________

    def applyRightErrorCovMatrix_inv(self, t_x):
        # return x . R^(-1)
        return self.m_errorGenerator.applyRightCovMatrix_inv(t_x)

    #_________________________

    def applyRightErrorStdDevMatrix_inv(self, t_x):
        # return x . R^(-1/2)
        return self.m_errorGenerator.applyRightStdDevMatrix_inv(t_x)

    #_________________________

    def applyRightErrorStdDevMatrix_inv_local(self, t_x, t_dimensions):
        # return x[..., dimensions] . R^(-1/2)[dimensions, dimensions]
        return self.m_errorGenerator.applyRightStdDevMatrix_inv_local(t_x, t_dimensions)

    #_________________________

    def errorStdDevMatrix_inv(self, t_t):
        # return the inverse of the standard deviation matrix
        return self.m_errorGenerator.stdDevMatrix_inv(t_t)

    #_________________________

    def errorCovarianceMatrix(self, t_t):
        # return the inverse of the standard deviation matrix
        return self.m_errorGenerator.covarianceMatrix(t_t)

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

    #_________________________

    def nearest_y_dimensions(self, t_nearest_x_dimensions):
        # nearest dimensions in observation space
        raise NotImplementedError

    #_________________________

    def cast_localisation_coefficients_to_observation_space(self, t_localisation_coefficients):
        # cast loc. coeff. from state space into observation space
        raise NotImplementedError

#__________________________________________________

