#! /usr/bin/env python

#__________________________________________________
# pyLorenz/filters/pf
# entkfn.py
#__________________________________________________
# author        : colonel
# last modified : 2016/10/15
#__________________________________________________
#
# class to handle an ensemble transform kalman filter of finite size
# i.e. the hierarchical counterpart of EnTKF
#

import numpy as np

from abstractenkf import AbstractEnKF

#__________________________________________________

class EnTKF_N(AbstractEnKF):

    #_________________________

    def __init__(self, t_integrator, t_observationOperator, t_Ns, t_covarianceInflation, t_minimiser, t_epsilon, t_maxZeta, t_U = None):
        AbstractEnKF.__init__(self, t_integrator, t_observationOperator, t_Ns, t_covarianceInflation)
        self.setEnTKF_NParameters(t_minimiser, t_epsilon, t_maxZeta, t_U)

    #_________________________

    def setEnTKF_NParameters(self, t_minimiser, t_epsilon, t_maxZeta, t_U):
        # minimiser
        self.m_minimiser = t_minimiser
        # epsilon
        self.m_epsilon   = t_epsilon
        # max value for zeta
        self.m_maxZeta   = t_maxZeta
        # U
        if t_U is None:
            self.m_U     = np.eye(self.m_Ns)
        else:
            self.m_U     = t_U

    #_________________________

    def analyse(self, t_index, t_t, t_observation):
        # analyse observation at time t

        # shortcut
        xf    = self.m_x[t_index]

        # Ensemble means
        xf_m  = xf.mean(axis = 0)
        Hxf   = self.m_observationOperator.deterministicObserve(xf, t_t)
        Hxf_m = Hxf.mean(axis = 0)

        # Normalized anomalies
        Xf    = ( xf - xf_m ) / np.sqrt( self.m_Ns - 1.0 )
        Yf    = ( Hxf - Hxf_m ) / np.sqrt( self.m_Ns - 1.0 )

        # Analyse
        Rm1_2 = self.m_observationOperator.errorStdDevMatrix_inv(t_t)
        #R = self.m_observationOperator.errorCovarianceMatrix(t_t)

        S                      = np.dot ( Yf , Rm1_2 )
        U, D, V                = np.linalg.svd(S)
        diag                   = np.zeros((self.m_Ns, t_observation.size))
        diag[:D.size, :D.size] = np.diag(D)
        #print 'svd de S:', np.abs(S - np.dot( U , np.dot( diag , V ) ) ).max()

        delta                  = np.dot ( V , np.dot ( Rm1_2 , t_observation - Hxf_m ) )

        # dual cost to minimise
        #diagonal1 = np.dot(np.transpose(diag), diag)
        diagonal               = np.zeros(delta.size)
        diagonal[:D.size]      = D * D
        #print 'diag diff = ', np.abs(np.diag(diagonal1)-diagonal).max()
        #delta0 = t_observation - Hxf_m
        #YYT = np.dot(np.transpose(Yf), Yf)
        #def dualCost1(zeta):
            #return ( 0.5 * np.dot ( delta0 , np.dot( np.linalg.inv( R + ( ( self.m_Ns - 1.0 ) / zeta ) * YYT ) , delta0 ) ) +
                    #0.5 * epsilon * zeta +
                    #( self.m_Ns + 1.0 ) * 0.5 * np.log( ( self.m_Ns + 1.0 ) / zeta ) )
        def dualCost(zeta):
            return ( ( delta * delta / ( 1.0 + diagonal * ( self.m_Ns - 1.0 ) / zeta ) ).sum() +
                    self.m_epsilon * zeta -
                    ( self.m_Ns + 1.0 ) * np.log( zeta ) )
                    #( self.m_Ns + 1.0 ) * 0.5 * np.log( ( self.m_Ns + 1.0 ) / zeta ) )

        """
        zeta = np.linspace(0.0, 2.0*self.m_Ns, self.m_Nzeta)
        #dcZeta1 = np.zeros(self.m_Nzeta)
        dcZeta2 = np.zeros(self.m_Nzeta)
        for i in range(self.m_Nzeta):
            if i > 0:
                #dcZeta1[i] = dualCost1(zeta[i])
                dcZeta2[i] = dualCost2(zeta[i])

        #dcZeta1[0]=dcZeta1[1]
        dcZeta2[0]=dcZeta2[1]
        #imin1 = np.argmin(dcZeta1)
        imin2 = np.argmin(dcZeta2)
        #print 'zeta min1 = ', zeta[imin1]
        print 'zeta min2 = ', zeta[imin2]

        #return dcZeta1,dcZeta2
        """

        (zetaa, nit) = self.m_minimiser.minimiseInterval(dualCost, 0.0, self.m_maxZeta, self.m_maxZeta)

        #zetaa        = max(min(zetaa, self.m_maxZeta), 1.0)
        #zetaa        = min(zetaa, self.m_maxZeta)
        #print 'zetaa = ', zetaa
        #if zetaa < 0.1:
            #self.m_dualCost = dualCost2
            #print 'too much inflation'
            #raise Exception
        """
        if zetaa > self.m_Ns - 1.0:
            print 'deflation is used !!', zetaa
        else:
            print zetaa
        """


        # analyse weights
        delta              = np.dot(diag, delta)
        diagonal           = np.zeros(self.m_Ns)
        diagonal[:D.size]  = D * D
        #print 'diag diff = ', np.abs(diagonal - np.diag(np.dot(diag, np.transpose(diag)))).max()
        diagonal          += zetaa / ( self.m_Ns - 1.0 )
        delta             /= diagonal
        wa                 = np.dot( U , delta )

        #delta0 = t_observation - Hxf_m
        #YYT = np.dot(np.transpose(Yf), Yf)
        #wa_naiv = np.dot( Yf, np.dot( np.linalg.inv( (zetaa/(self.m_Ns-1.0)) * R + YYT ) , delta0) )
        #print 'wa diff =', np.abs(wa-wa_naiv).max()

        # new ensemble
        Xa = np.dot( self.m_U , np.dot( U / np.sqrt( diagonal ) , np.transpose(U) ) )

        #Ha = np.dot( Yf , np.dot( np.linalg.inv(R) , np.transpose(Yf))) + (zetaa/(self.m_Ns-1.0))*np.eye(self.m_Ns)
        #a, b, c = np.linalg.svd(Ha)
        #print 'svd Ha :', np.abs(Ha-np.dot(a, np.dot(np.diag(b), c))).max()
        #srHam1 = np.dot(a, np.dot( np.diag( 1.0 / np.sqrt(b)), c))
        #print 'srHam1 : ', np.abs(np.dot(srHam1, srHam1)-np.linalg.inv(Ha)).max()
        #print 'Xa  :', np.abs(Xa - np.transpose(np.dot(srHam1, self.m_U))).max()

        self.m_x[t_index] = xf_m + np.dot( wa + np.sqrt( self.m_Ns - 1.0 ) * Xa , Xf )

#__________________________________________________

