
/*__________________________________________________
 * model/
 * lorenz63.h
 *__________________________________________________
 * author        : colonel
 * last modified : 2016/9/16
 *__________________________________________________
 *
 * class to handle the step function of a Lorenz 1963 model
 *
 */

#ifndef MODEL_LORENZ63_H
#define MODEL_LORENZ63_H

#include "../utils/random/independantGaussianRNG.h"

namespace model
{

    enum IntegrationMethod {EulerExpl, RK2, RK4} ;

    template <std::size_t N, typename Real>
        using IGRNG = utils::random::IndependantGaussianRNG<N, Real, std::default_random_engine, std::normal_distribution> ;

    template <typename Real = double, template <std::size_t, typename> class ModelErrorGenerator = IGRNG>
        class Lorenz63Model
        {
            public:
                Lorenz63Model() ;
                void setModelParameters(Real t_sigma = 10.0, Real t_rho = 2.667, Real t_beta = 28.0) ;
                ModelErrorGenerator <3, Real> & modelErrorgenerator() ;
                void setIntegrationMethod(IntegrationMethod t_integrationMethod = RK4) ;

                void forward(Real t_dt, const Real * t_stateN, Real * t_stateNPP, bool addModelError = false) ;

            private:
                void computeDerivative(const Real * t_state) ;

                // Model parameters
                Real m_sigma ;
                Real m_rho ;
                Real m_beta ;

                // Model error
                ModelErrorGenerator <3, Real> m_meg ;

                // Integration method
                IntegrationMethod m_integrationMethod ;

                // Temporary variable for derivative
                Real m_dx[3] ;
                Real m_dxSum[3] ;
        };

}

#endif // MODEL_LORENZ63_H

