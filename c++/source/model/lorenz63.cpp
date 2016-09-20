
/*__________________________________________________
 * model/
 * lorenz63.cpp
 *__________________________________________________
 * author        : colonel
 * last modified : 2016/9/16
 *__________________________________________________
 *
 * implementation of class Lorenz63Model
 *
 */

#ifndef MODEL_LORENZ63_CPP
#define MODEL_LORENZ63_CPP

#include "lorenz63.h"

namespace model
{

    // constructor
    template <typename Real, template <std::size_t, typename> class Integrator, template <std::size_t, typename> class ModelErrorGenerator>
        Lorenz63Model <Real, Integrator, ModelErrorGenerator> ::
        Lorenz63Model() :
            m_sigma(10.0),
            m_rho(28.0),
            m_beta(2.667),
            m_meg(),
            m_integrator()
        {
        }

    // set model parameters
    template <typename Real, template <std::size_t, typename> class Integrator, template <std::size_t, typename> class ModelErrorGenerator>
        void Lorenz63Model <Real, Integrator, ModelErrorGenerator> ::
        setModelParameters(Real t_sigma, Real t_rho, Real t_beta)
        {
            m_sigma = t_sigma ;
            m_rho   = t_rho ;
            m_beta  = t_beta ;
        }

    // access function for the model error generator
    // used to set the rng and the model error parameters
    template <typename Real, template <std::size_t, typename> class Integrator, template <std::size_t, typename> class ModelErrorGenerator>
        ModelErrorGenerator <3, Real> & Lorenz63Model <Real, Integrator, ModelErrorGenerator> ::
        modelErrorGenerator()
        {
            return m_meg ;
        }

    // step forward
    // t_stateNPP = Model(t_stateN) + error
    // computed using the integration method
    template <typename Real, template <std::size_t, typename> class Integrator, template <std::size_t, typename> class ModelErrorGenerator>
        void Lorenz63Model <Real, Integrator, ModelErrorGenerator> ::
        forward(Real t_dt, const Real * t_stateN, Real * t_stateNPP, bool addModelError)
        {
            m_integrator.forward(t_dt, t_stateN, t_stateNPP, *this) ;

            if(addModelError)
            {
                m_meg.addError(t_stateNPP) ;
            }
        }

    // compute dx according to the model
    template <typename Real, template <std::size_t, typename> class Integrator, template <std::size_t, typename> class ModelErrorGenerator>
        void Lorenz63Model <Real, Integrator, ModelErrorGenerator> ::
        computeDerivative(const Real * t_state, Real * t_dstate)
        {
            t_dstate[0] = m_sigma * ( t_state[1] - t_state[0] ) ;
            t_dstate[1] = ( m_rho - t_state[2] ) * t_state[0] - t_state[1] ;
            t_dstate[2] = t_state[0] * t_state[1] - m_beta * t_state[2] ;
        }

}

#endif // MODEL_LORENZ63_CPP

