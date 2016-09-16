
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
#include "../utils/vector/vectorOperations.h"

#include <iostream>

namespace model
{

    // constructor
    template <typename Real, template <std::size_t, typename> class ModelErrorGenerator>
        Lorenz63Model <Real, ModelErrorGenerator> ::
        Lorenz63Model() :
            m_sigma(10.0),
            m_rho(28.0),
            m_beta(2.667),
            m_meg(),
            m_integrationMethod(RK4)
        {
        }

    // set model parameters
    template <typename Real, template <std::size_t, typename> class ModelErrorGenerator>
        void Lorenz63Model <Real, ModelErrorGenerator> ::
        setModelParameters(Real t_sigma, Real t_rho, Real t_beta)
        {
            m_sigma = t_sigma ;
            m_rho   = t_rho ;
            m_beta  = t_beta ;
        }

    // access function for the model error generator
    // used to set the rng and the model error parameters
    template <typename Real, template <std::size_t, typename> class ModelErrorGenerator>
        ModelErrorGenerator <3, Real> & Lorenz63Model <Real, ModelErrorGenerator> ::
        modelErrorgenerator()
        {
            return m_meg ;
        }

    // set integration method
    template <typename Real, template <std::size_t, typename> class ModelErrorGenerator>
        void Lorenz63Model <Real, ModelErrorGenerator> ::
        setIntegrationMethod(IntegrationMethod t_integrationMethod)
        {
            m_integrationMethod = t_integrationMethod ;
        }

    // step forward
    // t_stateNPP = Model(t_stateN) + error
    template <typename Real, template <std::size_t, typename> class ModelErrorGenerator>
        void Lorenz63Model <Real, ModelErrorGenerator> ::
        forward(Real t_dt, const Real * t_stateN, Real * t_stateNPP, bool addModelError)
        {
            switch(m_integrationMethod)
            {
                case EulerExpl:
                    {
                        computeDerivative(t_stateN) ;
                        utils::vector::vectorLinearCombination(3, t_stateNPP, t_stateN, m_dx, t_dt) ;
                    }
                    break ;

                case RK2:
                    {
                        computeDerivative(t_stateN) ;
                        utils::vector::vectorLinearCombination(3, t_stateNPP, t_stateN, m_dx, t_dt/2.0) ;

                        computeDerivative(t_stateNPP) ;
                        utils::vector::vectorLinearCombination(3, t_stateNPP, t_stateN, m_dx, t_dt) ;
                    }
                    break ;

                case RK4:
                    {
                        computeDerivative(t_stateN) ;
                        utils::vector::vectorScalarMultiplication(3, m_dxSum, m_dx, 1.0/6.0) ;
                        utils::vector::vectorLinearCombination(3, t_stateNPP, t_stateN, m_dx, t_dt/2.0) ;
                        
                        computeDerivative(t_stateNPP) ;
                        utils::vector::vectorLinearCombination(3, m_dxSum, m_dxSum, m_dx, 1.0/3.0) ;
                        utils::vector::vectorLinearCombination(3, t_stateNPP, t_stateN, m_dx, t_dt/2.0) ;

                        computeDerivative(t_stateNPP) ;
                        utils::vector::vectorLinearCombination(3, m_dxSum, m_dxSum, m_dx, 1.0/3.0) ;
                        utils::vector::vectorLinearCombination(3, t_stateNPP, t_stateN, m_dx, t_dt) ;

                        computeDerivative(t_stateNPP) ;
                        utils::vector::vectorLinearCombination(3, m_dxSum, m_dxSum, m_dx, 1.0/6.0) ;
                        utils::vector::vectorLinearCombination(3, t_stateNPP, t_stateN, m_dxSum, t_dt) ;
                    }
                    break ;

                default:
                    break ;

            }

            if(addModelError)
            {
                m_meg.addError(t_stateNPP) ;
            }
        }

    // compute dx according to the model
    // result is stored in m_dx
    template <typename Real, template <std::size_t, typename> class ModelErrorGenerator>
        void Lorenz63Model <Real, ModelErrorGenerator> ::
        computeDerivative(const Real * t_state)
        {
            m_dx[0] = m_sigma * ( t_state[1] - t_state[0] ) ;
            m_dx[1] = ( m_rho - t_state[2] ) * t_state[0] - t_state[1] ;
            m_dx[2] = t_state[0] * t_state[1] - m_beta * t_state[2] ;
        }

}

#endif // MODEL_LORENZ63_CPP

