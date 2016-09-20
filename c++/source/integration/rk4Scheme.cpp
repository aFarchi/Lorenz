
/*__________________________________________________
 * integration/
 * rk4Scheme.cpp
 *__________________________________________________
 * author        : colonel
 * last modified : 2016/9/16
 *__________________________________________________
 *
 * implementation of class RK4Scheme
 *
 */

#ifndef INTEGRATION_RK4SCHEME_CPP
#define INTEGRATION_RK4SCHEME_CPP

#include "rk4Scheme.h"
#include "../utils/vector/vectorOperations.h"

namespace integration
{

    template <std::size_t N, typename Real>
        template <class Model>
        void RK4Scheme <N, Real> ::
        forward(Real t_dt, const Real * t_stateN, Real * t_stateNPP, Model & t_model)
        {
            t_model.computeDerivative(t_stateN, m_dx) ;
            utils::vector::vectorScalarMultiplication(N, m_dxSum, m_dx, 1.0/6.0) ;
            utils::vector::vectorLinearCombination(N, t_stateNPP, t_stateN, m_dx, t_dt/2.0) ;

            t_model.computeDerivative(t_stateNPP, m_dx) ;
            utils::vector::vectorLinearCombination(N, m_dxSum, m_dxSum, m_dx, 1.0/3.0) ;
            utils::vector::vectorLinearCombination(N, t_stateNPP, t_stateN, m_dx, t_dt/2.0) ;

            t_model.computeDerivative(t_stateNPP, m_dx) ;
            utils::vector::vectorLinearCombination(N, m_dxSum, m_dxSum, m_dx, 1.0/3.0) ;
            utils::vector::vectorLinearCombination(N, t_stateNPP, t_stateN, m_dx, t_dt) ;

            t_model.computeDerivative(t_stateNPP, m_dx) ;
            utils::vector::vectorLinearCombination(N, m_dxSum, m_dxSum, m_dx, 1.0/6.0) ;
            utils::vector::vectorLinearCombination(N, t_stateNPP, t_stateN, m_dxSum, t_dt) ;
        }

}

#endif // INTEGRATION_RK2SCHEME_CPP

