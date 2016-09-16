
/*__________________________________________________
 * integration/
 * eulerExplScheme.cpp
 *__________________________________________________
 * author        : colonel
 * last modified : 2016/9/16
 *__________________________________________________
 *
 * implementation of class EulerExplScheme
 *
 */

#ifndef INTEGRATION_EULEREXPLSCHEME_CPP
#define INTEGRATION_EULEREXPLSCHEME_CPP

#include "eulerExplScheme.h"
#include "../utils/vector/vectorOperations.h"

namespace integration
{

    template <std::size_t N, typename Real>
        template <class Model>
        void EulerExplScheme<N, Real> ::
        forward(Real t_dt, const Real * t_stateN, Real * t_stateNPP, Model & t_model)
        {
            t_model.computeDerivative(t_stateN, m_dx) ;
            utils::vector::vectorLinearCombination(N, t_stateNPP, t_stateN, m_dx, t_dt) ;
        }

}

#endif // INTEGRATION_EULEREXPLSCHEME_CPP

