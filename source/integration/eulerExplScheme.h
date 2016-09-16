
/*__________________________________________________
 * integration/
 * eulerExplScheme.h
 *__________________________________________________
 * author        : colonel
 * last modified : 2016/9/16
 *__________________________________________________
 *
 * class to handle an integration function based on an Euler Explicit Scheme
 *
 */

#ifndef INTEGRATION_EULEREXPLSCHEME_H
#define INTEGRATION_EULEREXPLSCHEME_H

#include <cstddef>

namespace integration
{

    template <std::size_t N, typename Real>
        class EulerExplScheme
        {
            public:
                // apply scheme
                template <class Model>
                    void forward(Real t_dt, const Real * t_stateN, Real * t_stateNPP, Model & t_model) ;

            private:
                // Temporary variable for derivative
                Real m_dx[N] ;

        };

}

#endif // INTEGRATION_EULEREXPLSCHEME_H

