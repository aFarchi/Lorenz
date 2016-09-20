
/*__________________________________________________
 * integration/
 * rk4Scheme.h
 *__________________________________________________
 * author        : colonel
 * last modified : 2016/9/16
 *__________________________________________________
 *
 * class to handle an integration function based on the RK4 scheme
 *
 */

#ifndef INTEGRATION_RK4SCHEME_H
#define INTEGRATION_RK4SCHEME_H

#include <cstddef>

namespace integration
{

    template <std::size_t N, typename Real>
        class RK4Scheme
        {
            public:
                // apply scheme
                template <class Model>
                    void forward(Real t_dt, const Real * t_stateN, Real * t_stateNPP, Model & t_model) ;

            private:
                // Temporary variable for derivative
                Real m_dx[N] ;
                Real m_dxSum[N] ;

        };

}

#endif // INTEGRATION_RK4SCHEME_H

