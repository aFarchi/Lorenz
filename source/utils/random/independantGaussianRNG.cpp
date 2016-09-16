
/*__________________________________________________
 * utils/random/
 * independantGaussianRNG.cpp
 *__________________________________________________
 * author        : colonel
 * last modified : 2016/9/16
 *__________________________________________________
 *
 * implementation of class IndependantGaussianRNG
 *
 */

#ifndef UTILS_RANDOM_INDEPENDANTGAUSSIANRNG_CPP
#define UTILS_RANDOM_INDEPENDANTGAUSSIANRNG_CPP

#include "independantGaussianRNG.h"

namespace utils
{

    namespace random
    {

        // set common random number generator
        template <std::size_t N, typename Real, class RNG>
            void IndependantGaussianRNG<N, Real, RNG> ::
            setRNG(RNG * t_rng)
            {
                m_rng = t_rng ;
            }

        // set distribution parameters
        template <std::size_t N, typename Real, class RNG>
            void IndependantGaussianRNG<N, Real, RNG> ::
            setParameters(const Real * t_mean, const Real * t_var)
            {
                for(std::size_t i = 0 ; i < N ; ++i)
                {
                    m_distribution[i] = Normal <Real>(t_mean[i], t_var[i]) ;
                }
            }

        // add error to t_vector
        template <std::size_t N, typename Real, class RNG>
            void IndependantGaussianRNG<N, Real, RNG> ::
            addError(Real * t_vector)
            {
                for(std::size_t i = 0 ; i < N ; ++i)
                {
                    t_vector[i] += m_distribution[i](*m_rng) ;
                }
            }

    }

}

#endif // UTILS_RANDOM_INDEPENDANTGAUSSIANRNG_CPP

