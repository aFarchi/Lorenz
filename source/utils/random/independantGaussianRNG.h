
/*__________________________________________________
 * utils/random/
 * independantGaussianRNG.h
 *__________________________________________________
 * author        : colonel
 * last modified : 2016/9/16
 *__________________________________________________
 *
 * class to handle a gaussian random number generator
 * when the covariance matrix is diagonal
 *
 */

#ifndef UTILS_RANDOM_INDEPENDANTGAUSSIANRNG_H
#define UTILS_RANDOM_INDEPENDANTGAUSSIANRNG_H

#include <random>

namespace utils
{

    namespace random
    {
        template <typename Real>
            using Normal = std::normal_distribution <Real> ;

        template <std::size_t N, typename Real = double, class RNG = std::default_random_engine>
            class IndependantGaussianRNG
            {
                public:
                    //IndependantGaussianRNG() ;
                    void setRNG(RNG * t_rng) ;
                    void setParameters(const Real * t_mean, const Real * t_var) ;

                    void addError(Real * t_vector) ;

                private:
                    // common random number generator
                    RNG * m_rng ;

                    // distribution
                    Normal <Real> m_distribution[N] ;

            };

    }

}

#endif // UTILS_RANDOM_INDEPENDANTGAUSSIANRNG_H

