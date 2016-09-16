
/*__________________________________________________
 * utils/vector/
 * vectorOperations.cpp
 *__________________________________________________
 * author        : colonel
 * last modified : 2016/9/16
 *__________________________________________________
 *
 * implementation of the vector functions
 *
 */

#ifndef UTILS_VECTOR_VECTOROPERATIONS_CPP
#define UTILS_VECTOR_VECTOROPERATIONS_CPP

#include "vectorOperations.h"

namespace utils
{

    namespace vector
    {

        // compute v = v1
        template <typename Real>
            void vectorCopy(std::size_t t_size, Real * t_v, const Real * t_v1)
            {
                for ( std::size_t i = 0 ; i < t_size ; ++i )
                {
                    t_v[i] = t_v1[i] ;
                }
            }

        // compute v = lambda . v1 
        template <typename Real>
            void vectorScalarMultiplication(std::size_t t_size, Real * t_v, const Real * t_v1, Real t_lambda)
            {
                for ( std::size_t i = 0 ; i < t_size ; ++i )
                {
                    t_v[i] = t_lambda * t_v1[i] ;
                }
            }

        // compute v = v1 + v2
        template <typename Real>
            void vectorAddition(std::size_t t_size, Real * t_v, const Real * t_v1, const Real * t_v2)
            {
                for ( std::size_t i = 0 ; i < t_size ; ++i )
                {
                    t_v[i] = t_v1[i] + t_v2[i] ;
                }
            }

        // compute v = v1 + lambda * v2
        template <typename Real>
            void vectorLinearCombination(std::size_t t_size, Real * t_v, const Real * t_v1, const Real * t_v2, Real t_lambda)
            {
                for ( std::size_t i = 0 ; i < t_size ; ++i )
                {
                    t_v[i] = t_v1[i] + t_lambda * t_v2[i] ;
                }
            }

        // stores v to file
        template <typename Real>
            void vectorToFile(std::size_t t_size, const Real* t_v, FILE* t_file)
            {
                fwrite(t_v, sizeof(Real), t_size, t_file) ;
            }

    }

}

#endif // UTILS_VECTOR_VECTOROPERATIONS_CPP

