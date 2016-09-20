
/*__________________________________________________
 * utils/vector/
 * vectorOperations.h
 *__________________________________________________
 * author        : colonel
 * last modified : 2016/9/16
 *__________________________________________________
 *
 * some functions to handle vectors (ie 1D arrays)
 *
 */

#ifndef UTILS_VECTOR_VECTOROPERATIONS_H
#define UTILS_VECTOR_VECTOROPERATIONS_H

#include <cstddef>
#include <stdio.h>

namespace utils
{

    namespace vector
    {

        // compute v = v1
        template <typename Real>
            void vectorCopy(std::size_t t_size, Real * t_v, const Real * t_v1) ;

        // compute v = lambda . v1 
        template <typename Real>
            void vectorScalarMultiplication(std::size_t t_size, Real * t_v, const Real * t_v1, Real t_lambda) ;

        // compute v = v1 + v2
        template <typename Real>
            void vectorAddition(std::size_t t_size, Real * t_v, const Real * t_v1, const Real * t_v2) ;

        // compute v = v1 + lambda * v2
        template <typename Real>
            void vectorLinearCombination(std::size_t t_size, Real * t_v, const Real * t_v1, const Real * t_v2, Real t_lambda) ;

        // stores v to file
        template <typename Real>
            void vectorToFile(std::size_t t_size, const Real* t_v, FILE* t_file) ;

    }

}

#endif // UTILS_VECTOR_VECTOROPERATIONS_H

