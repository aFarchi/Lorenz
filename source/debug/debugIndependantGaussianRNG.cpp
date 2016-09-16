
/*__________________________________________________
 * debug/
 * debugIndependantGaussianRNG.cpp
 *__________________________________________________
 * author        : colonel
 * last modified : 2016/9/16
 *__________________________________________________
 *
 * code snippet to test class IndependantGaussianRNG
 * displays a sample of N(5,2)
 *
 */

#include <iostream>

#include "../utils/random/independantGaussianRNG.cpp"

using Real = double ;
using RNG  = std::default_random_engine ;
using GRNG = utils::random::IndependantGaussianRNG <2, Real, RNG> ;

int main()
{
    const int nrolls = 10000 ;
    const int nstars = 100 ;

    RNG commonRNG ;
    GRNG gaussianRNG ;

    Real mean[2] ;
    Real var[2] ;
    Real number[2] ;

    mean[0] = 5.0 ;
    var[0]  = 2.0 ;

    mean[1] = 5.0 ;
    var[1]  = 0.5 ;

    gaussianRNG.setRNG(&commonRNG) ;
    gaussianRNG.setParameters(mean, var) ;

    int p[2][10] ;
    for(std::size_t i = 0 ; i < 10 ; ++i)
    {
        p[0][i] = 0 ;
        p[1][i] = 0 ;
    }

    for(int i = 0 ; i < nrolls ; ++i)
    {
        number[0] = 0.0 ;
        number[1] = 0.0 ;
        gaussianRNG.addError(number) ;
        if(number[0] >= 0.0 and number[0] < 10.0)
        {
            ++p[0][static_cast<std::size_t>(number[0])] ;
        }
        if(number[1] >= 0.0 and number[1] < 10.0)
        {
            ++p[1][static_cast<std::size_t>(number[1])] ;
        }
    }

    for(std::size_t i = 0 ; i < 10 ; ++i)
    {
        std::cout << i << "-" << (i+1) << ": " ;
        std::cout << std::string(p[0][i]*nstars/nrolls, '*') << std::endl ;
    }
    for(std::size_t i = 0 ; i < 10 ; ++i)
    {
        std::cout << i << "-" << (i+1) << ": " ;
        std::cout << std::string(p[1][i]*nstars/nrolls, '*') << std::endl ;
    }

    return 0 ;

}

