
/*__________________________________________________
 * debug/
 * debugLorenz63.cpp
 *__________________________________________________
 * author        : colonel
 * last modified : 2016/9/16
 *__________________________________________________
 *
 * code snippet to debug class Lorenz63Model
 *
 */

#include <iostream>

#include "../utils/random/independantGaussianRNG.cpp"
#include "../utils/vector/vectorOperations.cpp"
#include "../integration/eulerExplScheme.cpp"
#include "../integration/rk2scheme.cpp"
#include "../integration/rk4scheme.cpp"
#include "../model/lorenz63.cpp"

using Real   = double ;
using RNG    = std::default_random_engine ;
template <typename T>
using Normal = std::normal_distribution<T> ;
template <std::size_t N, typename T>
using GRNG   = utils::random::IndependantGaussianRNG <N, T, RNG, Normal> ;
template <std::size_t N, typename T>
using IS     = integration::RK4Scheme<N, T> ;
//using IS     = integration::RK2Scheme<N, T> ;
//using IS     = integration::EulerExplScheme<N, T> ;
using L63M   = model::Lorenz63Model<Real, IS, GRNG> ;

int main()
{

    // model
    const std::size_t Nx = 3 ;
    L63M model ;

    // model parameters
    Real sigma = 10.0 ;
    Real beta  = 8.0 / 3.0 ;
    Real rho   = 28.0 ;
    model.setModelParameters(sigma, rho, beta) ;

    // time sample parameters
    const Real dt = 0.01 ;
    const std::size_t Nt = 1000 ;

    // state and initial condition
    Real state[Nt][Nx] ;
    state[0][0] = 2.0 ;
    state[0][1] = 3.0 ;
    state[0][2] = 4.0 ;

    // model forward without error
    for(std::size_t i = 1 ; i < Nt ; ++i)
    {
        model.forward(dt, state[i-1], state[i], false) ;
    }

    // save state evolution into file
    std::string outputDir("/Users/aFarchi/Desktop/test/Lorenz/") ;
    FILE * output = fopen((outputDir+"state.bin").c_str(), "wb") ;
    for(std::size_t i = 0 ; i < Nt ; ++i)
    {
        utils::vector::vectorToFile(Nx, state[i], output) ;
    }
    fclose(output) ;

    return 0 ;

}

